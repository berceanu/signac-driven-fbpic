"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import math
import pathlib

import numpy as np
import signac
import unyt as u
from prepic import Plasma, lwfa

import util

# The number of output hdf5 files, such that Nz * Nr * NUMBER_OF_H5 * size(float64)
# easily fits in RAM
NUMBER_OF_H5 = 50
SQRT_FACTOR = math.sqrt(2 * math.log(2))


def get_dz(zmax, zmin, Nz):
    """Longitudinal resolution (meters)."""
    return (zmax - zmin) / Nz


def main():
    """Main function, for defining the parameter(s) to be varied in the simulations."""
    project = signac.init_project(
        "fbpic-project",
        workspace="/scratch/berceanu/runs/signac-driven-fbpic/workspace_lwfa/",
    )

    focal_positions = np.array(
        [200e-6, 400e-6, 600e-6, 800e-6, 1000e-6, 1200e-6, 1400e-6, 1600e-6, 1800e-6]
    )

    for zfoc in focal_positions:
        sp = dict(
            random_seed=42,  # deterministic random seed
            # TODO: move to job document
            nranks=4,  # number of MPI ranks (default 4); it's also the number of GPUs used per job
            # The simulation box
            lambda0=0.8e-6,  # Laser wavelength (default 0.815e-6)
            lambda0_over_dz=24,  # Δz = lambda0 / lambda0_over_dz (default 32)
            dr_over_dz=10,  # Δr = dr_over_dz * Δz (default 5)
            zmin=-60.0e-6,  # Left end of the simulation box (meters)
            zmax=0.0e-6,  # Right end of the simulation box (meters)
            rmax=70.0e-6,  # Length of the box along r (meters) (default 70.0e-6)
            r_boundary_conditions="reflective",  #  'reflective' (default) / 'open' more expensive
            n_order=32,  # Order of the stencil for z derivatives in the Maxwell solver (-1, 32 default, 16)
            Nm=3,  # Number of modes used (default 3)
            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=0.0e-6,
            n_e=8.0 * 1.0e18 * 1.0e6,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z (default 2)
            p_nr=2,  # Number of particles per cell along r (default 2)
            # The laser
            a0=2.4,  # Laser amplitude
            # Laser waist, converted from experimental FWHM@intensity
            w0=22.0e-6 / SQRT_FACTOR,
            # Laser duration, converted from experimental FWHM@intensity
            tau=25.0e-15 / SQRT_FACTOR,
            z0=-10.0e-6,  # Laser centroid
            zfoc_from_nozzle_center=zfoc,  # Laser focal position, measured from the center of the gas jet
            profile_flatness=6,  # Flatness of laser profile far from focus (larger means flatter) (default 100)
            # The density profile
            flat_top_dist=0.0e-6,  # plasma flat top distance
            sigma_right=1471.0e-6,
            center_left=3000.0e-6,
            sigma_left=1471.0e-6,
            power=1.8,
            current_correction="curl-free",  # "curl-free" (default, faster) or "cross-deposition" (more local)
            # do not change below this line ##############
            Nz=None,  # Number of gridpoints along z
            Nr=None,  # Number of gridpoints along r
            p_rmax=None,  # Maximal radial position of the plasma (meters)
            p_nt=None,  # Number of particles per cell along theta (default 4*Nm)
            n_c=None,  # critical plasma density for this laser (electrons.meters^-3)
            center_right=None,
            p_zmax=None,  # Position of the end of the plasma (meters)
            L_interact=None,
            # Period in number of timesteps
            diag_period=None,
            # Timestep (seconds)
            dt=None,
            # Interaction time (seconds) (to calculate number of PIC iterations)
            # (i.e. the time it takes for the moving window to slide across the plasma)
            T_interact=None,
            # Number of iterations to perform
            N_step=None,
        )
        laser = lwfa.Laser.from_a0(
            a0=sp["a0"] * u.dimensionless,
            τL=sp["tau"] * u.second,
            beam=lwfa.GaussianBeam(w0=sp["w0"] * u.meter, λL=sp["lambda0"] * u.meter),
        )
        sp["n_c"] = laser.ncrit.to_value("1/m**3")
        sp["E0"] = (laser.E0 / sp["a0"]).to_value("volt/m")
        sp["zR"] = laser.beam.zR.to_value("m")

        sp["Nz"] = int(
            (sp["zmax"] - sp["zmin"]) * sp["lambda0_over_dz"] / sp["lambda0"]
        )
        dz = get_dz(sp["zmax"], sp["zmin"], sp["Nz"])
        dr = sp["dr_over_dz"] * dz
        sp["Nr"] = int(sp["rmax"] / dr)

        sp["p_nt"] = 4 * sp["Nm"]
        sp["p_rmax"] = 0.9 * sp["rmax"]
        # Laser focal position
        sp["zfoc"] = util.nozzle_center_offset(sp["zfoc_from_nozzle_center"])

        sp["center_right"] = sp["center_left"] + sp["flat_top_dist"]
        sp["p_zmax"] = sp["center_right"] + 2 * sp["sigma_right"]

        sp["L_interact"] = sp["p_zmax"] - sp["p_zmin"]
        sp["dt"] = (sp["zmax"] - sp["zmin"]) / sp["Nz"] / u.clight.to_value("m/s")
        sp["T_interact"] = (
            sp["L_interact"] + (sp["zmax"] - sp["zmin"])
        ) / u.clight.to_value("m/s")
        sp["N_step"] = int(sp["T_interact"] / sp["dt"])
        sp["N_step"] = util.round_to_nearest(sp["N_step"], base=NUMBER_OF_H5) + 1
        sp["diag_period"] = (sp["N_step"] - 1) // NUMBER_OF_H5
        project.open_job(sp).init()

    for job in project:
        Δz = (get_dz(job.sp.zmax, job.sp.zmin, job.sp.Nz) * u.meter).to(u.micrometer)
        Δr = (job.sp.rmax / job.sp.Nr * u.meter).to(u.micrometer)

        job.doc.setdefault("Δz", f"{Δz:.3f}")
        job.doc.setdefault("Δr", f"{Δr:.3f}")

        count = job.sp.p_nt * job.sp.p_nr * job.sp.p_nz * job.sp.Nz * job.sp.Nr
        job.doc.setdefault("macroparticle_count", f"{count:.2e}")

        plasma = Plasma(n_pe=job.sp.n_e * u.meter ** (-3))
        job.doc.setdefault("λp", f"{plasma.λp:.3f}")

        p = pathlib.Path(job.ws)
        for folder in ("rhos", "phasespaces"):
            pathlib.Path(p / folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
