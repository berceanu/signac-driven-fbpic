"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import pathlib
import math
import numpy as np

import unyt as u
from prepic import Plasma, lwfa
import signac
from util import nozzle_center_offset

# The number of output hdf5 files, such that Nz * Nr * NUMBER_OF_H5 * size(float64)
# easily fits in RAM
NUMBER_OF_H5 = 200
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

    for _ in range(1):  # placeholder FIXME
        sp = dict(
            # The simulation box
            z_rezolution_factor=32,  # Δz = lambda0 / z_rezolution_factor (default 20)
            zmin=-100.0e-6,  # Left end of the simulation box (meters)
            zmax=0.0e-6,  # Right end of the simulation box (meters)
            rmax=70.0e-6,  # Length of the box along r (meters)
            r_boundary_conditions="reflective",  #  'reflective' (default) / 'open' more expensive
            n_order=32,  # Order of the stencil for z derivatives in the Maxwell solver
            Nm=3,  # Number of modes used
            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=0.0e-6,
            n_e=8.0e18 * 1.0e6,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z
            p_nr=2,  # Number of particles per cell along r
            # The laser
            a0=2.4,  # Laser amplitude
            w0=22.0e-6
            / SQRT_FACTOR,  # Laser waist, converted from experimental FWHM@intensity
            tau=25.0e-15
            / SQRT_FACTOR,  # Laser duration, converted from experimental FWHM@intensity
            z0=-10.0e-6,  # Laser centroid
            zfoc=nozzle_center_offset(1400e-6),  # Focal position
            lambda0=0.815e-6,  # Laser wavelength
            profile_flatness=6,  # Flatness of laser profile far from focus (larger means flatter)
            # The density profile
            flat_top_dist=1000.0e-6,  # plasma flat top distance
            sigma_right=500.0e-6,
            center_left=1000.0e-6,
            sigma_left=500.0e-6,
            power=2.0,
            # do not change below this line ##############
            Nz=None,  # Number of gridpoints along z
            Nr=None,  # Number of gridpoints along r
            p_rmax=None,  # Maximal radial position of the plasma (meters)
            p_nt=None,  # Number of particles per cell along theta, should be 4*Nm
            n_c=None,  # critical plasma density for this laser (electrons.meters^-3)
            center_right=None,
            p_zmax=None,  # Position of the end of the plasma (meters)
            L_interact=None,
            # Period in number of timesteps
            diag_period=None,
            # TODO add electron tracking period
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
            (sp["zmax"] - sp["zmin"]) * sp["z_rezolution_factor"] / sp["lambda0"]
        )
        dz = get_dz(sp["zmax"], sp["zmin"], sp["Nz"])
        dr = 5 * dz
        sp["Nr"] = int(sp["rmax"] / dr)

        sp["p_nt"] = 4 * sp["Nm"]
        sp["p_rmax"] = sp["rmax"]

        sp["center_right"] = sp["center_left"] + sp["flat_top_dist"]
        sp["p_zmax"] = sp["center_right"] + 2 * sp["sigma_right"]

        sp["L_interact"] = sp["p_zmax"] - sp["p_zmin"]
        sp["dt"] = (sp["zmax"] - sp["zmin"]) / sp["Nz"] / u.clight.to_value("m/s")
        sp["T_interact"] = (
            sp["L_interact"] + (sp["zmax"] - sp["zmin"])
        ) / u.clight.to_value("m/s")
        sp["N_step"] = int(sp["T_interact"] / sp["dt"])
        sp["diag_period"] = math.ceil(sp["N_step"] / NUMBER_OF_H5)

        project.open_job(sp).init()

    for job in project:
        Δz = (get_dz(job.sp.zmax, job.sp.zmin, job.sp.Nz) * u.meter).to(u.micrometer)
        Δr = (job.sp.rmax / job.sp.Nr * u.meter).to(u.micrometer)

        job.doc.setdefault("Δz", f"{Δz:.3f}")
        job.doc.setdefault("Δr", f"{Δr:.3f}")

        plasma = Plasma(n_pe=job.sp.n_e * u.meter ** (-3))
        job.doc.setdefault("λp", f"{plasma.λp:.3f}")

        job.doc.setdefault("x", nozzle_center_offset(job.sp.zfoc))

        p = pathlib.Path(job.ws)
        for folder in ("rhos", "phasespaces"):
            pathlib.Path(p / folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
