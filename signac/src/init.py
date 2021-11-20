"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import math
import pathlib
from dataclasses import dataclass

import numpy as np
import signac
import unyt as u
from prepic import Plasma, lwfa
from small import L_vacuum

import util

# The number of output hdf5 files, such that Nz * Nr * NUMBER_OF_H5 * size(float64)
# easily fits in RAM
NUMBER_OF_H5 = 33
SQRT_FACTOR = math.sqrt(2 * math.log(2))


@dataclass
class LaserChirp:
    """Class for computing GDD, chirp and laser duration."""

    tau: float  # laser duration, in seconds, converted from experimental FWHM@intensity
    chirp_stretching: float = 1.001  # pulse lengthening due to chirp
    chirp_sign: int = -1  # negative chirp

    @property
    def ctau(self) -> float:
        """Pulse length."""
        return u.clight.to_value("m/s") * self.tau

    @property
    def ctau_stretched(self) -> float:
        return self.ctau * self.chirp_stretching

    @property
    def gdd(self) -> float:
        return (
            self.chirp_sign
            * 0.5
            * self.tau ** 2
            * math.sqrt(self.chirp_stretching ** 2 - 1)
        )


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
        [
            2000.0e-6,
        ]
    )

    for focal_plane in focal_positions:
        sp = dict(
            random_seed=42,  # deterministic random seed
            # TODO: move to job document
            nranks=4,  # number of MPI ranks (default 4); it's also the number of GPUs used per job
            # The simulation box
            lambda0=0.815e-6,  # Laser wavelength (default 0.815e-6)
            lambda0_over_dz=24,  # Δz = lambda0 / lambda0_over_dz (default 32)
            dr_over_dz=3,  # Δr = dr_over_dz * Δz (default 5)
            r_boundary_conditions="reflective",  #  'reflective' (default) / 'open' more expensive
            n_order=64,  # Order of the stencil for z derivatives in the Maxwell solver (-1, 32 default, 16)
            Nm=3,  # Number of modes used (default 3)
            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=0.0e-6,
            n_e=0.9 * 1.0e18 * 1.0e6,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z (default 2)
            p_nr=3,  # Number of particles per cell along r (default 2)
            # The laser
            a0=5.0,  # Laser amplitude
            # Laser waist, converted from experimental FWHM@intensity
            w0=22.0e-6,
            # Laser duration, converted from experimental FWHM@intensity
            tau=29.0e-15 / SQRT_FACTOR,
            z0=0.0e-6,  # Laser centroid
            # TODO is the laser focal plane measured from the nozzle center?
            zfoc=focal_plane,  # Laser focal position, measured from the center of the gas jet
            # The density profile
            ramp_up=1000.0e-6,
            plateau=20e3 * 1.0e-6,
            ramp_down=1000.0e-6,
            L_vacuum=1000.0e-6,  # length of vacuum region after the plasma
            current_correction="curl-free",  # "curl-free" (default, faster) or "cross-deposition" (more local)
            # channel guiding
            w_matched=400000000000000000.0e-6,
            # do not change below this line ##############
            rel_delta_n_over_w2=None,  # relative change divided by w_matched^2 that allows guiding
            zmin=None,  # Left end of the simulation box (meters)
            zmax=None,  # Right end of the simulation box (meters)
            rmax=None,  # Length of the box along r (meters) (default 70.0e-6)
            phi2_chirp=None,  # GDD
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
        laser_chirp = LaserChirp(sp["tau"])
        plasma = Plasma(n_pe=sp["n_e"] * u.meter ** (-3))

        sp["phi2_chirp"] = laser_chirp.gdd

        sp["zmax"] = 2.2 * laser_chirp.ctau_stretched
        sp["zmin"] = -2 * laser_chirp.ctau_stretched - 1.5 * plasma.λp
        sp["rmax"] = 5 * sp["w0"]

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

        sp["rel_delta_n_over_w2"] = 1.0 / (
            math.pi * 2.81e-15 * sp["w_matched"] ** 4 * sp["n_e"]
        )

        sp["p_zmax"] = sp["ramp_up"] + sp["plateau"] + sp["ramp_down"]

        sp["L_interact"] = sp["p_zmax"] - sp["p_zmin"] + sp["L_vacuum"]
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

        gamma_wake = math.sqrt(job.sp.n_c / job.sp.n_e)
        job.doc.setdefault("Γwake", f"{gamma_wake:.3f}")

        p = pathlib.Path(job.ws)
        for folder in ("rhos", "phasespaces"):
            pathlib.Path(p / folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
