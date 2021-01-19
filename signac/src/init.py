#!/usr/bin/env python3
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


def main():
    """Main function, for defining the parameter(s) to be varied in the simulations."""
    project = signac.init_project(
        "fbpic-project",
        workspace="/scratch/berceanu/runs/signac-driven-fbpic/workspace_lwfa/",
    )

    x_from_center = np.array([200, 600, 1000, 1400, 1800]) * 1e-6

    for focus in nozzle_center_offset(x_from_center):
        sp = dict(
            # The simulation box
            Nz=2048,  # Number of gridpoints along z
            zmin=-100.0e-6,  # Left end of the simulation box (meters)
            zmax=0.0e-6,  # Right end of the simulation box (meters)
            Nr=256,  # Number of gridpoints along r
            rmax=25.0e-6,  # Length of the box along r (meters)
            Nm=3,  # Number of modes used
            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=0.0e-6,
            # Maximal radial position of the plasma (meters)
            p_rmax=25.0e-6,
            n_e=8.0e18 * 1.0e6,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z
            p_nr=2,  # Number of particles per cell along r
            p_nt=None,  # Number of particles per cell along theta, should be 4*Nm
            # The laser
            a0=2.4,  # Laser amplitude
            w0=13.6e-6,  # Laser waist, converted from experimental FWHM@intensity
            tau=27.8e-15
            / SQRT_FACTOR,  # Laser duration, converted from experimental FWHM@intensity
            z0=-10.0e-6,  # Laser centroid
            zfoc=focus,  # Focal position
            lambda0=0.815e-6,  # Laser wavelength
            profile_flatness=300,  # Flatness of laser profile far from focus (larger means flatter)
            # The density profile
            flat_top_dist=1000.0e-6,  # plasma flat top distance
            sigma_right=500.0e-6,
            center_left=1000.0e-6,
            sigma_left=500.0e-6,
            power=2.0,
            # do not change below this line ##############
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
        sp["p_nt"] = 4 * sp["Nm"]

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

    project.write_statepoints()

    for job in project:
        Δz = ((job.sp.zmax - job.sp.zmin) / job.sp.Nz * u.meter).to(u.micrometer)
        Δr = (job.sp.rmax / job.sp.Nr * u.meter).to(u.micrometer)

        job.doc.setdefault("Δz", f"{Δz:.3f}")
        job.doc.setdefault("Δr", f"{Δr:.3f}")

        plasma = Plasma(n_pe=job.sp.n_e * u.meter ** (-3))
        job.doc.setdefault("λp", f"{plasma.λp:.3f}")

        p = pathlib.Path(job.ws)
        for folder in ("rhos", "phasespaces"):
            pathlib.Path(p / folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
