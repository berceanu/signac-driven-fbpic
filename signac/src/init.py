#!/usr/bin/env python3
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import pathlib

import math
import shutil
import unyt as u
import signac
import numpy as np

# The number of output hdf5 files, such that Nz * Nr * NUMBER_OF_H5 * size(float64)
# easily fits in RAM
NUMBER_OF_H5 = 200


def main():
    """Main function, for defining the parameter(s) to be varied in the simulations."""
    project = signac.init_project(
        "fbpic-project",
        workspace="/scratch/berceanu/runs/signac-driven-fbpic/workspace/",
    )

    for ne in np.linspace(6, 4e2, 16) * 1e13 * 1e6:
        sp = dict(
            # The simulation box
            Nz=4096,  # Number of gridpoints along z
            zmin=-4000.0e-6,  # Left end of the simulation box (meters)
            zmax=-200.0e-6,  # Right end of the simulation box (meters)
            Nr=512,  # Number of gridpoints along r
            rmax=300.0e-6,  # Length of the box along r (meters)
            Nm=4,  # Number of modes used
            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=-100.0e-6,
            # Maximal radial position of the plasma (meters)
            p_rmax=290.0e-6,
            n_e=ne,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z
            p_nr=2,  # Number of particles per cell along r
            p_nt=6,  # Number of particles per cell along theta
            # do not change below this line ##############
            p_zmax=68400.0e-6,  # Position of the end of the plasma (meters)
            # The density profile
            ramp_start=-100.0e-6,
            ramp_length=100.0e-6,  # increase (up to `p_zmax`) !
            # The interaction length of the simulation (meters)
            # increase (up to `p_zmax`) to simulate longer distance!
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
        p = pathlib.Path(job.ws)
        pathlib.Path(p / "rhos").mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
