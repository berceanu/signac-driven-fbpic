#!/usr/bin/env python3
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging

# import numpy as np
import signac
from scipy.constants import physical_constants

# TODO test small sims on parallel multi-cores
# TODO adapt parameters to large injection run
# TODO test multiple large injection runs on parallel GPUs @ CETAL

c_light = physical_constants["speed of light in vacuum"][0]


def main():
    project = signac.init_project("fbpic-minimal-project")

    for Nm in (2, 3, 4):
        sp = dict(
            # The simulation box
            Nz=800,  # Number of gridpoints along z
            zmin=-10.0e-6,  # Left end of the simulation box (meters)
            zmax=30.0e-6,  # Right end of the simulation box (meters)
            Nr=50,  # Number of gridpoints along r
            rmax=20.0e-6,  # Length of the box along r (meters)
            Nm=Nm,  # Number of modes used

            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=25.0e-6,
            # Maximal radial position of the plasma (meters)
            p_rmax=18.0e-6,
            n_e=4.0e18 * 1.0e6,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z
            p_nr=2,  # Number of particles per cell along r
            p_nt=4,  # Number of particles per cell along theta

            # The laser
            a0=4.0,  # Laser amplitude
            w0=5.0e-6,  # Laser waist
            ctau=5.0e-6,  # Laser duration
            z0=15.0e-6,  # Laser centroid


            # do not change below this line #
            p_zmax=500.0e-6,  # Position of the end of the plasma (meters)
            # Minimal radial position of the plasma (meters)
            p_rmin=0.0,

            # The density profile
            ramp_start=30.0e-6,
            ramp_length=40.0e-6,  # increase (up to `p_zmax`) !

            # The interaction length of the simulation (meters)
            # increase (up to `p_zmax`) to simulate longer distance!
            L_interact=0.0e-6,  # 50.e-6 in fbpic LWFA example

            # Period in number of timesteps
            # change to 100 for long simulations!
            diag_period=None,
            # Timestep (seconds)
            dt=None,
            # Interaction time (seconds) (to calculate number of PIC iterations)
            # (i.e. the time it takes for the moving window to slide across the plasma)
            T_interact=None,
            # Number of iterations to perform
            N_step=None,
        )

        sp["dt"] = (sp["zmax"] - sp["zmin"]) / sp["Nz"] / c_light
        sp["T_interact"] = (sp["L_interact"] + (sp["zmax"] - sp["zmin"])) / c_light
        sp["N_step"] = int(sp["T_interact"] / sp["dt"])
        sp["diag_period"] = int(sp["N_step"] / 4)

        project.open_job(sp).init()

    project.write_statepoints()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
