#!/usr/bin/env python3
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging

import prepic.lwfa as lwfa
import signac
import unyt as u
import numpy as np


# TODO use scaling laws to estimate some of the input params
# todo remove `deprecated.py` and `init_minimal.py`

def main():
    """Main function, for defining the parameter(s) to be varied in the simulations."""
    project = signac.init_project("fbpic-project")

    for a0 in np.linspace(start=0.5, stop=5.0, num=6):
        sp = dict(
            # The simulation box
            Nz=4096,  # Number of gridpoints along z
            zmin=-70.0e-6,  # Left end of the simulation box (meters)
            zmax=30.0e-6,  # Right end of the simulation box (meters)
            Nr=256,  # Number of gridpoints along r
            rmax=30.0e-6,  # Length of the box along r (meters)
            Nm=2,  # Number of modes used
            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=0.0e-6,
            # Maximal radial position of the plasma (meters)
            p_rmax=27.0e-6,
            n_e=7.5e18 * 1.0e6,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z
            p_nr=2,  # Number of particles per cell along r
            p_nt=4,  # Number of particles per cell along theta
            # The laser
            a0=a0,  # Laser amplitude
            w0=9.0e-6,  # Laser waist
            ctau=9.0e-6,  # Laser duration
            z0=0.0e-6,  # Laser centroid
            lambda0=0.8e-6,  # Laser wavelength (meters)
            n_c=None,  # critical plasma density for this laser (electrons.meters^-3)
            # do not change below this line ##############
            p_zmax=2250.0e-6,  # Position of the end of the plasma (meters)
            # Minimal radial position of the plasma (meters)
            p_rmin=0.0,
            # The density profile
            ramp_start=0.0e-6,
            ramp_length=375.0e-6,  # increase (up to `p_zmax`) !
            # The interaction length of the simulation (meters)
            # increase (up to `p_zmax`) to simulate longer distance!
            L_interact=None,
            # Period in number of timesteps
            diag_period=100,
            # Timestep (seconds)
            dt=None,
            # Interaction time (seconds) (to calculate number of PIC iterations)
            # (i.e. the time it takes for the moving window to slide across the plasma)
            T_interact=None,
            # Number of iterations to perform
            N_step=None,
        )

        laser = lwfa.Laser.from_a0(
            a0=sp["a0"],
            τL=(sp["ctau"] * u.meter) / u.clight,
            beam=lwfa.GaussianBeam(w0=sp["w0"] * u.meter, λL=sp["lambda0"] * u.meter),
        )
        sp["n_c"] = laser.ncrit.to_value('1/m**3')

        sp["L_interact"] = 900.0e-6 - (sp["zmax"] - sp["zmin"])
        sp["dt"] = (sp["zmax"] - sp["zmin"]) / sp["Nz"] / u.clight.to_value('m/s')
        sp["T_interact"] = (sp["L_interact"] + (sp["zmax"] - sp["zmin"])) / u.clight.to_value('m/s')
        sp["N_step"] = int(sp["T_interact"] / sp["dt"])

        project.open_job(sp).init()

    project.write_statepoints()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
