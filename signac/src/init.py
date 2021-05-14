#!/usr/bin/env python3
"""Initialize the project's data space.

Iterates over all defined state points and initializes
the associated job workspace directories."""
import logging
import pathlib
import math
import numpy as np

import unyt as u
from prepic import Plasma
import signac

# The number of output hdf5 files, such that Nz * Nr * NUMBER_OF_H5 * size(float64)
# easily fits in RAM
NUMBER_OF_H5 = 200


def main():
    """Main function, for defining the parameter(s) to be varied in the simulations."""
    project = signac.init_project(
        "fbpic-project",
        workspace="/scratch/berceanu/runs/signac-driven-fbpic/workspace_pwfa/",
    )

    for ne in np.geomspace(2e15, 6e16, 16) * 1e6:
        sp = dict(
            # The simulation box
            Nz=512,  # Number of gridpoints along z
            zmin=-900.0e-6,  # Left end of the simulation box (meters)
            zmax=900.0e-6,  # Right end of the simulation box (meters)
            Nr=256,  # Number of gridpoints along r
            rmax=800.0e-6,  # Length of the box along r (meters)
            Nm=3,  # Number of modes
            # The particles
            # Position of the beginning of the plasma (meters)
            p_zmin=1000.0e-6,
            # Maximal radial position of the plasma (meters)
            p_rmax=400.0e-6,
            n_e=ne,  # Density (electrons.meters^-3)
            p_nz=2,  # Number of particles per cell along z
            p_nr=2,  # Number of particles per cell along r
            p_nt=12,  # Number of particles per cell along theta, should be 4*Nm
            # do not change below this line ##############
            p_zmax=69500.0e-6,  # Position of the end of the plasma (meters)
            # The density profile
            ramp_start=1000.0e-6,
            ramp_length=100.0e-6,  # increase (up to `p_zmax`) !
            # The injected electron bunch
            bunch_charge=-200.0e-12,  # Charge in Coulomb
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
        sp["L_interact"] = sp["p_zmax"]
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

        for txt_file in ("density_1_inlet_spacers.txt", "exp_4deg.txt"):
            src = pathlib.Path(txt_file)
            dest = pathlib.Path(job.fn(txt_file))
            dest.write_text(src.read_text())

        p = pathlib.Path(job.ws)
        for folder in ("rhos", "centroids"):
            pathlib.Path(p / folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
