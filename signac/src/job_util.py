"""Module containing various utility job-related functions."""
from openpmd_viewer.addons import LpaDiagnostics
import pathlib
import math
import numpy as np
from util import round_to_nearest
import logging


logger = logging.getLogger(__name__)


def get_time_series_from(job):
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path)
    logger.info("Read time series from %s." % h5_path)
    return time_series


def num_saved_iterations(N_step, diag_period):
    return math.ceil((N_step - 0) / diag_period)


def number_of_saved_iterations(job):
    return num_saved_iterations(job.sp.N_step, job.sp.diag_period)


def saved_iterations(N_step, diag_period):
    return np.arange(0, N_step, diag_period, dtype=int)


def estimate_saved_iterations(job):
    return saved_iterations(job.sp.N_step, job.sp.diag_period)


def main():
    """Main entry point."""
    import random
    import signac

    random.seed(24)

    proj = signac.get_project(search=False)
    job = random.choice(list(iter(proj)))
    print(job)

    ts = get_time_series_from(job)


if __name__ == "__main__":
    main()
