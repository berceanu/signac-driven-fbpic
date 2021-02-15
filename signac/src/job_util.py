"""Module containing various utility job-related functions."""
from openpmd_viewer.addons import LpaDiagnostics
import pathlib
import math

def get_time_series_from(job):
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path)
    return time_series


def number_of_saved_iterations(job):
    return math.ceil((job.sp.N_step - 0) / job.sp.diag_period)


def main():
    """Main entry point."""
    import random
    import signac

    random.seed(24)

    proj = signac.get_project(search=False)
    job = random.choice(list(iter(proj)))
    print(job)    

    ts = get_time_series_from(job)
    print(ts.iterations.size)
    print(number_of_saved_iterations(job))

if __name__ == '__main__':
    main()
