"""Module containing various utility job-related functions."""
from openpmd_viewer.addons import LpaDiagnostics
import pathlib
import math
import numpy as np


def is_h5_path(job):
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    if not h5_path.is_dir():
        raise FileNotFoundError(f"{h5_path} doesn't exist.")
    return h5_path

def get_time_series_from(job):
    h5_path = is_h5_path(job)
    time_series = LpaDiagnostics(h5_path)
    return time_series


def num_saved_iterations(N_step, diag_period):
    return math.ceil((N_step - 1) / diag_period)


def number_of_saved_iterations(job):
    return num_saved_iterations(job.sp.N_step, job.sp.diag_period)


def saved_iterations(N_step, diag_period):
    return np.arange(0, N_step, diag_period, dtype=int)


def estimate_saved_iterations(job):
    return saved_iterations(job.sp.N_step, job.sp.diag_period)


def estimate_diags_fnames(job, digits=8, extension=".h5"):
    return (
        f"data{iteration:0{digits}d}{extension}"
        for iteration in estimate_saved_iterations(job)
    )


def get_diags_fnames(job):
    h5_path = is_h5_path(job)
    fnames = (
        p.name
        for p in sorted(
            h5_path.glob("*.h5"), key=lambda p: extract_iteration_number(p.name)
        )
    )
    return fnames


def extract_iteration_number(diags_fname):
    stem, ext = diags_fname.split(".")
    return int(stem[4:])


def main():
    """Main entry point."""
    import random
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    job = random.choice(list(iter(proj)))
    print(job.ws)

    ts = get_time_series_from(job)
    assert number_of_saved_iterations(job) == estimate_saved_iterations(job).size
    assert ts.iterations.size == number_of_saved_iterations(job)

    edf = list(estimate_diags_fnames(job))
    other_esi = [extract_iteration_number(f) for f in edf]
    esi = estimate_saved_iterations(job)
    assert set(esi) == set(other_esi)

    gdf = list(get_diags_fnames(job))
    assert set(edf) == set(gdf)


if __name__ == "__main__":
    main()
