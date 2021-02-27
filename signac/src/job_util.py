"""Module containing various utility job-related functions."""
import itertools
import logging
import pathlib

import numpy as np
from openpmd_viewer.addons import LpaDiagnostics

import util

logger = logging.getLogger(__name__)

def get_key_values(project, key, key_class=float):
    schema = project.detect_schema()
    return sorted(schema[key][key_class])


def is_h5_path(job):
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    if not h5_path.is_dir():
        raise FileNotFoundError(f"{h5_path} doesn't exist.")
    return h5_path


def get_time_series_from(job):
    h5_path = is_h5_path(job)
    time_series = LpaDiagnostics(h5_path)
    logger.info("Read time series from %s." % h5_path)
    return time_series


def saved_iterations(job):
    return np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=int)


def estimate_diags_fnames(job, digits=8, extension=".h5"):
    return (
        f"data{iteration:0{digits}d}{extension}" for iteration in saved_iterations(job)
    )


def extract_iteration_number(diags_fname):
    stem, ext = diags_fname.split(".")
    return int(stem[4:])


def get_diags_fnames(job):
    h5_path = is_h5_path(job)
    fnames = (
        p.name
        for p in sorted(
            h5_path.glob("*.h5"), key=lambda p: extract_iteration_number(p.name)
        )
    )
    return fnames


def estimated_time_of_arrival(job):
    try:
        h5_path = is_h5_path(job)
    except FileNotFoundError:
        return "∞"
    paths = h5_path.glob("*.h5")
    paths1, paths2 = itertools.tee(paths, 2)
    if len(list(paths1)) < 2:
        return "∞"
    (oldest, t_old), (newest, t_new) = util.oldest_newest(paths2)
    delta_t = t_new - t_old

    it = np.array(tuple(extract_iteration_number(p.name) for p in (oldest, newest)))
    delta_it = np.diff(it).item()
    final_iteration = job.sp.N_step - 1

    remaining_iterations = final_iteration - it[1]
    runtime = remaining_iterations * delta_t / delta_it

    return str(t_new + runtime).split(".")[0]



def main():
    """Main entry point."""
    import random

    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    job = random.choice(list(iter(proj)))
    print(job.ws)

    edf = list(estimate_diags_fnames(job))
    gdf = list(get_diags_fnames(job))
    assert set(edf) == set(gdf)


if __name__ == "__main__":
    main()
