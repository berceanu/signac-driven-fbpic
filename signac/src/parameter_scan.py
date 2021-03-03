"""
Computes the electron energy spectra for the whole parameter space and loads it
into the N-dimensional array class.
"""
from itertools import product

import signac
import xarray as xr
import numpy as np

import job_util
from energy_histograms import job_energy_histogram
from xarray_spectra import XSpectra


def two_parameters_study(project, keys=("a0", "n_e")):

    vy = job_util.get_key_values(project, keys[0])
    vx = job_util.get_key_values(project, keys[1])

    spectra = list()
    for val_y, val_x in product(vy, vx):
        job = next(iter(project.find_jobs(filter={keys[0]: val_y, keys[1]: val_x})))
        spectrum = job_energy_histogram(job)
        spectra.append(spectrum)


# Plan: 1. compute all the spectra via job_energy_histogram.
#       2. load them into XSpectra.
#       3. profit!


def main():
    """Main entry point."""
    proj = signac.get_project(search=False)

    a0 = job_util.get_key_values(proj, "a0")
    n_e = job_util.get_key_values(proj, "n_e")
    energy = np.linspace(1, 499, 499)

    charge = np.zeros((len(a0), len(n_e), energy.shape[0]))

    for i, j in np.ndindex(charge.shape[:-1]):
        match = proj.find_jobs(filter={"a0": a0[i], "n_e": n_e[j]})
        assert len(match) == 1, "More than 1 job found."
        job = next(iter(match))
        charge[i, j, :] = job_energy_histogram(job)

    spectra = xr.DataArray(
        charge,
        dims=("a0", "n_e", "E"),
        coords={"a0": a0, "n_e": n_e, "E": energy},
    )


if __name__ == "__main__":
    main()
