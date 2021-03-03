"""
Computes the electron energy spectra for the whole parameter space and loads it
into the N-dimensional array class.
"""
import job_util
from itertools import product
from energy_histograms import job_energy_histogram
from xarray_spectra import XSpectra
import signac


def two_parameters_study(project, keys=("a0", "n_e")):
    spectra = list()

    vy = job_util.get_key_values(project, keys[0])
    vx = job_util.get_key_values(project, keys[1])

    for val_y, val_x in product(vy, vx):
        job = next(iter(project.find_jobs(filter={keys[0]: val_y, keys[1]: val_x})))
        spectrum = job_energy_histogram(job)
        spectra.append(spectrum)


def main():
    """Main entry point."""
    proj = signac.get_project(search=False)

    for job in proj:  # ~14s
        hist = job_energy_histogram(job)


if __name__ == "__main__":
    main()
