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
    #
    spectra.a0.attrs["plot_label"] = r"$a_0$"
    #
    spectra.n_e.attrs["plot_label"] = r"$n_e$ ($10^{18}\,\mathrm{cm^{-3}}$)"
    spectra.n_e.attrs["units"] = "1 / meter ** 3"
    spectra.n_e.attrs["to_units"] = "1 / centimeter ** 3"
    spectra.n_e.attrs["scaling_factor"] = 1.0e-18
    ##
    # ds = xr.Dataset({"spectra": spectra})
    # ds.to_zarr("spectra.zarr")

    xs = XSpectra(spectra, dim_mapping={"y": "a0", "x": "n_e"})
    # xs.sample({"n_e": 7.9e24}, "a0", vmax=40.0, left_xlim=50.0)
    xs.sample({"a0": 3.1}, "n_e", vmax=40.0, left_xlim=50.0)


if __name__ == "__main__":
    main()
