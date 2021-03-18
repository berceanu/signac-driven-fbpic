"""
Computes the electron energy spectra for the whole parameter space and loads it
into the N-dimensional array class.
"""
import random

import numpy as np
import xarray as xr

import job_util
from energy_histograms import job_energy_histogram
from util import first
from xarray_spectra import XFigure, XSpectra, generate_slices

import signac

DIM_MAPPING = {"y": "power", "x": "n_e"}


def main():
    """Main entry point."""
    proj = signac.get_project(search=False)

    y_values = job_util.get_key_values(proj, DIM_MAPPING["y"])
    x_values = job_util.get_key_values(proj, DIM_MAPPING["x"])
    energy = np.linspace(1, 499, 499)

    random.seed(12)

    chosen_y_vals = generate_slices(
        DIM_MAPPING["y"], np.array(random.sample(y_values, 4)), DIM_MAPPING["x"]
    )
    chosen_x_vals = generate_slices(
        DIM_MAPPING["x"], np.array(random.sample(x_values, 4)), DIM_MAPPING["y"]
    )
    assert len(chosen_y_vals + chosen_x_vals) == 8, "choose precisely 8 values"

    charge = np.zeros((len(y_values), len(x_values), energy.shape[0]))

    for i, j in np.ndindex(charge.shape[:-1]):
        match = proj.find_jobs(
            filter={DIM_MAPPING["y"]: y_values[i], DIM_MAPPING["x"]: x_values[j]}
        )
        assert len(match) == 1, "More than 1 job found."
        job = first(match)
        charge[i, j, :] = job_energy_histogram(job)

    spectra = xr.DataArray(
        charge,
        dims=("power", "n_e", "E"),
        coords={"power": y_values, "n_e": x_values, "E": energy},
    )
    #
    spectra.coords[DIM_MAPPING["y"]].attrs["plot_label"] = r"$\alpha$"
    #
    spectra.coords[DIM_MAPPING["x"]].attrs[
        "plot_label"
    ] = r"$n_e$ ($10^{18}\,\mathrm{cm^{-3}}$)"
    spectra.coords[DIM_MAPPING["x"]].attrs["units"] = "1 / meter ** 3"
    spectra.coords[DIM_MAPPING["x"]].attrs["to_units"] = "1 / centimeter ** 3"
    spectra.coords[DIM_MAPPING["x"]].attrs["scaling_factor"] = 1.0e-18

    xs = XSpectra(spectra, gaussian_std=10, dim_mapping=DIM_MAPPING)
    #
    xf = XFigure(xs, chosen_y_vals + chosen_x_vals)
    xf.render()
    xf.savefig()


if __name__ == "__main__":
    main()
