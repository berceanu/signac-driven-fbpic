"""
Computes the electron energy spectra for the whole parameter space and loads it
into the N-dimensional array class.
"""
import numpy as np
import signac
import xarray as xr

import job_util
from energy_histograms import job_energy_histogram
from util import first
from xarray_spectra import XFigure, XSpectra, generate_slices


def main():
    """Main entry point."""
    proj = signac.get_project(search=False)

    a0 = job_util.get_key_values(proj, "a0")
    n_e = job_util.get_key_values(proj, "n_e")
    energy = np.linspace(1, 499, 499)

    s1 = generate_slices("a0", np.array((2.4, 2.7, 3.0, 3.1)))
    s2 = generate_slices("n_e", np.array((7.4, 7.7, 7.9, 8.1)) * 1.0e24)
    s = s1 + s2
    assert(len(s) == 8), "choose precisely 8 values"

    charge = np.zeros((len(a0), len(n_e), energy.shape[0]))

    for i, j in np.ndindex(charge.shape[:-1]):
        match = proj.find_jobs(filter={"a0": a0[i], "n_e": n_e[j]})
        assert len(match) == 1, "More than 1 job found."
        job = first(match)
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

    xs = XSpectra(spectra, gaussian_std=10)
    #
    xf = XFigure(xs, s)
    xf.render()
    xf.savefig()


if __name__ == "__main__":
    main()
