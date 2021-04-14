"""
Automatic search for best fit amongst simulated spectra.
"""
import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic
from matplotlib import figure
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg

import util
import job_util
import mpl_util
import energy_histograms
import signac

ic.configureOutput(includeContext=True)


def compute_simulated_spectra(project, *, from_energy, to_energy):
    """
    Loop though the project's statepoints and compute the energy spectrum for each.
    Return all spectra inside an XArray.
    """
    powers = job_util.get_key_values(project, "power")
    densities = job_util.get_key_values(project, "n_e")

    energy = np.linspace(
        from_energy, to_energy, to_energy - from_energy, endpoint=False
    )
    charge = np.zeros((len(powers), len(densities), energy.shape[0]))

    for i, j in np.ndindex(charge.shape[:-1]):
        match = project.find_jobs(filter={"power": powers[i], "n_e": densities[j]})
        assert len(match) == 1, "Only one job needs to match."
        job = util.first(match)

        hist = energy_histograms.job_energy_histogram(
            job,
            bins=to_energy - from_energy,
            erange=(from_energy, to_energy),
            normalized=True,
        )
        charge[i, j, :] = hist

    spectra = xr.DataArray(
        charge,
        dims=("power", "n_e", "E"),
        coords={"power": powers, "n_e": densities, "E": energy},
    )
    #
    spectra.coords["power"].attrs["plot_label"] = r"$\alpha$"
    #
    spectra.coords["n_e"].attrs["plot_label"] = r"$n_e$ ($10^{18}\,\mathrm{cm^{-3}}$)"

    return spectra


def max_cross_correlation(
    simulated_spectra, experimental_spectrum, cutoff_from_maximum=0.8
):
    weights = np.tanh((experimental_spectrum / cutoff_from_maximum) ** 2)

    se = (simulated_spectra * experimental_spectrum).sum(dim="E")
    s = simulated_spectra.sum(dim="E")
    e = experimental_spectrum.sum(dim="E")

    rd = (se / (s * e)) ** 2
    cross_corr = (rd * weights).sum(dim="E")

    ind = cross_corr.where(cross_corr == cross_corr.max(), drop=True).squeeze()
    return ind.power.values.item(), ind.n_e.values.item()


def plot_experimental_spectrum(axes, spectrum):
    axes.step(
        spectrum.E.values,
        spectrum.values,
        "black",
        label="experiment",
    )
    axes.set_xlabel(r"$E$ ($\mathrm{MeV}$)")
    axes.set_ylabel(r"$\frac{\mathrm{d} N}{\mathrm{d} E}$ ($\mathrm{a.u.}$)")
    return axes


def plot_simulated_spectrum(axes, spectrum):
    label = (
        f"{spectrum.n_e.attrs['plot_label']} = {spectrum.n_e.values / 1.0e+24}, "
        f"{spectrum.power.attrs['plot_label']} = {spectrum.power.values:.3f}"
    )
    axes.step(spectrum.E.values, spectrum.values, label=label)
    axes.legend(fontsize=4)
    return axes, label


def read_experimental_spectrum(path_to_csv, *, from_energy, to_energy):
    csv_df = pd.read_csv(
        path_to_csv,
        comment="#",
        names=["E", "dN_over_dE"],
    )
    csv_df.set_index("E", inplace=True)

    new_index = np.linspace(
        from_energy,
        to_energy,
        to_energy - from_energy,
        endpoint=False,
        dtype=int,
    )
    df_ = csv_df.reindex(new_index, method="nearest")

    df_["dN_over_dE_normalized"] = util.normalize_to_interval(0, 1, df_["dN_over_dE"])
    return df_["dN_over_dE_normalized"].to_xarray()


def main():
    """Main entry point."""
    FROM_ENERGY = 71  # MeV
    TO_ENERGY = 500  # MeV

    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    experimental_spectrum = read_experimental_spectrum(
        csv_path, from_energy=FROM_ENERGY, to_energy=TO_ENERGY
    )

    proj = signac.get_project(search=False)
    simulated_spectra = compute_simulated_spectra(
        proj, from_energy=FROM_ENERGY, to_energy=TO_ENERGY
    )

    # manual selection
    # optim_power, optim_density = 3.0, 8.1e24

    # automatic selection
    optim_power, optim_density = max_cross_correlation(
        simulated_spectra,
        experimental_spectrum,
    )

    selected_simulated_spectrum = simulated_spectra.sel(
        power=optim_power, n_e=optim_density, method="nearest"
    )

    with rc_context():
        mpl_util.mpl_publication_style()

        fig = figure.Figure()
        _ = FigureCanvasAgg(fig)
        axs = fig.add_subplot(111)

        axs = plot_experimental_spectrum(axs, experimental_spectrum)
        axs, label = plot_simulated_spectrum(axs, selected_simulated_spectrum)

        axs.figure.savefig(util.slugify(label) + ".png")


if __name__ == "__main__":
    main()
