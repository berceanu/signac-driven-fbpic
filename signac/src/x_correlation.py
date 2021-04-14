"""
Automatic search for best fit amongst simulated spectra.
"""
import pathlib

import numpy as np
import pandas as pd
import xarray as xr
from icecream import ic
from matplotlib import rc_context
from matplotlib import pyplot

import util
import job_util
import mpl_util
import energy_histograms
import signac

ic.configureOutput(includeContext=True)

FROM_ENERGY = 71  # MeV
TO_ENERGY = 500  # MeV


def compute_simulated_spectra(
    project,
    *,
    from_energy,
    to_energy,
    cone_aperture=None,
):
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

        charge[i, j, :] = energy_histograms.job_energy_histogram(
            job,
            bins=to_energy - from_energy,
            erange=(from_energy, to_energy),
            normalized=True,
            cone_aperture=cone_aperture,
        )

    spectra = xr.DataArray(
        charge,
        dims=("power", "n_e", "E"),
        coords={"power": powers, "n_e": densities, "E": energy},
    )
    spectra.coords["power"].attrs["plot_label"] = r"$\alpha$"
    spectra.coords["n_e"].attrs["plot_label"] = r"$n_e$ ($10^{18}\,\mathrm{cm^{-3}}$)"
    return spectra


def pearson_correlation(simulated_spectra, experimental_spectrum):
    return xr.corr(simulated_spectra, experimental_spectrum, dim="E")


def weighted_correlation(
    simulated_spectra, experimental_spectrum, *, cutoff_from_maximum=0.8
):
    weights = np.tanh((experimental_spectrum / cutoff_from_maximum) ** 2)

    se_sum = (simulated_spectra * experimental_spectrum).sum(dim="E")
    s_sum = simulated_spectra.sum(dim="E")
    e_sum = experimental_spectrum.sum(dim="E")

    relative_deviation = (se_sum / (s_sum * e_sum)) ** 2
    cross_corr = (relative_deviation * weights).sum(dim="E")
    return cross_corr


def best_match(cross_correlation_function, simulated_spectra, experimental_spectrum):
    x_corr = cross_correlation_function(simulated_spectra, experimental_spectrum)

    ind = x_corr.where(x_corr == x_corr.max(), drop=True).squeeze()
    power, n_e = ind.power.values.item(), ind.n_e.values.item()

    return simulated_spectra.sel(power=power, n_e=n_e, method="nearest")


def compute_spectra_and_match(
    experimental_spectrum, *, cone_aperture=0.01, x_corr_func=weighted_correlation
):
    simulated_spectra = compute_simulated_spectra(
        signac.get_project(search=False),
        from_energy=FROM_ENERGY,
        to_energy=TO_ENERGY,
        cone_aperture=cone_aperture,
    )
    return best_match(x_corr_func, simulated_spectra, experimental_spectrum)


def plot_experimental_spectrum(axes, spectrum):
    axes.step(
        spectrum.E.values,
        spectrum.values,
        "black",
        label="experiment",
        linewidth=0.8,
    )
    axes.set_xlabel(r"$E$ ($\mathrm{MeV}$)")
    axes.set_ylabel(r"$\frac{\mathrm{d} N}{\mathrm{d} E}$ ($\mathrm{a.u.}$)")
    return axes


def plot_simulated_spectrum(axes, spectrum, *, cone_aperture):
    label = (
        f"{spectrum.n_e.attrs['plot_label']} = {spectrum.n_e.values.item() / 1e24}, "
        f"{spectrum.power.attrs['plot_label']} = {spectrum.power.values.item():.3f}"
    )
    if cone_aperture is not None:
        label += f", $2\\theta = {float(cone_aperture) * 1e3:.1f}$ mrad"

    axes.step(
        spectrum.E.values,
        spectrum.values,
        linewidth=0.8,
        label=label,
    )
    axes.legend(fontsize=3)
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
    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    experimental_spectrum = read_experimental_spectrum(
        csv_path, from_energy=FROM_ENERGY, to_energy=TO_ENERGY
    )

    matched_spectra = {}
    for x_corr_foo in (weighted_correlation, pearson_correlation):
        matched_spectra[x_corr_foo.__name__] = {}
        for aperture in (None, 0.01):
            selected_simulated_spectrum = compute_spectra_and_match(
                experimental_spectrum,
                cone_aperture=aperture,
                x_corr_func=x_corr_foo,
            )
            matched_spectra[x_corr_foo.__name__][aperture] = selected_simulated_spectrum

    for x_corr_foo in matched_spectra:
        with rc_context():
            mpl_util.mpl_publication_style()

            fig, axs = pyplot.subplots()

            axs = plot_experimental_spectrum(axs, experimental_spectrum)
            for aperture in matched_spectra[x_corr_foo]:
                axs, _ = plot_simulated_spectrum(
                    axs,
                    matched_spectra[x_corr_foo][aperture],
                    cone_aperture=aperture,
                )
            axs.set_title(f"$\\text{{{x_corr_foo}}}$")
            fig.savefig(f"{x_corr_foo}.png")


if __name__ == "__main__":
    main()
