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


# load xarray dataset from file
ds_spectra = xr.open_dataset("spectra.nc")
# extract the (only) array from the dataset
spectra = ds_spectra["spectra"]


# Read the experimental spectrum
Energy_min = 70  # minimum enregy, MeV
Energy_max = 450  # maximum energy, MeV

Exp_Spectrum = np.loadtxt("Experimental.txt", delimiter=",")
Exp_Energy_full = Exp_Spectrum[:, 0]  # Exp. energy scan
Exp_Counts_full = Exp_Spectrum[:, 1]  # Exp. spectrum

# Selected energy range

Exp_Energy = Exp_Energy_full[
    (Exp_Energy_full >= Energy_min) * (Exp_Energy_full < Energy_max)
]
Exp_Counts = Exp_Counts_full[
    (Exp_Energy_full >= Energy_min) * (Exp_Energy_full < Energy_max)
]


# Normalyze Counts:
Exp_Counts = Exp_Counts / np.max(Exp_Counts)


# SIMULATIONS
Npower = 8
Nn = 8

Power = np.linspace(1.5, 3, Npower)  # power set
N_e = np.linspace(7.4, 8.1, Nn) * 1e24  # electron density set, 1/m^3

Cutoff_from_maximum = 0.8  # Data weights are increasingly reduced below that threshold


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
    ic(ind.power, ind.n_e)

    return ind.power, ind.n_e


def my_cross_corellation(
    simulated_spectra, experimental_spectrum, cutoff_from_maximum=0.8
):
    simulated_count = simulated_spectra.values
    experimental_count = experimental_spectrum.values

    weight = np.tanh((experimental_count / cutoff_from_maximum) ** 2)

    numerator = np.sum(
        simulated_count * experimental_count[np.newaxis, np.newaxis, :],
        axis=-1,
    )

    relative_deviation = numerator ** 2 / (
        np.sum(simulated_count, axis=-1) ** 2 * np.sum(experimental_count) ** 2
    )

    x_corr = np.sum(
        relative_deviation[:, :, np.newaxis] * weight[np.newaxis, np.newaxis, :],
        axis=-1,
    )

    ind = np.unravel_index(np.argmax(x_corr), x_corr.shape)
    ic(ind)

    return x_corr, ind


XCorr = np.zeros((Npower, Nn))  # The output weighted chi^2  matrix


for i in range(0, Nn):
    power = Power[i]

    for j in range(0, Nn):
        n_e = N_e[j]

        # Read data
        Num_Spectrum = spectra.sel(power=power, n_e=n_e, method="nearest")

        # Numerical spectrum
        Num_Energy_full = Num_Spectrum.E.values  # Exp. energy scan
        Num_Counts_full = Num_Spectrum.values  # Exp. spectrum
        Num_Energy = Num_Energy_full[
            (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
        ]
        Num_Counts = Num_Counts_full[
            (Num_Energy_full >= Energy_min) * (Num_Energy_full < Energy_max)
        ]

        # Find the (weighted) Exp/Num matching
        Num_Counts_interp = np.interp(Exp_Energy, Num_Energy, Num_Counts)
        Num_Counts_interp = Num_Counts_interp / np.max(Num_Counts_interp)

        Weight = np.tanh((Exp_Counts / Cutoff_from_maximum) ** 2)

        Rel_deviation = np.sum(Num_Counts_interp * Exp_Counts) ** 2 / (
            np.sum(Num_Counts_interp) ** 2 * np.sum(Exp_Counts) ** 2
        )
        xcorr = np.sum(Weight * Rel_deviation)
        XCorr[i, j] = xcorr

# ic(XCorr.shape, XCorr)

# Best case (maximum xcorr)
result = np.where(XCorr == XCorr.max())

power = Power[int(result[0])]
n_e = N_e[int(result[1])]

print("Optimised density  \t\t= ", n_e, "$1/m^3$")
print("Optimised power  \t\t= ", power)

# TODO remove ".png" from everywhere
# TODO remove ic statements


def plot_experimental_spectrum(axes, spectrum):
    """Create 1D spectrum figure."""
    axes.step(
        spectrum.E.values,
        spectrum.values,
        "black",
        label="experiment",
    )
    axes.set_xlabel(r"$E$ ($\mathrm{MeV}$)")
    axes.set_ylabel(r"$\frac{\mathrm{d} N}{\mathrm{d} E}$ ($\mathrm{a.u.}$)")

    return axes


def plot_on_top(axes, simulated_spectrum):
    """Plot simulated simulated_spectrum on top of the experimental one."""
    label = f"{simulated_spectrum.n_e.attrs['plot_label']} = {simulated_spectrum.n_e.values / 1.0e+24}, {simulated_spectrum.power.attrs['plot_label']} = {simulated_spectrum.power.values:.3f}"

    axes.step(simulated_spectrum.E.values, simulated_spectrum.values, label=label)
    axes.legend(fontsize=4)

    return axes, label


def read_spectrum(path_to_csv, *, from_energy, to_energy):
    """Read spectrum data from CSV file, return dataframe."""
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
    experimental_spectrum = read_spectrum(
        csv_path, from_energy=FROM_ENERGY, to_energy=TO_ENERGY
    )

    proj = signac.get_project(search=False)
    spectra = compute_simulated_spectra(
        proj, from_energy=FROM_ENERGY, to_energy=TO_ENERGY
    )

    max_cross_correlation(spectra, experimental_spectrum)

    with rc_context():
        mpl_util.mpl_publication_style()

        fig = figure.Figure()
        _ = FigureCanvasAgg(fig)
        axs = fig.add_subplot(111)

        axs = plot_experimental_spectrum(axs, experimental_spectrum)

        selected_spectrum = spectra.sel(power=2.57, n_e=7.8e24, method="nearest")
        # selected_spectrum = spectra.sel(power=3.0, n_e=8.1e24, method="nearest")
        axs, label = plot_on_top(
            axs,
            simulated_spectrum=selected_spectrum,
        )
        axs.figure.savefig(util.slugify(label) + ".png")


if __name__ == "__main__":
    main()
