"""
Automatic search for best fit amongst simulated spectra.
"""
import pathlib

import numpy as np
from numpy.core.numeric import cross
import pandas as pd
import xarray as xr
from icecream import ic
from matplotlib import cm, colors, figure
from matplotlib import pyplot as plt
from matplotlib import rc_context, ticker
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

ic(Exp_Energy)
ic(Exp_Counts)

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


def cross_corellation(
    simulated_spectra, experimental_spectrum, cutoff_from_maximum=0.8
):
    exp_counts = experimental_spectrum["dN_over_dE_normalized"]

    # Num_Spectrum = spectra.sel(power=power, n_e=n_e, method="nearest")

    # weight = np.tanh((exp_counts / cutoff_from_maximum) ** 2)

    # relative_deviation = np.sum(Num_Counts_interp * exp_counts) ** 2 / (
    #     np.sum(Num_Counts_interp) ** 2 * np.sum(exp_counts) ** 2
    # )
    return exp_counts


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


def plot_spectrum(spectrum, axes):
    """Visualize spectrum as a line."""
    axes.step(
        spectrum.index.values,
        spectrum.dN_over_dE_normalized.values,
        "black",
        label="experiment",
    )
    axes.set_xlabel(r"$E$ ($\mathrm{MeV}$)")
    axes.set_ylabel(r"$\frac{\mathrm{d} N}{\mathrm{d} E}$ ($\mathrm{a.u.}$)")
    return axes


def spectrum_figure(spectrum, plotter, savefig=True):
    """Create spectrum figure, either 1D or colormap."""
    with rc_context():
        mpl_util.mpl_publication_style()

        sfig = figure.Figure()
        _ = FigureCanvasAgg(sfig)
        axes = sfig.add_subplot(111)

        plotter(spectrum, axes)

        if savefig:
            sfig.savefig(f"spectrum_exp_{plotter.__name__}" + ".png")

    return sfig


# TODO remove ".png" from everywhere
# TODO remove ic statements


def plot_on_top(sfig, spectrum):
    """Plot simulated spectrum on top of the experimental one."""

    label = f"{spectrum.n_e.attrs['plot_label']} = {spectrum.n_e.values / 1.0e+24}, {spectrum.power.attrs['plot_label']} = {spectrum.power.values:.3f}"

    with rc_context():
        mpl_util.mpl_publication_style()

        axes = sfig.axes
        assert len(axes) == 1, "Figure contains multiple Axes"
        axes = util.first(axes)

        axes.step(spectrum.E.values, spectrum.values, label=label)
        axes.legend(fontsize=4)

        sfig.savefig(util.slugify(label) + ".png")

    return axes


def read_spectrum(path_to_csv, *, from_energy, to_energy):
    """Read spectrum data from CSV file, return dataframe."""
    csv_df = pd.read_csv(
        path_to_csv,
        comment="#",
        names=["E_MeV_float", "dN_over_dE"],
    )

    csv_df.set_index("E_MeV_float", inplace=True)
    new_index = np.linspace(
        from_energy, to_energy, to_energy - from_energy, endpoint=False
    )
    df_reindexed = csv_df.reindex(new_index, method="nearest")

    ic(df_reindexed)

    # csv_df.loc[:, "E_MeV"] = new_index
    csv_df.loc[:, "E_MeV"] = csv_df.loc[:, "E_MeV_float"].astype(int)

    grouped = csv_df.groupby(["E_MeV"])
    df_ = grouped[["dN_over_dE"]].agg(["mean"])

    df_.columns = df_.columns.get_level_values(0)
    df_["dN_over_dE"] = df_["dN_over_dE"].astype(np.float64)
    df_["dN_over_dE_normalized"] = util.normalize_to_interval(0, 1, df_["dN_over_dE"])

    return df_.query(f"E_MeV >= {from_energy} & E_MeV < {to_energy}")


def main():
    """Main entry point."""
    FROM_ENERGY = 71  # MeV
    TO_ENERGY = 500  # MeV

    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    spectrum = read_spectrum(csv_path, from_energy=FROM_ENERGY, to_energy=TO_ENERGY)

    ic(spectrum.index.values[0], spectrum.index.values[-1], spectrum.index.values.shape)

    proj = signac.get_project(search=False)

    spectra = compute_simulated_spectra(
        proj, from_energy=FROM_ENERGY, to_energy=TO_ENERGY
    )
    selected_spectrum = spectra.sel(power=2.57, n_e=7.8e24, method="nearest")

    ic(
        selected_spectrum.E.values[0],
        selected_spectrum.E.values[-1],
        selected_spectrum.E.values.shape,
    )

    my_fig = spectrum_figure(spectrum, plot_spectrum, savefig=False)
    # plot_on_top(
    #     my_fig, proj, job_filter={"power.$near": [3.0, 0.01], "n_e.$near": 8.1e24}
    # )
    plot_on_top(
        my_fig,
        selected_spectrum,
    )

    cross_corellation(spectra, spectrum)


if __name__ == "__main__":
    main()
