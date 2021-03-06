"""
Postprocessing and visualizing experimental data, such as the electron spectrum/histogram.
"""
import pathlib

import numpy as np
import pandas as pd
from matplotlib import cm, colors, figure, rc_context, ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg

import energy_histograms
import mpl_util
import util
import signac


def read_spectrum(path_to_csv):
    """Read spectrum data from CSV file, return dataframe."""
    csv_df = pd.read_csv(
        path_to_csv,
        comment="#",
        names=["E_MeV_float", "dN_over_dE"],
    )
    csv_df.loc[:, "E_MeV"] = csv_df.loc[:, "E_MeV_float"].astype(int)
    grouped = csv_df.groupby(["E_MeV"])
    df_ = grouped[["dN_over_dE"]].agg(["mean"])
    df_.columns = df_.columns.get_level_values(0)
    df_["dN_over_dE"] = df_["dN_over_dE"].astype(np.float64)
    df_["dN_over_dE_normalized"] = util.normalize_to_interval(0, 1, df_["dN_over_dE"])

    print(df_.index)
    return df_


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


def pcolor_spectrum(spectrum, axes):
    """Visualize spectrum as color band."""
    img = axes.pcolorfast(
        spectrum.index.values,
        [0.0, 1.0],
        spectrum.dN_over_dE_normalized.values[np.newaxis, :-1],
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        cmap=cm.get_cmap("turbo"),
        rasterized=True,
    )
    axes.yaxis.set(
        minor_locator=ticker.NullLocator(),
        minor_formatter=ticker.NullFormatter(),
        major_locator=ticker.NullLocator(),
        major_formatter=ticker.NullFormatter(),
    )
    for pos in "right", "left", "top", "bottom":
        axes.spines[pos].set_visible(False)
    #
    axes.set_xlabel(r"$E$ ($\mathrm{MeV}$)")
    axes.set_xlim(71, 499)
    cbar = mpl_util.add_colorbar(axes, img)
    cbar.ax.set_title(r"$\frac{\mathrm{d} N}{\mathrm{d} E}$ ($\mathrm{a.u.}$)")
    return img, cbar


def spectrum_figure(spectrum, plotter):
    """Create spectrum figure, either 1D or colormap."""
    with rc_context():
        mpl_util.mpl_publication_style()

        fig = figure.Figure()
        _ = FigureCanvasAgg(fig)
        axs = fig.add_subplot(111)

        plotter(spectrum, axs)
        fig.savefig(f"spectrum_exp_{plotter.__name__}")

    return fig


def plot_on_top(fig, project, job_filter):
    """Plot simulated spectrum on top of the experimental one."""

    def filter_to_label():
        s = ""
        for key, value in job_filter.items():
            if "power" in key:
                my_key = r"$\alpha$"
                my_value = value
            elif "n_e" in key:
                my_key = r"$n_e$"
                my_value = f"${util.latex_float(value / 1.0e+6)}$"
            else:
                my_key = key
                my_value = value
            s += f"{my_key} = {my_value}, "
        return s[:-2]

    match = project.find_jobs(filter=job_filter)
    assert len(match) == 1, "More than 1 job found."
    job = util.first(match)

    energy = np.linspace(1, 499, 499)
    hist = energy_histograms.job_energy_histogram(job)

    mask = energy >= 71
    energy = energy[mask]
    hist = hist[mask]
    charge = util.normalize_to_interval(0, 1, hist)

    with rc_context():
        mpl_util.mpl_publication_style()

        axs = fig.axes
        assert len(axs) == 1, "Figure contains multiple Axes"
        axs = util.first(axs)

        lbl = filter_to_label()
        axs.step(energy, charge, label=lbl)
        axs.set_xlim(71, 499)
        axs.legend(fontsize=5)

        fig.savefig(util.slugify(lbl))

    return axs


def main():
    """Main entry point."""

    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    spectrum = read_spectrum(csv_path)

    spectrum_figure(spectrum, pcolor_spectrum)

    proj = signac.get_project(
        root="/scratch/berceanu/runs/signac-driven-fbpic/", search=False
    )

    for p, n in zip((2.125, 2.2, 2.125, 2.15), (7.8e24, 7.775e24, 7.65e24, 7.65e24)):
        fig = spectrum_figure(spectrum, plot_spectrum)
        plot_on_top(fig, proj, job_filter={"power.$near": p, "n_e.$near": n})


if __name__ == "__main__":
    main()
