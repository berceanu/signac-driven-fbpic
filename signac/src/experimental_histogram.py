"""
Postprocessing and visualizing experimental data, such as the electron spectrum/histogram.
"""
import pathlib

import numpy as np
import pandas as pd
from matplotlib import figure, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg

import mpl_util
import util


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

    return df_


def plot_spectrum(df, ax):
    ax.step(df.index, df.dN_over_dE_normalized)
    ax.set_xlabel("E (MeV)")
    ax.set_ylabel("dN/dE (a.u.)")


def main():
    """Main entry point."""

    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    df = read_spectrum(csv_path)

    with rc_context():
        mpl_util.mpl_publication_style()

        fig = figure.Figure()
        _ = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        plot_spectrum(df, ax)
        fig.savefig("spectrum_exp")


if __name__ == "__main__":
    main()
