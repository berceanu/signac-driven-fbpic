import pandas as pd
import pathlib
from matplotlib import cm, colors, figure, rc_context, ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np
from fast_histogram import histogram1d
import math
import mpl_util
import util

def read_spectrum(path_to_csv):
    df = pd.read_csv(
        path_to_csv,
        comment="#",
        names=["E_MeV_float", "dN_over_dE"],
    )
    df.loc[:, "E_MeV"] = df.loc[:, "E_MeV_float"].astype(int)
    grouped = df.groupby(["E_MeV"])
    df_ = grouped[["dN_over_dE"]].agg(["mean"])
    df_.columns = df_.columns.get_level_values(0)
    df_["dN_over_dE"] = df_["dN_over_dE"].astype(np.float64)
    df_["scaled_dN_over_dE"] = util.normalize_to_interval(0,1,df_["dN_over_dE"])

    return df_



def plot_spectrum(df, ax):
    ax.step(
        df.index,
        df.scaled_dN_over_dE
    )
    ax.set_xlabel("E (MeV)")
    ax.set_ylabel("dN/dE (a.u.)")

def main():
    """Main entry point."""


    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    df = read_spectrum(csv_path)

    with rc_context():
        mpl_util.mpl_publication_style()

        fig = figure.Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        plot_spectrum(df, ax)
        fig.savefig("exp")


if __name__ == "__main__":
    main()
