import pandas as pd
import pathlib
from matplotlib import cm, colors, figure, rc_context, ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

def read_spectrum(path_to_csv):
    df = pd.read_csv(
    path_to_csv,
    comment='#',
    names=["E_MeV", "dN_over_dE"],
    )
    return df

def plot_spectrum(df, ax):
    ax.plot("E_MeV", "dN_over_dE", data=df)
    

def main():
    """Main entry point."""

    fig = figure.Figure(figsize=(20, 8), dpi=192)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    df = read_spectrum(csv_path)

    plot_spectrum(df, ax)

    fig.savefig("exp.png")

if __name__ == '__main__':
    main()
