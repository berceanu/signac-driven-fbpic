import math
from prepic import Plasma, lwfa
import unyt as u
import pandas as pd
import pathlib
from matplotlib import pyplot
from matplotlib import cm, colors, figure, rc_context, ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg
import numpy as np

SQRT_FACTOR = math.sqrt(2 * math.log(2))

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
    csv_path = pathlib.Path.cwd() / "experimental_spectrum.csv"
    df = read_spectrum(csv_path)
    # print(df)
    print(np.diff(df.E_MeV))

    fig = figure.Figure()
    _ = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    plot_spectrum(df, ax)

    fig.savefig("exp.png")




    laser = lwfa.Laser.from_a0(
        a0=2.4 * u.dimensionless,
        τL=27.8 / SQRT_FACTOR * u.femtosecond,
        beam=lwfa.GaussianBeam(
            w0=22.0 / SQRT_FACTOR * u.micrometer, λL=815 * u.nanometer
        ),
    )
    plasma = Plasma(n_pe=8.0e18 * u.cm ** (-3))

    print(laser)
    print(plasma)


if __name__ == "__main__":
    main()
