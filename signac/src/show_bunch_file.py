import pandas as pd
import numpy as np
from scipy.constants import physical_constants
import datashader as ds
from datashader.utils import export_image
from colorcet import fire

c_light = physical_constants["speed of light in vacuum"][0]
m_e = physical_constants["electron mass"][0]
q_e = physical_constants["elementary charge"][0]
mc2 = m_e * c_light ** 2 / (q_e * 1e6)  # 0.511 MeV


def read_bunch(txt_file):
    df = pd.read_csv(
        txt_file,
        delim_whitespace=True,
        names=["x_m", "y_m", "z_m", "ux", "uy", "uz"],
    )
    # convert to microns
    df["x_mu"] = df.x_m * 1e6
    df["y_mu"] = df.y_m * 1e6
    df["z_mu"] = df.z_m * 1e6

    # remove first 3 columns
    # df = df.drop(["x_m", "y_m", "z_m"], axis=1)

    # compute gamma factor
    df["gamma"] = np.sqrt(1 + df.ux ** 2 + df.uy ** 2 + df.uz ** 2)

    # compute energy
    df["energy_MeV"] = mc2 * df.gamma

    df["percent_c"] = np.sqrt(1 - 1 / df.gamma ** 2) * 100.0

    return df


def shade_bunch(coord1, coord2):
    cvs = ds.Canvas(plot_width=700, plot_height=700)
    agg = cvs.points(df, coord1, coord2)
    img = ds.tf.shade(agg, cmap=fire, how="linear")
    export_image(img, f"bunch_{coord1}_{coord2}", background="black", export_path=".")


if __name__ == "__main__":
    # plot via datashader
    df = read_bunch("../exp_4deg.txt")
    print(df[["x_mu","y_mu","z_mu"]].describe())

    shade_bunch("x_mu", "y_mu")
    shade_bunch("y_mu", "z_mu")
    shade_bunch("x_mu", "z_mu")
