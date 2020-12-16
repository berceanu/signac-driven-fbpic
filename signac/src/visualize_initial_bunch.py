import pathlib
import pandas as pd
import numpy as np
import datashader as ds
from datashader.utils import export_image
from colorcet import fire
import unyt as u


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
    df["energy_MeV"] = (u.electron_mass * u.clight ** 2).to_value("MeV") * df.gamma

    df["percent_c"] = np.sqrt(1 - 1 / df.gamma ** 2) * 100.0

    return df


def shade_bunch(df, coord1, coord2, export_path=pathlib.Path.cwd()):
    cvs = ds.Canvas(
        plot_width=4200, plot_height=700, x_range=(-1800, 1800), y_range=(-300, 300)
    )
    agg = cvs.points(df, coord1, coord2)
    img = ds.tf.shade(agg, cmap=fire, how="linear")
    export_image(
        img, f"bunch_{coord1}_{coord2}", background="black", export_path=export_path
    )


def main():
    import random
    from openpmd_viewer.addons import LpaDiagnostics
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    ids = [job.id for job in proj]
    job = proj.open_job(id=random.choice(ids))
    print(f"job {job.id}")

    # plot via datashader
    df = read_bunch(job.fn("exp_4deg.txt"))

    print(df.describe())
    print(df[["x_mu", "y_mu", "z_mu"]].describe())

    shade_bunch(df, "z_mu", "x_mu")


if __name__ == "__main__":
    main()
