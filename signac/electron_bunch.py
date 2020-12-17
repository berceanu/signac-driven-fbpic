import pathlib
import pandas as pd
import numpy as np
import datashader as ds
from datashader.utils import export_image
from colorcet import fire
import unyt as u
from openpmd_api import Series, Access, Dataset, Mesh_Record_Component, Unit_Dimension


SCALAR = Mesh_Record_Component.SCALAR


def bunch_to_openpmd(bunch_txt, outdir=pathlib.Path.cwd()):
    # read bunch data from txt file
    df = pd.read_csv(
        bunch_txt,
        delim_whitespace=True,
        names=["x_m", "y_m", "z_m", "ux", "uy", "uz"],
    )
    # open file for writing
    f = Series(str(outdir / "data_%05T.h5"), Access.create)

    # all required openPMD attributes will be set to reasonable default values
    # (all ones, all zeros, empty strings,...)
    # manually setting them enforces the openPMD standard
    f.set_meshes_path("fields")
    f.set_particles_path("particles")

    # new iteration
    cur_it = f.iterations[0]

    # particles
    electrons = cur_it.particles["bunch"]
    electrons.set_attribute("attribute", "value")  # FIXME

    electrons["charge"][SCALAR].make_constant(u.electron_charge.to_value("C"))
    electrons["charge"].unit_dimension = {
        Unit_Dimension.T: 1,
        Unit_Dimension.I: 1,
    }

    electrons["mass"][SCALAR].make_constant(u.electron_mass.to_value("kg"))
    electrons["mass"].unit_dimension = {
        Unit_Dimension.M: 1,
    }

    electrons["weighting"][SCALAR].make_constant(4993.207300706606)  # FIXME

    # position
    particlePos_x = df.x_m.to_numpy(dtype=np.float32)
    particlePos_y = df.y_m.to_numpy(dtype=np.float32)
    particlePos_z = df.z_m.to_numpy(dtype=np.float32)

    d = Dataset(particlePos_x.dtype, extent=particlePos_x.shape)
    electrons["position"]["x"].reset_dataset(d)
    electrons["position"]["y"].reset_dataset(d)
    electrons["position"]["z"].reset_dataset(d)

    # momentum
    mc = u.electron_mass.to_value("kg") * u.clight.to_value("m/s")
    particleMom_x = df.ux.to_numpy(dtype=np.float32) * mc
    particleMom_y = df.uy.to_numpy(dtype=np.float32) * mc
    particleMom_z = df.uz.to_numpy(dtype=np.float32) * mc

    d = Dataset(particleMom_x.dtype, extent=particleMom_x.shape)
    electrons["momentum"]["x"].reset_dataset(d)
    electrons["momentum"]["y"].reset_dataset(d)
    electrons["momentum"]["z"].reset_dataset(d)
    electrons["momentum"].unit_dimension = {
        Unit_Dimension.M: 1,
        Unit_Dimension.L: 1,
        Unit_Dimension.T: -1,
    }

    # positionOffset
    electrons["positionOffset"]["x"].make_constant(0.0)
    electrons["positionOffset"]["y"].make_constant(0.0)
    electrons["positionOffset"]["z"].make_constant(0.0)

    # store all chunks
    electrons["position"]["x"].store_chunk(particlePos_x)
    electrons["position"]["y"].store_chunk(particlePos_y)
    electrons["position"]["z"].store_chunk(particlePos_z)

    electrons["momentum"]["x"].store_chunk(particleMom_x)
    electrons["momentum"]["y"].store_chunk(particleMom_y)
    electrons["momentum"]["z"].store_chunk(particleMom_z)

    # at any point in time you may decide to dump already created output to
    # disk note that this will make some operations impossible (e.g. renaming
    # files)
    f.flush()

    # now the file is closed
    del f


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
    job = random.choice(list(proj))
    print(f"job {job.id}")

    # plot via datashader
    df = read_bunch(job.fn("exp_4deg.txt"))
    # print(df.describe())
    # print(df[["x_mu", "y_mu", "z_mu"]].describe())

    # shade_bunch(df, "z_mu", "x_mu")

    # bunch_to_openpmd(bunch_txt=job.fn("exp_4deg.txt"), outdir=pathlib.Path(job.ws))
    bunch_to_openpmd(bunch_txt=job.fn("exp_4deg.txt"))


if __name__ == "__main__":
    main()
