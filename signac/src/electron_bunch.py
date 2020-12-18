import pathlib
import pandas as pd
import numpy as np
from matplotlib import pyplot
import datashader as ds
from datashader.utils import export_image
from colorcet import fire
import unyt as u
from openpmd_api import Series, Access, Dataset, Mesh_Record_Component, Unit_Dimension
from openpmd_viewer.addons import LpaDiagnostics
from simulation_diagnostics import particle_energy_histogram

SCALAR = Mesh_Record_Component.SCALAR
mc = u.electron_mass.to_value("kg") * u.clight.to_value("m/s")


def plot_bunch_energy_histogram(opmd_dir, export_dir):
    time_series = LpaDiagnostics(opmd_dir, check_all_files=True)

    hist, energy_bins, _ = particle_energy_histogram(
        tseries=time_series, iteration=0, species="bunch", cutoff=np.inf
    )
    fig, ax = pyplot.subplots()

    ax.set_xlabel("E (MeV)")
    ax.set_ylabel("dQ/dE (pC/MeV)")

    energy = energy_bins[1:]
    charge = hist

    mask = (energy > 50) & (energy < 150)  # MeV
    energy = energy[mask]
    charge = charge[mask]

    ax.step(
        energy,
        charge,
    )
    fig.savefig(pathlib.Path(export_dir) / "energy_histogram.png")
    pyplot.close(fig)


def read_bunch(txt_file):
    df = pd.read_csv(
        txt_file,
        delim_whitespace=True,
        names=["x_m", "y_m", "z_m", "ux", "uy", "uz"],
    )
    return df


def bunch_openpmd_to_dataframe(workdir=pathlib.Path.cwd()):
    f = Series(str(workdir / "bunch" / "data_%05T.h5"), Access.read_only)

    cur_it = f.iterations[0]
    electrons = cur_it.particles["bunch"]

    # charge = electrons["charge"][SCALAR]
    # unit_dim = electrons["charge"].unit_dimension
    # x_unit = electrons["charge"][SCALAR].unit_SI

    # for attribute in ("bunch_charge", "n_physical_particles", "n_macro_particles"):
    #     value = electrons.get_attribute(attribute)
    #     print(f"{attribute}: {value}")

    x_m = electrons["position"]["x"]
    y_m = electrons["position"]["y"]
    z_m = electrons["position"]["z"]

    ux = electrons["momentum"]["x"]
    uy = electrons["momentum"]["y"]
    uz = electrons["momentum"]["z"]

    w = electrons["weighting"][SCALAR]

    x_m_data = x_m.load_chunk()
    y_m_data = y_m.load_chunk()
    z_m_data = z_m.load_chunk()

    ux_data = ux.load_chunk()
    uy_data = uy.load_chunk()
    uz_data = uz.load_chunk()

    w_data = w.load_chunk()

    f.flush()

    d = {
        "x_m": x_m_data,
        "y_m": y_m_data,
        "z_m": z_m_data,
        "ux": ux_data / mc,
        "uy": uy_data / mc,
        "uz": uz_data / mc,
        "w": w_data,
    }
    df = pd.DataFrame(d)

    # multiply your read data with x_unit to covert to SI
    # unit_dim = electrons["position"].unit_dimension
    # x_unit = electrons["position"]["x"].unit_SI

    del f

    return df


def write_bunch_openpmd(bunch_txt, bunch_charge, outdir=pathlib.Path.cwd()):
    # read bunch data from txt file
    df = read_bunch(bunch_txt)

    # open file for writing
    f = Series(str(outdir / "bunch" / "data_%05T.h5"), Access.create)

    # all required openPMD attributes will be set to reasonable default values
    # (all ones, all zeros, empty strings,...)
    # manually setting them enforces the openPMD standard
    f.set_meshes_path("fields")
    f.set_particles_path("particles")

    # new iteration
    cur_it = f.iterations[0]

    # particles
    electrons = cur_it.particles["bunch"]

    n_physical_particles = bunch_charge / u.electron_charge.to_value("C")
    n_macro_particles = df.shape[0]
    electrons.set_attribute("bunch_charge", bunch_charge)
    electrons.set_attribute("n_physical_particles", n_physical_particles)
    electrons.set_attribute("n_macro_particles", n_macro_particles)

    electrons["charge"][SCALAR].make_constant(u.electron_charge.to_value("C"))
    electrons["charge"].unit_dimension = {
        Unit_Dimension.T: 1,
        Unit_Dimension.I: 1,
    }
    electrons["charge"][SCALAR].unit_SI = 1.0

    electrons["mass"][SCALAR].make_constant(u.electron_mass.to_value("kg"))
    electrons["mass"].unit_dimension = {
        Unit_Dimension.M: 1,
    }
    electrons["mass"][SCALAR].unit_SI = 1.0

    # position
    particlePos_x = df.x_m.to_numpy(dtype=np.float64)
    particlePos_y = df.y_m.to_numpy(dtype=np.float64)
    particlePos_z = df.z_m.to_numpy(dtype=np.float64)

    d = Dataset(particlePos_x.dtype, extent=particlePos_x.shape)
    electrons["position"]["x"].reset_dataset(d)
    electrons["position"]["y"].reset_dataset(d)
    electrons["position"]["z"].reset_dataset(d)
    electrons["position"].unit_dimension = {
        Unit_Dimension.L: 1,
    }
    for coord in "x", "y", "z":
        electrons["position"][coord].unit_SI = 1.0

    # weighting
    fill_value = n_physical_particles / n_macro_particles
    particle_weighting = np.full_like(particlePos_x, fill_value)

    d = Dataset(particle_weighting.dtype, extent=particle_weighting.shape)
    electrons["weighting"][SCALAR].reset_dataset(d)

    # momentum
    particleMom_x = df.ux.to_numpy(dtype=np.float64) * mc
    particleMom_y = df.uy.to_numpy(dtype=np.float64) * mc
    particleMom_z = df.uz.to_numpy(dtype=np.float64) * mc

    d = Dataset(particleMom_x.dtype, extent=particleMom_x.shape)
    electrons["momentum"]["x"].reset_dataset(d)
    electrons["momentum"]["y"].reset_dataset(d)
    electrons["momentum"]["z"].reset_dataset(d)
    electrons["momentum"].unit_dimension = {
        Unit_Dimension.M: 1,
        Unit_Dimension.L: 1,
        Unit_Dimension.T: -1,
    }
    for coord in "x", "y", "z":
        electrons["momentum"][coord].unit_SI = 1.0

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

    electrons["weighting"][SCALAR].store_chunk(particle_weighting)

    # at any point in time you may decide to dump already created output to
    # disk note that this will make some operations impossible (e.g. renaming
    # files)
    f.flush()

    # now the file is closed
    del f


def shade_bunch(df, coord1, coord2, export_path=pathlib.Path.cwd()):
    cvs = ds.Canvas(
        plot_width=4200, plot_height=700, x_range=(-1800, 1800), y_range=(-300, 300)
    )
    # TODO read from openPMD
    # TODO export images to job.ws / "bunch"
    # convert to microns
    df["x_mu"] = df.x_m * 1e6
    df["y_mu"] = df.y_m * 1e6
    df["z_mu"] = df.z_m * 1e6

    agg = cvs.points(df, coord1, coord2)
    img = ds.tf.shade(agg, cmap=fire, how="linear")
    export_image(
        img, f"bunch_{coord1}_{coord2}", background="black", export_path=export_path
    )


def main():
    import random
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    job = random.choice(list(proj))
    print(f"job {job.id}")

    # plot via datashader
    df = read_bunch(job.fn("exp_4deg.txt"))
    # print(df.describe())

    plot_bunch_energy_histogram(
        opmd_dir=pathlib.Path(job.ws) / "bunch",
        export_dir=pathlib.Path.cwd(),
    )
    # shade_bunch(df, "z_mu", "x_mu", export_path=pathlib.Path.cwd() / "bunch")

    bunch_openpmd_to_dataframe(workdir=pathlib.Path(job.ws))

    # write_bunch_openpmd(
    #     bunch_txt=job.fn("exp_4deg.txt"),
    #     outdir=pathlib.Path.cwd(),
    #     bunch_charge=-200.0e-12,
    # ) TODO uncomment


if __name__ == "__main__":
    main()
