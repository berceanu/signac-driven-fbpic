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
from util import w_std, w_ave
from simulation_diagnostics import particle_energy_histogram, centroid_plot
from dataclasses import dataclass
from unyt.array import unyt_quantity

SCALAR = Mesh_Record_Component.SCALAR
mc = u.electron_mass.to_value("kg") * u.clight.to_value("m/s")


@dataclass
class Sphere:
    x: unyt_quantity
    y: unyt_quantity
    z: unyt_quantity
    r: unyt_quantity

    def volume(self) -> unyt_quantity:
        return 4 / 3 * np.pi * self.r ** 3


def bunch_centroid_plot(workdir=pathlib.Path.cwd() / "bunch"):
    time_series = LpaDiagnostics(workdir, check_all_files=True)
    centroid_plot(iteration=0, tseries=time_series, save_path=workdir)


def bunch_density(df):
    weights = df.w.to_numpy(dtype=np.float64) * u.dimensionless

    pos_x = df.x_um.to_numpy(dtype=np.float64) * u.micrometer
    pos_y = df.y_um.to_numpy(dtype=np.float64) * u.micrometer
    pos_z = df.z_um.to_numpy(dtype=np.float64) * u.micrometer

    sigmas = [w_std(pos, weights) for pos in (pos_x, pos_y, pos_z)]
    radius = min(sigmas)  # sphere radius
    sphere = Sphere(
        w_ave(pos_x, weights), w_ave(pos_y, weights), w_ave(pos_z, weights), radius
    )

    sph_x = np.full_like(pos_x, sphere.x)
    sph_y = np.full_like(pos_y, sphere.y)
    sph_z = np.full_like(pos_z, sphere.z)
    sph_r = np.full_like(pos_x, sphere.r)

    inside_sphere = (pos_x - sph_x) ** 2 + (pos_y - sph_y) ** 2 + (
        pos_z - sph_z
    ) ** 2 < sph_r ** 2

    electron_count = np.count_nonzero(inside_sphere) * np.mean(weights)
    density = electron_count / sphere.volume()

    return density, sphere, sigmas


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


def write_bunch(df, txt_file):
    df.insert(loc=0, column='z_m', value=df["z_um"] * 1e-6)
    df.insert(loc=0, column='y_m', value=df["y_um"] * 1e-6)
    df.insert(loc=0, column='x_m', value=df["x_um"] * 1e-6)


    df.drop(columns=['x_um', 'y_um', 'z_um'], inplace=True)

    df.to_csv(txt_file, sep='\t', float_format="%.6e", index=False, encoding="utf-8")


def bunch_openpmd_to_dataframe(series_path=pathlib.Path.cwd() / "bunch" / "data_%05T.h5", iteration=0):
    f = Series(str(series_path), Access.read_only)

    cur_it = f.iterations[iteration]
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
        "x_um": x_m_data * 1e6,  # meters to microns
        "y_um": y_m_data * 1e6,
        "z_um": z_m_data * 1e6,
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
        plot_width=4200,
        plot_height=350,
        x_range=(-900, 900),
        y_range=(50, 200),  # microns
    )
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

    df = read_bunch(job.fn("exp_4deg.txt"))
    # print(df.describe())
    del df

    write_bunch_openpmd(
        bunch_txt=job.fn("exp_4deg.txt"),
        outdir=pathlib.Path.cwd(),
        bunch_charge=-200.0e-12,  # Coulomb
    )
    plot_bunch_energy_histogram(
        opmd_dir=pathlib.Path.cwd() / "bunch",
        export_dir=pathlib.Path.cwd() / "bunch",
    )
    df = bunch_openpmd_to_dataframe(series_path=pathlib.Path(job.ws) / "bunch" / "data_%05T.h5")
    shade_bunch(df, "z_um", "x_um", export_path=pathlib.Path.cwd() / "bunch")
    bunch_centroid_plot(workdir=pathlib.Path.cwd() / "bunch")

    fbpic_df = bunch_openpmd_to_dataframe(series_path=pathlib.Path.cwd() / "diags" / "hdf5" / "data%08T.h5", iteration=20196)
    print(fbpic_df.describe())
    write_bunch(fbpic_df, pathlib.Path.cwd() / "bunch" / "out_bunch.txt" )


    rho, sph, stdxyz = bunch_density(df)
    print()
    print(
        f"Sphere centered at (x = {sph.x:.2f}, y = {sph.y:.2f}, z = {sph.z:.2f}), with radius {sph.r:.2f}."
    )
    print(f"Corresponding density is {rho.to(u.cm**(-3)):.2e}.")
    print()
    print(f"σ_x = {stdxyz[0]:.0f}; σ_y = {stdxyz[1]:.0f}; σ_z = {stdxyz[2]:.0f}")

    time_series = LpaDiagnostics(pathlib.Path.cwd() / "bunch", check_all_files=True)

    mean_gamma, gamma_std = time_series.get_mean_gamma(iteration=0, species="bunch")
    print()
    print(f"<𝛾> = {mean_gamma:.0f}; σ_<𝛾> = {gamma_std:.1f}")

    charge = time_series.get_charge(iteration=0, species="bunch") * u.Coulomb
    print()
    print("Q = {0:.1f}".format(charge.to("pC")))

    div_x, div_y = time_series.get_divergence(iteration=0, species="bunch") * u.radian
    print()
    print(
        "x/y-plane divergence {0:.3f}/{1:.3f}".format(
            div_x.to(u.mrad), div_y.to(u.mrad)
        )
    )

    ε_n_rms = (
        time_series.get_emittance(iteration=0, species="bunch") * u.meter * u.radian
    )  # π*m*rad
    print()
    print(
        "x/y-plane normalized emittance {0:.3f}/{1:.3f}".format(
            ε_n_rms[0].to(u.mm * u.mrad) / np.pi, ε_n_rms[1].to(u.mm * u.mrad) / np.pi
        )
    )

    # https://github.com/fbpic/fbpic/issues/385
    # TODO: use Gaussian density profile with 2 ramps, close to experiment
    # TODO: use Gaussian bunch, with parameters close to the experimental one
    # TODO: can we see hosing? can we see self-modulation?
    # TODO: find computable quantity in Remi's paper


if __name__ == "__main__":
    main()
