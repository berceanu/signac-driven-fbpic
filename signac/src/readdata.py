"""Fit the centroid positions for a given electron bunch."""
import pathlib
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import colorcet as cc


def read_bunch(txt_file):
    df = pd.read_csv(
        txt_file,
        delim_whitespace=True,
        names=["x_m", "y_m", "z_m", "ux", "uy", "uz", "w"],
        header=0,
    )
    return df


def convert_to_human_units(bunch_df):
    bunch_df.insert(loc=0, column="z_mm", value=bunch_df["z_m"] * 1e3)
    bunch_df.insert(loc=0, column="y_um", value=bunch_df["y_m"] * 1e6)
    bunch_df.insert(loc=0, column="x_um", value=bunch_df["x_m"] * 1e6)

    bunch_df.drop(columns=["x_m", "y_m", "z_m"], inplace=True)


def compute_bunch_histogram(txt_file):
    df = read_bunch(txt_file)
    convert_to_human_units(df)

    pos_x = df.x_um.to_numpy(dtype=np.float64)
    pos_z = df.z_mm.to_numpy(dtype=np.float64)

    H, zedges, xedges = np.histogram2d(
        pos_z,
        pos_x,
        bins=(190, 190),
    )
    Z, X = np.meshgrid(zedges, xedges)

    return H.T, Z, X


def plot_bunch_histogram(H, Z, X, ax=None):
    if ax is None:
        ax = pyplot.gca()

    img = ax.pcolormesh(Z, X, H, cmap=cc.m_fire)
    cbaxes = inset_axes(
        ax,
        width="3%",  # width = 10% of parent_bbox width
        height="100%",  # height : 50%
        loc=2,
        bbox_to_anchor=(1.01, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar = ax.get_figure().colorbar(
        mappable=img, orientation="vertical", ticklocation="right", cax=cbaxes
    )
    cbar.set_label(r"number of electrons")

    ax.set_xlabel(r"$z$ ($\mathrm{mm}$)")
    ax.set_ylabel(r"$x$ ($\mathrm{\mu m}$)")

    return ax


def compute_centroid(H, Z, X):
    centroid = np.ma.average(X[:-1, :-1], weights=H, axis=0)
    return centroid


def readbeam2(
    z_coords,
    x_coords,
    counts,
    nbz=190,
    nbx=190,
):
    refval = 0

    centroid = []
    we = []
    centroid_z = []

    for i in range(0, nbz):
        if refval < max(counts[i, :]):
            refval = max(counts[i, :])

    for i in range(0, nbz):
        counts[i, :][counts[i, :] < 0.15 * refval] = 0
        if max(counts[i, :]) > 0.2 * refval and i > 20 and i < 180:
            centroid.append(np.average(x_coords, weights=counts[i, :]))
            centroid_z.append(z_coords[i])
            we.append(sum(counts[i, :]))

    return centroid_z, centroid


def readbeam(datadir):
    """Given path to .txt data file, fit centroid."""
    data = np.loadtxt(datadir, skiprows=1)

    x = data[:, 0]
    z = data[:, 2]

    nbz = 190
    nbx = 190
    z_min = min(z)
    z_max = max(z)
    x_min = min(x)
    x_max = max(x)

    fig, ax = pyplot.subplots()

    h = ax.hist2d(z, x, bins=(nbz, nbx), range=[[z_min, z_max], [x_min, x_max]])
    counts = h[0]
    z_coords = np.linspace(z_min, z_max, nbz)
    x_coords = np.linspace(x_min, x_max, nbx)

    refval = 0

    centroid = []
    we = []
    centroid_z = []

    for i in range(0, nbz):
        if refval < max(counts[i, :]):
            refval = max(counts[i, :])

    for i in range(0, nbz):
        counts[i, :][counts[i, :] < 0.15 * refval] = 0
        if max(counts[i, :]) > 0.2 * refval and i > 20 and i < 180:
            centroid.append(np.average(x_coords, weights=counts[i, :]))
            centroid_z.append(z_coords[i])
            we.append(sum(counts[i, :]))

    ax.plot(centroid_z, centroid)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("x (m)")

    fig.savefig("alessio.png", bbox_inches="tight")
    pyplot.close(fig)

    m_cent = abs(np.mean(centroid))
    m_cent_w = abs(np.average(centroid, weights=we))

    return m_cent, m_cent_w


def main():
    """Main entry point."""
    p = pathlib.Path.cwd() / "final_bunch_66dc81.txt"

    m_cent, m_cent_w = readbeam(p)
    print(m_cent, m_cent_w)

    H, Z, X = compute_bunch_histogram(p)
    r, c = Z.shape

    centroid = compute_centroid(H, Z, X)
    centroid_z, centroid_ale = readbeam2(
        Z[r // 2, :-1].copy(), X[:-1, c // 2].copy(), H.T.copy()
    )

    fig, ax = pyplot.subplots()

    ax = plot_bunch_histogram(H, Z, X, ax)
    ax.plot(Z[r // 2, :-1], centroid)
    ax.plot(centroid_z, centroid_ale)

    fig.savefig("not_alessio.png", bbox_inches="tight")
    pyplot.close(fig)


if __name__ == "__main__":
    main()
