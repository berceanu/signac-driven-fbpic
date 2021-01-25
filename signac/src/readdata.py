"""Fit the centroid positions for a given electron bunch."""
import pathlib
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from numpy.core.defchararray import count
import pandas as pd
import colorcet as cc
import numpy.ma as ma


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
    w = df.w.to_numpy(dtype=np.float64)

    H, zedges, xedges = np.histogram2d(
        pos_z,
        pos_x,
        bins=(190, 190),
        weights=w,
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
    return Z[Z.shape[0] // 2, :-1], centroid


def orig_readbeam(
    z_coords,
    x_coords,
    counts,
    nbz=190,
):
    refval = counts.max()

    centroid = []
    centroid_z = []

    counts[counts < 0.15 * refval] = 0.0

    for i in range(0, nbz):
        if max(counts[i, :]) > 0.2 * refval and i > 20 and i < 180:
            centroid.append(np.average(x_coords, weights=counts[i, :]))
            centroid_z.append(z_coords[i])

    return centroid_z, centroid


def readbeam(
    z_coords,
    x_coords,
    counts,
):
    nbz, nbx = counts.shape
    z_coords_masked = ma.masked_all((nbz,), dtype=z_coords.dtype)
    centroid_masked = ma.masked_all((nbz,), dtype=counts.dtype)

    i = np.arange(nbz)
    border_mask = np.logical_and(i > 20, i < 180)

    refval = counts.max()
    filtered_counts = ma.array(counts, mask=counts < 0.15 * refval)
    row_mask = np.logical_and(border_mask, filtered_counts.max(axis=1) > 0.2 * refval)

    counts_mask = filtered_counts[row_mask, :]
    weighted_average = (np.sum(x_coords * counts_mask, axis=1)) / np.sum(
        counts_mask, axis=1
    )

    centroid_masked[row_mask] = weighted_average
    z_coords_masked[row_mask] = z_coords[row_mask]

    return z_coords_masked, centroid_masked


def main():
    """Main entry point."""
    p = pathlib.Path.cwd() / "final_bunch_66dc81.txt"

    H, Z, X = compute_bunch_histogram(p)
    z_centroid, centroid = compute_centroid(H, Z, X)

    centroid_z_cut, centroid_cut = readbeam(
        Z[Z.shape[0] // 2, :-1], X[:-1, X.shape[1] // 2], H.T
    )
    orig_centroid_z_cut, orig_centroid_cut = orig_readbeam(
        Z[Z.shape[0] // 2, :-1].copy(), X[:-1, X.shape[1] // 2].copy(), H.T.copy()
    )

    fig, ax = pyplot.subplots()

    ax = plot_bunch_histogram(H, Z, X, ax)

    # ax.plot(z_centroid, centroid, label="no cut")
    ax.plot(centroid_z_cut, centroid_cut, "o", markersize=5, label="vectorized")
    ax.plot(orig_centroid_z_cut, orig_centroid_cut, "s", markersize=2, label="original")

    ax.legend()

    fig.savefig("bunch_fit.png", bbox_inches="tight")
    pyplot.close(fig)


if __name__ == "__main__":
    main()
