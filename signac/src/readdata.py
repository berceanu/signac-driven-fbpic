"""Fit the centroid positions for a given electron bunch."""
import pathlib
import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


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


def bin_centers(bin_edges):
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def compute_bunch_histogram(txt_file, *, nbx=200, nbz=200):
    df = read_bunch(txt_file)
    convert_to_human_units(df)

    pos_x = df.x_um.to_numpy(dtype=np.float64)
    pos_z = df.z_mm.to_numpy(dtype=np.float64)
    w = df.w.to_numpy(dtype=np.float64)

    H, zedges, xedges = np.histogram2d(
        pos_z,
        pos_x,
        bins=(nbz, nbx),
        weights=w,
    )
    Z, X = np.meshgrid(zedges, xedges)

    z_centers, x_centers = map(bin_centers, (zedges, xedges))

    H = H.T  # Let each row list bins with common x range.

    return H, Z, X, z_centers, x_centers


def plot_bunch_histogram(H, Z, X, *, ax=None):
    if ax is None:
        ax = pyplot.gca()

    img = ax.pcolormesh(Z, X, H, cmap=cm.get_cmap("magma"))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.02)
    cbar = ax.figure.colorbar(
        img,
        cax=cax,
    )
    cbar.set_label(r"number of electrons")

    ax.set_xlabel(r"$z$ ($\mathrm{mm}$)")
    ax.set_ylabel(r"$x$ ($\mathrm{\mu m}$)")

    return ax


def readbeam(
    z_coords,
    x_coords,
    counts,
):
    nbz, _ = counts.shape

    refval = counts.max()

    centroid = []
    centroid_z = []

    counts[counts < 0.15 * refval] = 0.0

    for i in range(0, nbz):
        if max(counts[i, :]) > 0.2 * refval and i > 20 and i < 180:
            centroid.append(np.average(x_coords, weights=counts[i, :]))
            centroid_z.append(z_coords[i])

    return centroid_z, centroid


def vectorized_readbeam(
    z_coords,
    x_coords,
    counts,
    *,
    z_min_index=20,
    z_max_index=180,
    col_max_threshold=0.2,
    lower_bound=0.15,
):
    _, nbz = counts.shape
    z_coords_masked = np.ma.masked_all((nbz,), dtype=z_coords.dtype)
    centroid_masked = np.ma.masked_all((nbz,), dtype=counts.dtype)

    z_index = np.arange(nbz)
    border_mask = np.logical_and(z_index > z_min_index, z_index < z_max_index)

    filtered_counts = np.ma.array(counts, mask=counts < lower_bound * counts.max())
    col_mask = np.logical_and(
        border_mask, filtered_counts.max(axis=0) > col_max_threshold * counts.max()
    )

    counts_mask = filtered_counts[:, col_mask]
    weighted_average = (np.sum(counts_mask * x_coords[:, np.newaxis], axis=0)) / np.sum(
        counts_mask, axis=0
    )  # axis = 0 sums the values in each column

    centroid_masked[col_mask] = weighted_average
    z_coords_masked[col_mask] = z_coords[col_mask]

    return z_coords_masked, centroid_masked


def main():
    """Main entry point."""
    p = pathlib.Path.cwd() / "final_bunch_66dc81.txt"

    H, Z, X, z_coords, x_coords = compute_bunch_histogram(p, nbx=200, nbz=200)

    centroid_z_cut, centroid_cut = vectorized_readbeam(z_coords, x_coords, H)
    # must pass in a copy, as the original array is changed in-place!
    orig_centroid_z_cut, orig_centroid_cut = readbeam(z_coords, x_coords, H.T.copy())

    fig, ax = pyplot.subplots()

    ax = plot_bunch_histogram(H, Z, X, ax=ax)

    ax.plot(centroid_z_cut, centroid_cut, "o", markersize=4, label="vectorized")
    ax.plot(orig_centroid_z_cut, orig_centroid_cut, "s", markersize=1, label="original")

    ax.legend()

    fig.savefig("bunch_fit.png", bbox_inches="tight")
    pyplot.close(fig)


if __name__ == "__main__":
    main()
