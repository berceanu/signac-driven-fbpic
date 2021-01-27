"""Fit the centroid positions for a given electron bunch."""
import pathlib
import numpy as np
from matplotlib import pyplot, cm, transforms, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy.ma.extras import mask_cols
import pandas as pd
from statepoints_parser import parse_statepoints
import unyt as u
from scipy.constants import golden
from util import ffmpeg_command, shell_run


def read_bunch(txt_file):
    """Read positions, velocities and weights of bunch electrons from the text file."""
    df = pd.read_csv(
        txt_file,
        delim_whitespace=True,
        names=["x_m", "y_m", "z_m", "ux", "uy", "uz", "w"],
        header=0,
    )
    return df


def convert_to_human_units(bunch_df):
    """Convert the x and y coordinates from meters to microns and z to mm."""
    bunch_df.insert(loc=0, column="z_mm", value=bunch_df["z_m"] * 1e3)
    bunch_df.insert(loc=0, column="y_um", value=bunch_df["y_m"] * 1e6)
    bunch_df.insert(loc=0, column="x_um", value=bunch_df["x_m"] * 1e6)

    bunch_df.drop(columns=["x_m", "y_m", "z_m"], inplace=True)


def bin_centers(bin_edges):
    """Given the bin edges from a histogram, compute the bin centers.
    Array size is reduced by 1 element."""
    return (bin_edges[:-1] + bin_edges[1:]) / 2


def compute_bunch_histogram(txt_file, *, nbx=200, nbz=200, range=None):
    """Compute the electron bunch x vs z 2D histogram, with given number of bins."""
    df = read_bunch(txt_file)
    convert_to_human_units(df)

    pos_x = df.x_um.to_numpy(dtype=np.float64)
    pos_z = df.z_mm.to_numpy(dtype=np.float64)
    w = df.w.to_numpy(dtype=np.float64)

    H, zedges, xedges = np.histogram2d(
        pos_z,
        pos_x,
        bins=(nbz, nbx),
        weights=w,  # convert to count of real electrons
        range=range,
    )
    Z, X = np.meshgrid(zedges, xedges)

    z_centers, x_centers = map(bin_centers, (zedges, xedges))

    H = H.T  # Let each row list bins with common x range.

    return H, Z, X, z_centers, x_centers


def plot_bunch_histogram(H, Z, X, *, ax=None, vmin=0.0, vmax=None):
    """Given the histogram data H and the (2D) bin edges Z and X, plot them."""
    if ax is None:
        ax = pyplot.gca()

    img = ax.pcolormesh(
        Z,
        X,
        H,
        cmap=cm.get_cmap("magma"),
        norm=colors.Normalize(vmin=vmin, vmax=vmax),
    )

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


def bunch_centroid(
    z_coords,
    x_coords,
    counts,
    *,
    z_min_index=-1,
    z_max_index=None,
    threshold_col_max=-1,
    lower_bound=-1,
):
    """
    Refactored, vectorized function for computing the electron bunch centroid.

    Parameters
    ----------
    z_coords : ndarray
        1D array of `float` containing z coordinates
    x_coords : ndarray
        1D array of `float` containing x coordinates
    counts : ndarray
        2D array of `float` containing histogram data (electron counts)

    Returns
    -------
    z_coords_masked, weighted_average_masked
        2 1D masked arrays of `float` representing the centroid coordinates and values
    """

    _, nbz = counts.shape
    max_count = counts.max()

    if z_max_index is None:
        z_max_index = nbz

    counts_masked = np.ma.masked_all_like(counts)

    z_index = np.arange(nbz)
    border_mask = np.logical_and(z_min_index < z_index, z_index < z_max_index)

    mask_2d = counts < lower_bound * max_count
    col_mask = np.logical_and(
        border_mask, counts.max(axis=0) > threshold_col_max * max_count
    )

    counts_masked[:, col_mask] = counts[:, col_mask]
    counts_masked.mask = np.ma.mask_or(np.ma.getmask(counts_masked), mask_2d)

    weighted_average_masked = (
        np.sum(counts_masked * x_coords[:, np.newaxis], axis=0)
    ) / np.sum(
        counts_masked, axis=0
    )  # axis = 0 sums the values in each column

    z_coords_masked = np.ma.masked_where(
        np.ma.getmask(weighted_average_masked), z_coords
    )

    return z_coords_masked, weighted_average_masked


def main():
    """Main entry point."""
    p = pathlib.Path.cwd()
    pathlib.Path(p / "bunch_centroid").mkdir(parents=True, exist_ok=True)

    runs_dir = pathlib.Path.home() / "tmp" / "runs"
    txt_files = runs_dir.glob("final_bunch_*.txt")
    sorted_bunch_fn_to_density = parse_statepoints(runs_dir)

    tfs = list(txt_files)
    ordered_tfs = sorted(tfs, key=lambda tf: sorted_bunch_fn_to_density[tf.name])

    number_of_jobs = len(sorted_bunch_fn_to_density)
    job_densities = np.empty(number_of_jobs) * u.cm ** (-3)
    job_centroid_positions = np.empty(number_of_jobs) * u.micrometer

    for i, fn in enumerate(ordered_tfs):
        n_e = sorted_bunch_fn_to_density[fn.name]
        print(f"{fn.name} -> {n_e:.1e}")

        H, Z, X, z_coords, x_coords = compute_bunch_histogram(
            fn,
            nbx=200,
            nbz=200,
            range=[[70.0, 72.0], [-900, 600]],
        )

        centroid_z, centroid = bunch_centroid(
            z_coords,
            x_coords,
            H,
            z_min_index=20,
            z_max_index=180,
            threshold_col_max=0.2,
            lower_bound=0.15,
        )
        x_avg = centroid.mean()
        job_centroid_positions[i] = x_avg * u.micrometer
        job_densities[i] = n_e

        fig, ax = pyplot.subplots()

        ax = plot_bunch_histogram(H, Z, X, ax=ax, vmax=1.0e6)
        ax.plot(centroid_z, centroid, "o", markersize=3, label="centroid")
        ax.legend()
        ax.hlines(
            y=x_avg,
            xmin=z_coords[0],
            xmax=z_coords[-1],
            linestyle="dashed",
            color="0.75",
        )
        trans = transforms.blended_transform_factory(
            ax.get_yticklabels()[0].get_transform(), ax.transData
        )
        ax.text(
            0,
            x_avg,
            f"{x_avg:.0f}",
            color="0.75",
            transform=trans,
            ha="right",
            va="center",
        )
        ax.annotate(
            text=f"ne = {n_e:.1e}",
            xy=(0.1, 0.1),
            xycoords="axes fraction",
            color="C1",
        )
        n_e = f"{n_e.to_value(u.cm**(-3)):.1e}"
        fig.savefig(f"bunch_centroid_{n_e}.png", bbox_inches="tight")
        fig.savefig(
            pathlib.Path.cwd() / "bunch_centroid" / f"bunch_centroid_{i:06d}.png",
            bbox_inches="tight",
        )
        pyplot.close(fig)

    # x, y = zip(*sorted(zip(job_densities, job_centroid_positions)))
    x, y = job_densities, job_centroid_positions

    fig, ax = pyplot.subplots(figsize=(golden * 4, 4))
    ax.plot(x, y, "X--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ (cm${}^{-3}$)")
    ax.set_ylabel(r"$\langle x \rangle$ ($\mathrm{\mu m}$)")

    ax.grid(which="both")

    for x_c, y_c in zip(x, y):
        ax.annotate(f"{x_c.value:.1e}", xy=(x_c, y_c), textcoords='data', fontsize=6)

    fig.savefig("average_centroids.png", bbox_inches="tight")
    pyplot.close(fig)

    command = ffmpeg_command(
        input_files=pathlib.Path.cwd() / "bunch_centroid" / "bunch_centroid_*.png",
        output_file="bunch_centroid.mp4",
    )
    shell_run(command, shell=True)


if __name__ == "__main__":
    main()
