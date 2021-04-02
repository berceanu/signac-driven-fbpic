"""Repository of `fbpic` density functions."""
from itertools import cycle

import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec


def read_longitudinal_profile(txt_file="longitudinal_density.txt", offset=20):
    return read_density(txt_file=txt_file, offset=offset)


def read_transverse_profile(txt_file="transverse_density.txt", offset=-0.25):
    return read_density(txt_file=txt_file, offset=offset)


def plot_profile(position_m, norm_density, axs, y_offset=0.0):

    axs.plot(position_m * 1e6, norm_density + y_offset)

    # axs.set_xlabel("position_m")
    # axs.set_ylabel("norm_density")

    # axs.set_xlim(-80, 40_000 + 20)
    # axs.set_ylim(0.0, 1.2)

    return axs


def read_density(txt_file, every_nth=1, offset=0.0):
    df = pd.read_csv(
        txt_file,
        delim_whitespace=True,
        header=0,
    )
    # substract offset
    df.position_mm = df.position_mm + offset

    # convert to meters
    df["position_m"] = df.position_mm * 1e-3

    # normalize density
    df["norm_density"] = df.density_cm_3 / df.density_cm_3.max()
    # check density values between 0 and 1
    if not df.norm_density.between(0, 1).any():
        raise ValueError("The density contains values outside the range [0,1].")

    # return every nth item
    df = df.iloc[::every_nth, :]

    # return data as numpy arrays
    return df.position_m.values, df.norm_density.values


def make_gaussian_dens_func(job):
    def ramp(z, *, center, sigma, p):
        """Gaussian-like function."""
        return np.exp(-((np.abs(z - center) / sigma) ** p))

    # The density profile
    def dens_func(z, r):
        """
        User-defined function: density profile of the plasma

        It should return the relative density with respect to n_plasma,
        at the position x, y, z (i.e. return a number between 0 and 1)

        Parameters
        ----------
        z, r: 1darrays of floats
            Arrays with one element per macroparticle
        Returns
        -------
        n : 1d array of floats
            Array of relative density, with one element per macroparticles
        """

        # Allocate relative density
        n = np.ones_like(z)

        # before up-ramp
        n = np.where(z < 0.0, 0.0, n)

        # Make up-ramp
        n = np.where(
            z < job.sp.center_left,
            ramp(z, center=job.sp.center_left, sigma=job.sp.sigma_left, p=job.sp.power),
            n,
        )

        # Make down-ramp
        n = np.where(
            (z >= job.sp.center_right)
            & (z < job.sp.center_right + 2 * job.sp.sigma_right),
            ramp(
                z, center=job.sp.center_right, sigma=job.sp.sigma_right, p=job.sp.power
            ),
            n,
        )

        # after down-ramp
        n = np.where(z >= job.sp.center_right + 2 * job.sp.sigma_right, 0, n)

        # Add transverse guiding parabolic profile
        n = n * (1.0 + 0.25 * r ** 2 / job.sp.rmax ** 2)

        return n

    return dens_func


def plot_density_profile(profile_maker, fig_fname, job):
    """Plot the plasma density profile."""

    def mark_on_plot(*, ax, parameter: str, y=1.1):
        ax.annotate(text=parameter, xy=(job.sp[parameter] * 1e6, y), xycoords="data")
        ax.axvline(
            x=job.sp[parameter] * 1e6, linewidth=1, linestyle="dashed", color="0.75"
        )
        return ax

    num = int((job.sp.L_interact - job.sp.zmin) * 1e6 / 100 + 1)
    all_z = np.linspace(job.sp.zmin, job.sp.L_interact, num)
    all_r = np.linspace(-job.sp.rmax, job.sp.rmax, 512)
    dens_z = profile_maker(job)(all_z, 0.0)
    dens_r = profile_maker(job)(20e-3, all_r)

    fig = pyplot.figure(figsize=(30, 6.8))
    G = GridSpec(2, 1, figure=fig)
    ax_top = fig.add_subplot(G[0, :])
    ax_bottom = fig.add_subplot(G[1, :])

    for ax in (ax_top, ax_bottom):
        ax.set_ylim(0.0, 1.25)
        ax.set_ylabel("Density profile $n$")

    ax_top.plot(all_z * 1e6, dens_z, marker="o", linestyle="", markersize=5, alpha=0.2)
    ax_top.fill_between(all_z * 1e6, dens_z, alpha=0.3)
    ax_top.set_xlim(job.sp.zmin * 1e6 - 20, job.sp.L_interact * 1e6 + 20)
    ax_top.set_xlabel(r"$%s \;(\mu m)$" % "z")

    ax_bottom.plot(
        all_r * 1e6, dens_r, marker="o", linestyle="", markersize=5, alpha=0.2
    )
    ax_bottom.fill_between(all_r * 1e6, dens_r, alpha=0.3)
    ax_bottom.set_xlim(-job.sp.rmax * 1e6, job.sp.rmax * 1e6)
    ax_bottom.set_xlabel(r"$%s \;(\mu m)$" % "r")

    params_to_annotate = (
        "zmin",
        "zmax",
        "p_zmin",
        "p_zmax",
        "L_interact",
    )
    y_annotation_positions = (0.5, 0.7, 0.9, 1.1)
    pos_cycle = cycle(y_annotation_positions)
    #
    params_and_positions = [(p, next(pos_cycle)) for p in params_to_annotate]
    #
    for p, y_pos in params_and_positions:
        mark_on_plot(ax=ax_top, parameter=p, y=y_pos)

    ax_top.hlines(
        y=1.0,
        xmin=all_z[0] * 1e6,
        xmax=all_z[-1] * 1e6,
        linewidth=0.75,
        linestyle="dashed",
        color="0.75",
    )

    position_m, norm_density = read_longitudinal_profile()
    plot_profile(position_m, norm_density, ax_top)

    position_m, norm_density = read_transverse_profile()
    plot_profile(position_m, norm_density, ax_bottom, y_offset=0.25)

    fig.savefig(fig_fname)
    pyplot.close(fig)


def main():
    import random

    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    ids = [job.id for job in proj]
    job = proj.open_job(id=random.choice(ids))

    fig = pyplot.figure(figsize=(30, 4.8))
    axs = fig.add_subplot(111)
    plot_profile(*read_longitudinal_profile(), axs)
    fig.savefig("longitudinal.png")

    fig = pyplot.figure(figsize=(30, 4.8))
    axs = fig.add_subplot(111)
    plot_profile(*read_transverse_profile(), axs)
    fig.savefig("transverse.png")

    plot_density_profile(make_gaussian_dens_func, "initial_density_profile.png", job)


if __name__ == "__main__":
    main()
