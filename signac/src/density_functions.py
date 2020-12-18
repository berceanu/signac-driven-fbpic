"""Repository of `fbpic` density functions."""
from itertools import cycle
import numpy as np
import pandas as pd
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec
from scipy import interpolate


def read_density(txt_file, every_nth=20, offset=True):
    df = pd.read_csv(
        txt_file,
        delim_whitespace=True,
        names=["position_mm", "density_cm_3", "error_density_cm_3"],
    )

    # convert to meters
    df["position_m"] = df.position_mm * 1e-3

    # substract offset
    if offset:
        df.position_m = df.position_m - df.position_m.iloc[0]

    # normalize density
    df["norm_density"] = df.density_cm_3 / df.density_cm_3.max()
    # check density values between 0 and 1
    if not df.norm_density.between(0, 1).any():
        raise ValueError("The density contains values outside the range [0,1].")

    # return every nth item
    df = df.iloc[::every_nth, :]

    # return data as numpy arrays
    return df.position_m.to_numpy(), df.norm_density.to_numpy()


def make_experimental_dens_func(job):
    position_m, norm_density = read_density(job.fn("density_1_inlet_spacers.txt"))

    interp_z_min = position_m.min()
    interp_z_max = position_m.max()

    rho = interpolate.interp1d(
        position_m, norm_density, bounds_error=False, fill_value=(0.0, 0.0)
    )

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

        # only compute n if z is inside the interpolation bounds
        n = np.where(np.logical_and(z >= interp_z_min, z <= interp_z_max), rho(z), n)

        # Make linear ramp
        n = np.where(
            z < job.sp.ramp_start + job.sp.ramp_length,
            (z - job.sp.ramp_start) / job.sp.ramp_length * rho(interp_z_min),
            n,
        )

        # Supress density before the ramp
        n = np.where(z < job.sp.ramp_start, 0.0, n)

        return n

    return dens_func


def make_gaussian_dens_func(job):
    def ramp(z, *, center, sigma, p):
        """Gaussian-like function."""
        return np.exp(-(((z - center) / sigma) ** p))

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

    all_z = np.linspace(job.sp.zmin, job.sp.L_interact, 1000)
    dens = profile_maker(job)(all_z, 0.0)

    fig = pyplot.figure(figsize=(30, 4.8))
    G = GridSpec(2, 1, figure=fig)
    ax_top = fig.add_subplot(G[0, :])
    ax_bottom = fig.add_subplot(G[1, :])

    for ax in (ax_top, ax_bottom):
        ax.plot(all_z * 1e6, dens)
        ax.fill_between(all_z * 1e6, dens, alpha=0.3)
        ax.set_xlabel(r"$%s \;(\mu m)$" % "z")
        ax.set_ylim(0.0, 1.2)
        ax.set_xlim(left=job.sp.zmin * 1e6 - 20)
        ax.set_ylabel("Density profile $n$")

    ax_top.set_xlim(right=job.sp.L_interact * 1e6 + 20)
    ax_bottom.set_xlim(right=(job.sp.ramp_start + job.sp.ramp_length) * 1e6)

    params_to_annotate = (
        "zmin",
        "ramp_start",
        "zmax",
        "p_zmin",
        "p_zmax",
        "L_interact",
    )
    y_annotation_positions = (0.5, 0.7, 0.9, 1.1)
    pos_cycle = cycle(y_annotation_positions)

    params_and_positions = [(p, next(pos_cycle)) for p in params_to_annotate]

    for p, y_pos in params_and_positions:
        for ax in (ax_top, ax_bottom):
            mark_on_plot(ax=ax, parameter=p, y=y_pos)

    ax_top.hlines(
        y=1.0,
        xmin=all_z[0] * 1e6,
        xmax=all_z[-1] * 1e6,
        linewidth=0.75,
        linestyle="dashed",
        color="0.75",
    )

    fig.savefig(fig_fname)
    pyplot.close(fig)


def main():
    import random
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    ids = [job.id for job in proj]
    job = proj.open_job(id=random.choice(ids))

    plot_density_profile(
        make_experimental_dens_func, "initial_density_profile.png", job
    )


if __name__ == "__main__":
    main()