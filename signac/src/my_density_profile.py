import numpy as np
from matplotlib import pyplot

# Define the density function
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

    # Make ramp up
    inv_ramp_up = 1.0 / ramp_up
    n = np.where(z < ramp_up, z * inv_ramp_up, n)

    # Make ramp down
    inv_ramp_down = 1.0 / ramp_down
    n = np.where(
        (z >= ramp_up + plateau) & (z < ramp_up + plateau + ramp_down),
        -(z - (ramp_up + plateau + ramp_down)) * inv_ramp_down,
        n,
    )

    n = np.where(z >= ramp_up + plateau + ramp_down, 0, n)

    return n


def mark_on_plot(*, ax, parameter, y=1.1):
    ax.annotate(s=parameter, xy=(eval(parameter) * 1e6, y), xycoords="data")
    ax.axvline(x=eval(parameter) * 1e6, linestyle="--", color="red")
    return ax


if __name__ == "__main__":
    # The density profile
    ramp_up = 0.5e-3
    plateau = 3.5e-3
    ramp_down = 0.5e-3

    # The simulation box
    zmax = 0.0e-6  # Length of the box along z (meters)
    zmin = -30.0e-6

    # The particles of the plasma
    p_zmin = 0.0e-6  # Position of the beginning of the plasma (meters)
    p_zmax = ramp_up + plateau + ramp_down
    p_rmax = 100.0e-6  # Maximal radial position of the plasma (meters)
    n_e = 3.0e24  # The density in the labframe (electrons.meters^-3)

    L_interact = p_zmax - p_zmin  # the plasma length

    # plot density profile for checking
    all_z = np.linspace(zmin, L_interact, 1000)
    dens = dens_func(all_z, 0.0)

    fig, ax = pyplot.subplots(figsize=(30, 4.8))

    ax.plot(all_z * 1e6, dens)

    ax.set_xlabel(r"$%s \;(\mu m)$" % "z")
    ax.set_ylim(0.0, 1.2)
    ax.set_xlim(zmin * 1e6 - 20, L_interact * 1e6 + 20)
    ax.set_ylabel("Density profile $n$")

    mark_on_plot(ax=ax, parameter="zmin")
    mark_on_plot(ax=ax, parameter="zmax", y=0.7)
    mark_on_plot(ax=ax, parameter="p_zmin", y=0.9)
    mark_on_plot(ax=ax, parameter="ramp_up", y=0.7)
    mark_on_plot(ax=ax, parameter="L_interact", y=0.7)
    mark_on_plot(ax=ax, parameter="p_zmax")

    ax.annotate(
        s="ramp_up + plateau", xy=((ramp_up + plateau) * 1e6, 1.1), xycoords="data",
    )
    ax.axvline(
        x=(ramp_up + plateau) * 1e6, linestyle="--", color="red",
    )

    ax.fill_between(all_z * 1e6, dens, alpha=0.5)

    fig.savefig("my_density.png")
    pyplot.close(fig)
