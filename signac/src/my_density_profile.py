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

    def ramp(z, *, center, sigma, p):
        """Gaussian-like function."""
        return np.exp(-(((z - center) / sigma) ** p))

    # Allocate relative density
    n = np.ones_like(z)

    # before up-ramp
    n = np.where(z < 0.0, 0.0, n)

    # Make up-ramp
    n = np.where(
        z < center_left, ramp(z, center=center_left, sigma=sigma_left, p=power), n
    )

    # Make down-ramp
    n = np.where(
        (z >= center_right) & (z < center_right + 2 * sigma_right),
        ramp(z, center=center_right, sigma=sigma_right, p=power),
        n,
    )

    # after down-ramp
    n = np.where(z >= center_right + 2 * sigma_right, 0, n)

    return n


def mark_on_plot(*, ax, parameter, y=1.1):
    ax.annotate(s=parameter, xy=(eval(parameter) * 1e6, y), xycoords="data")
    ax.axvline(x=eval(parameter) * 1e6, linestyle="--", color="red")
    return ax


if __name__ == "__main__":
    # The density profile
    flat_top_dist = 1000.0e-6  # plasma flat top distance
    center_left = 1000.0e-6
    center_right = center_left + flat_top_dist
    sigma_left = 500.0e-6
    sigma_right = 500.0e-6
    power = 4.0

    # The simulation box
    zmax = 0.0e-6  # Length of the box along z (meters)
    zmin = -30.0e-6

    # The particles of the plasma
    p_zmin = 0.0e-6  # Position of the beginning of the plasma (meters)
    p_zmax = center_right + 2 * sigma_right
    p_rmax = 100.0e-6  # Maximal radial position of the plasma (meters)
    n_e = 5.307e18 * 1.0e6  # The density in the labframe (electrons.meters^-3)

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
    mark_on_plot(ax=ax, parameter="center_left", y=0.7)
    mark_on_plot(ax=ax, parameter="center_right", y=0.7)
    mark_on_plot(ax=ax, parameter="L_interact", y=0.7)
    mark_on_plot(ax=ax, parameter="p_zmax")

    ax.fill_between(all_z * 1e6, dens, alpha=0.5)

    fig.savefig("my_density.png")
    pyplot.close(fig)
