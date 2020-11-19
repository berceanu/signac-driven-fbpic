import numpy as np
from matplotlib import pyplot

# Define the density function
def dens_func( z, r ):
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
    # Make ramp up (note: use boosted-frame values of the ramp length)
    inv_ramp_up = 1./ramp_up
    n = np.where( z<ramp_up, z*inv_ramp_up, n )
    # Make ramp down
    inv_ramp_down = 1./ramp_down
    n = np.where( (z >= ramp_up+plateau) & \
                  (z < ramp_up+plateau+ramp_down),
              - (z - (ramp_up+plateau+ramp_down) )*inv_ramp_down, n )
    n = np.where( z >= ramp_up+plateau+ramp_down, 0, n)

    return n


def mark_on_plot(*, ax, parameter, y=1.1):
    ax.annotate(s=parameter, xy=(parameter * 1e6, y), xycoords="data")
    ax.axvline(x=parameter * 1e6, linestyle="--", color="red")
    return ax

if __name__ == "__main__":
    # The density profile
    ramp_up = .5e-3
    plateau = 3.5e-3
    ramp_down = .5e-3

    # The simulation box
    zmax = 0.e-6     # Length of the box along z (meters)
    zmin = -30.e-6

    # The particles of the plasma
    p_zmin = 0.e-6   # Position of the beginning of the plasma (meters)
    p_zmax = ramp_up + plateau + ramp_down
    p_rmax = 100.e-6 # Maximal radial position of the plasma (meters)
    n_e = 3.e24      # The density in the labframe (electrons.meters^-3)

    # plot density profile for checking
    all_z = np.linspace(zmin, p_zmax, 1000)
    dens = dens_func(all_z, 0.0)


    fig, ax = pyplot.subplots(figsize=(30, 4.8))
    ax.plot(all_z * 1e6, dens)
    ax.set_xlabel(r"$%s \;(\mu m)$" % "z")
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(zmin * 1e6 - 20, p_zmax * 1e6 + 20)
    ax.set_ylabel("Density profile $n$")

    mark_on_plot(ax=ax, parameter=zmin)
    mark_on_plot(ax=ax, parameter=zmax)
    mark_on_plot(ax=ax, parameter=p_zmin, y=0.9)
    # mark_on_plot(ax=ax, parameter=ramp_start, y=0.7)
    # mark_on_plot(ax=ax, parameter=L_interact)
    mark_on_plot(ax=ax, parameter=p_zmax)

    # ax.annotate(
    #     s="ramp_start + ramp_length",
    #     xy=(ramp_start * 1e6 + ramp_length * 1e6, 1.1),
    #     xycoords="data",
    # )
    # ax.axvline(
    #     x=ramp_start * 1e6 + ramp_length * 1e6,
    #     linestyle="--",
    #     color="red",
    # )

    ax.fill_between(all_z * 1e6, dens, alpha=0.5)

    fig.savefig("my_density.png")
    pyplot.close(fig)
