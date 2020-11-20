from collections import namedtuple
import numpy as np
import unyt as u
from matplotlib import pyplot
from prepic import GaussianBeam, Laser, Plasma, Simulation


def mark_on_plot(*, ax, parameter, y=1.1):
    ax.annotate(parameter, xy=(eval(parameter) * 1e6, y), xycoords="data")
    ax.axvline(x=eval(parameter) * 1e6, linestyle="--", color="red")
    return ax


E4Params = namedtuple(
    "E4Params",
    [
        "npe",  # electron plasma density
        "w0",  # laser beam waist (Gaussian beam assumed)
        "ɛL",  # laser energy on target (focused into the FWHM@intensity spot)
        "τL",  # laser pulse duration (FWHM@intensity)
        "prop_dist",  # laser propagation distance (acceleration length)
    ],
)

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


if __name__ == "__main__":
    # The simulation box
    Nz = 2425         # Number of gridpoints along z

    zmax = 0.0e-6  # Length of the box along z (meters)
    zmin = -100.0e-6
    Nr = 420          # Number of gridpoints along r
    rmax = 150.e-6   # Length of the box along r (meters)
    Nm = 2           # Number of modes used

    dt = (zmax-zmin)/Nz/c  # Timestep (seconds)

    # The density profile
    flat_top_dist = 1000.0e-6  # plasma flat top distance
    center_left = 1000.0e-6
    center_right = center_left + flat_top_dist
    sigma_left = 500.0e-6
    sigma_right = 500.0e-6
    power = 4.0

    # The particles of the plasma
    p_zmin = 0.0e-6  # Position of the beginning of the plasma (meters)
    p_zmax = center_right + 2 * sigma_right
    p_rmax = 100.0e-6  # Maximal radial position of the plasma (meters)
    n_e = 5.307e18 * 1.0e6  # The density in the labframe (electrons.meters^-3)
    p_nz = 2         # Number of particles per cell along z
    p_nr = 2         # Number of particles per cell along r
    p_nt = 6         # Number of particles per cell along theta

    # The laser
    a0 = 2.4         # Laser amplitude
    w0 = 18.7e-6     # Laser waist
    ctau = 7.495e-6  # Laser duration
    z0 = -10.e-6     # Laser centroid
    zfoc = 0.e-6     # Focal position
    lambda0 = 0.8e-6 # Laser wavelength

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

    param = E4Params(
        npe=n_e / u.meter ** 3,
        w0=18.7 * u.micrometer,
        ɛL=1.8 * u.joule,
        τL=25 * u.femtosecond,
        prop_dist=flat_top_dist * u.meter,
    )

    E4Params_beam = GaussianBeam(w0=param.w0)
    E4Params_laser = Laser(ɛL=param.ɛL, τL=param.τL, beam=E4Params_beam)
    E4Params_plasma = Plasma(
        n_pe=param.npe, laser=E4Params_laser, propagation_distance=param.prop_dist
    )
    sim_E4Params = Simulation(E4Params_plasma, box_length=97 * u.micrometer, ppc=2)

    print(E4Params_beam)
    print(E4Params_laser)
    print(f"critical density for this laser is {E4Params_laser.ncrit:.1e}")
    print(E4Params_plasma)
    print(sim_E4Params)
