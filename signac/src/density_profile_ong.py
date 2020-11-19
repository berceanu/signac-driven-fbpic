import numpy as np
from matplotlib import pyplot


def dens_func(z, *, center_left, center_right, sigma_left, sigma_right, power):
    def ramp(z, *, center, sigma, power):
        return np.exp(-(((z - center) / sigma) ** power))

    # Allocate relative density
    n = np.ones_like(z)

    # before up-ramp
    n = np.where(z < 0, 0, n)

    # Make up-ramp
    n = np.where(
        z < center_left, ramp(z, center=center_left, sigma=sigma_left, power=power), n
    )

    # Make down-ramp
    n = np.where(
        (z >= center_right) & (z < center_right + 2 * sigma_right),
        ramp(z, center=center_right, sigma=sigma_right, power=power),
        n,
    )

    # after down-ramp
    n = np.where(z >= center_right + 2 * sigma_right, 0, n)

    return n


if __name__ == "__main__":
    ne = 5.307e18  # electron plasma density cm$^{-3}$
    gasPower = 4

    #  lengths in microns
    flat_top_dist = 1000  # plasma flat top distance
    gasCenterLeft_SI = 1000
    gasCenterRight_SI = gasCenterLeft_SI + flat_top_dist
    gasSigmaLeft_SI = 500
    gasSigmaRight_SI = 500

    Nozzle_r = (gasCenterLeft_SI + gasCenterRight_SI) / 2 - gasSigmaLeft_SI
    FOCUS_POS_SI = 500 #microns

    all_z = np.linspace(0, gasCenterRight_SI + 2 * gasSigmaRight_SI, 3001)
    rho = dens_func(
        all_z,
        center_left=gasCenterLeft_SI,
        center_right=gasCenterRight_SI,
        sigma_left=gasSigmaLeft_SI,
        sigma_right=gasSigmaRight_SI,
        power=gasPower,
    )


    fig, ax = pyplot.subplots(figsize=(20, 4.8))

    ax.plot(all_z, ne * rho, color="black")

    ax.axvline(x=gasCenterLeft_SI, ymin=0, ymax=ne, linestyle="--")
    ax.axvline(x=gasCenterRight_SI, ymin=0, ymax=ne, linestyle="--")

    ax.axvline(x=FOCUS_POS_SI, ymin=0, ymax=ne, linestyle="--", color="red")

    ax.set_ylabel(r"Electron density (cm$^{-3}$)")
    ax.set_xlabel(r"Plasma length ($\mathrm{\mu m}$)")

    ax.annotate(
        r"Nozzle radius = %s $\mathrm{\mu m}$" % Nozzle_r,
        xy=(gasSigmaLeft_SI, ne / 3),
        xycoords="data",
        xytext=((gasCenterLeft_SI + gasCenterRight_SI) / 2, ne / 3),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
    )

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=-500)

    fig.savefig("density.png")
