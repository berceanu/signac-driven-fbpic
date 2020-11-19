import numpy as np
from matplotlib import pyplot


def func1(x1, gasPower):
    n1 = ne * np.exp(-(((x1 - gasCenterLeft_SI) / (gasSigmaLeft_SI)) ** gasPower))
    return n1


def func2(x2, gasPower):
    n2 = ne * np.exp(-(((x2 - gasCenterRight_SI) / (gasSigmaRight_SI)) ** gasPower))
    return n2


def func3(x3, gasPower):
    n3 = ne * np.exp(-(((x3 - x3) / (gasSigmaRight_SI)) ** gasPower))
    return n3


if __name__ == "__main__":
    flat_top_dist = 1.0  # plasma flat top distance (mm)
    ne = 5.307e18  # electron plasma density

    gasCenterLeft_SI = 1000
    gasCenterRight_SI = gasCenterLeft_SI + flat_top_dist * 1000
    gasSigmaLeft_SI = 500
    gasSigmaRight_SI = 500
    FOCUS_POS_SI = 500

    x1 = np.arange(0, gasCenterLeft_SI, 1)
    x2 = np.arange(gasCenterRight_SI, (gasCenterRight_SI + 2 * gasSigmaRight_SI), 1)
    x3 = np.arange(gasCenterLeft_SI, gasCenterRight_SI, 1)

    Nozzle_r = (gasCenterLeft_SI + gasCenterRight_SI) / 2 - gasSigmaLeft_SI
    Nozzle_r = Nozzle_r * 0.001
    Nozzle_r = round(Nozzle_r, 2)

    print("Nozzle radius =", Nozzle_r, "mm")

    fig, ax = pyplot.subplots(figsize=(30, 4.8))

    ax.plot(x1, func1(x1, 4), color="black")
    ax.plot(x2, func2(x2, 4), color="black")
    ax.plot(x3, func3(x3, 4), color="black")

    ax.axvline(x=gasCenterLeft_SI, ymin=0, ymax=ne, linestyle="--")
    ax.axvline(x=gasCenterRight_SI, ymin=0, ymax=ne, linestyle="--")
    ax.axvline(x=FOCUS_POS_SI, ymin=0, ymax=ne, linestyle="--", color="red")

    ax.set_ylabel(r"Electron density (cm$^{-3}$)")
    ax.set_xlabel(r"Plasma length ($\mathrm{\mu m}$)")

    ax.annotate(
        rf"Nozzle radius = {Nozzle_r} mm",
        xy=(gasSigmaLeft_SI, ne / 3),
        xycoords="data",
        xytext=((gasCenterLeft_SI + gasCenterRight_SI) / 2, ne / 3),
        textcoords="data",
        arrowprops=dict(arrowstyle="<->", connectionstyle="arc3"),
    )

    ax.set_ylim(ymin=0)
    ax.set_xlim(xmin=-500)

    fig.savefig("ong.png")
