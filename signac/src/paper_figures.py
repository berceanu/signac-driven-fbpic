import proplot as pplt
import numpy as np
from scipy.constants import golden
from matplotlib import ticker


def ticks_format(value, index):
    """
    get the value and returns the value as:
       integer: [0,99]
       1 digit float: [0.1, 0.99]
       n*10^m: otherwise
    To have all the number of the same size they are all returned as latex strings
    """
    exp = np.floor(np.log10(value))
    base = value / 10 ** exp
    if exp == 0 or exp == 1:
        return "${0:d}$".format(int(value))
    if exp == -1:
        return "${0:.1f}$".format(value)
    else:
        return "${0:d}\\times10^{{{1:d}}}$".format(int(base), int(exp))


def main():
    x, y = np.loadtxt("average_centroids.txt", unpack=True)

    fig, ax = pplt.subplots(aspect=(golden * 3, 3))
    ax.plot(
        x,
        y,
        "C1o:",
        mec="1.0",
    )

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ ($\mathrm{cm^{-3}}$)")
    ax.set_ylabel(r"$\langle x \rangle$ ($\mathrm{\mu m}$)")

    ax.grid(which="both")

    subs = [2.0, 4.0, 6.0]  # ticks to show per decade
    ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs))  # set the ticks position
    ax.xaxis.set_major_formatter(
        ticker.LogFormatterMathtext()
    )  # remove the major ticks
    ax.xaxis.set_minor_formatter(
        ticker.FuncFormatter(ticks_format)
    )  # add the custom ticks
    ax.tick_params(axis="x", which="minor", labelsize=5)

    # TODO: remove extension
    fig.savefig("average_centroids_cut_paper.png", transparent=False)


if __name__ == "__main__":
    main()
