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


def plot_average_centroids(x, y, ax, x_scale_is_log=True, save_as_png=True):
    ax.set_xlabel(r"$n_e$ ($\mathrm{cm^{-3}}$)")
    ax.set_ylabel(r"$\langle x \rangle$ ($\mathrm{\mu m}$)")

    ax.plot(
        x,
        y,
        "C1o:",
        mec="1.0",
    )

    ax.grid(which="both")
    ax.tick_params(axis="x", which="minor", labelsize=5)

    if x_scale_is_log:
        log_or_lin = "log"
        ax.set_xscale("log")

        subs = [2.0, 4.0, 6.0]  # ticks to show per decade
        ax.xaxis.set_minor_locator(
            ticker.LogLocator(subs=subs)
        )  # set the ticks position
        ax.xaxis.set_minor_formatter(
            ticker.FuncFormatter(ticks_format)
        )  # add the custom ticks
        ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
    else:  # linear scale on x axis
        log_or_lin = "linear"
        formatter = ticker.ScalarFormatter()
        formatter.set_useMathText(True)
        formatter.set_useOffset(False)
        ax.xaxis.set_minor_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)

    if save_as_png:
        ax.figure.savefig(
            f"average_centroids_cut_{log_or_lin}_paper.png", transparent=False
        )
    else:  # save as PDF
        ax.figure.savefig(f"average_centroids_cut_{log_or_lin}_paper")


def main():
    x, y = np.loadtxt("average_centroids.txt", unpack=True)

    fig, ax = pplt.subplots(aspect=(golden * 3, 3))
    plot_average_centroids(x, y, ax, x_scale_is_log=True, save_as_png=False)


if __name__ == "__main__":
    main()
