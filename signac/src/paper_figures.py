import proplot as pplt
import numpy as np
from scipy.constants import golden

def main():
    x, y = np.loadtxt('average_centroids.txt', unpack=True)

    fig, ax = pplt.subplots(aspect=(golden * 3, 3))
    ax.plot(x, y, "C1o:", mec="1.0",)

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ ($\mathrm{cm^{-3}}$)")
    ax.set_ylabel(r"$\langle x \rangle$ ($\mathrm{\mu m}$)")

    ax.grid(which="both")

    # TODO: remove extension
    fig.savefig("average_centroids_cut_paper.png", transparent=False)


if __name__ == "__main__":
    main()

