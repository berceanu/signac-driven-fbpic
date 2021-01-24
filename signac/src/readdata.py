import pathlib
import numpy as np
from matplotlib import pyplot


def readbeam(datadir):
    data = np.loadtxt(datadir, skiprows=1)

    x = data[:, 0]
    z = data[:, 2]

    nbz = 190
    nbx = 190
    z_min = min(z)
    z_max = max(z)
    x_min = min(x)
    x_max = max(x)

    fig, ax = pyplot.subplots()

    h = ax.hist2d(z, x, bins=(nbz, nbx), range=[[z_min, z_max], [x_min, x_max]])
    counts = h[0]
    z_coords = np.linspace(z_min, z_max, nbz)
    x_coords = np.linspace(x_min, x_max, nbx)

    refval = 0

    centroid = []
    we = []
    centroid_z = []

    for i in range(0, nbz):
        if refval < max(counts[i, :]):
            refval = max(counts[i, :])

    for i in range(0, nbz):
        counts[i, :][counts[i, :] < 0.15 * refval] = 0
        if max(counts[i, :]) > 0.2 * refval and i > 20 and i < 180:
            centroid.append(np.average(x_coords, weights=counts[i, :]))
            centroid_z.append(z_coords[i])
            we.append(sum(counts[i, :]))

    ax.plot(centroid_z, centroid)
    ax.set_xlabel("z (m)")
    ax.set_ylabel("x (m)")

    fig.savefig("alessio.png", bbox_inches="tight")
    pyplot.close(fig)

    m_cent = abs(np.mean(centroid))
    m_cent_w = abs(np.average(centroid, weights=we))

    return m_cent, m_cent_w


def main():
    p = pathlib.Path.cwd() / "final_bunch_66dc81.txt"

    m_cent, m_cent_w = readbeam(p)
    print(m_cent, m_cent_w)


if __name__ == "__main__":
    main()
