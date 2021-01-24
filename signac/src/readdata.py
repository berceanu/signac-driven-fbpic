import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.integrate import simps


def readbeam(datadir):
    data = np.loadtxt(datadir, skiprows=1)

    y = data[:, 1]
    x = data[:, 0]
    z = data[:, 2]
    ux = data[:, 3]
    uy = data[:, 4]
    uz = data[:, 5]

    nbz = 190
    nbx = 190
    z_min = min(z)
    z_max = max(z)
    x_min = min(x)
    x_max = max(x)

    h = plt.hist2d(z, x, bins=(nbz, nbx), range=[[z_min, z_max], [x_min, x_max]])
    counts = h[0]
    z_coords = np.linspace(z_min, z_max, nbz)
    x_coords = np.linspace(x_min, x_max, nbx)

    refval = 0

    centroid = []
    we = []
    centroid_z = []
    slope = []

    for i in range(0, nbz):
        if refval < max(counts[i, :]):
            refval = max(counts[i, :])

    for i in range(0, nbz):
        counts[i, :][counts[i, :] < 0.15 * refval] = 0
        if max(counts[i, :]) > 0.2 * refval and i > 20 and i < 180:
            # print(np.argmax(counts[i,:]))
            # centroid.append(x_coords[np.argmax(counts[i,:])])
            centroid.append(np.average(x_coords, weights=counts[i, :]))
            centroid_z.append(z_coords[i])
            we.append(sum(counts[i, :]))

    y_spl = UnivariateSpline(centroid_z, centroid, s=0, k=4)
    x_range = np.linspace(centroid_z[0], centroid_z[-1], 1000)

    plt.plot(x_range, y_spl(x_range))

    plt.plot(centroid_z, centroid)
    plt.xlabel("z (m)", fontsize=20)
    plt.ylabel("x (m)", fontsize=20)
    plt.show()

    y_spl_2d = y_spl.derivative(n=2)
    y2 = y_spl_2d(x_range) ** 2
    In = simps(y2, x_range)
    # plt.plot(x_range,y2)
    # plt.show()

    # mean_head = np.mean(centroid[len(centroid)-5:len(centroid)])
    m_cent = abs(np.mean(centroid))
    m_cent_w = abs(np.average(centroid, weights=we))
    # print(m_cent)#
    # print(m_cent_w)

    return m_cent, m_cent_w, In


def main():
    p = pathlib.Path.cwd() / "final_bunch_66dc81.txt"

    m_cent, m_cent_w, In = readbeam(p)
    print(m_cent_w)


if __name__ == "__main__":
    main()
