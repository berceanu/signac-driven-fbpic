import numpy as np
from matplotlib import pyplot


def main():
    left = -0.5
    bottom = -0.5
    right = 15.5
    top = 15.5

    y, x = np.ogrid[
        int(bottom + 0.5) : int(top + 0.5), int(left + 0.5) : int(right + 0.5)
    ]
    data = x + y


    fig, ax = pyplot.subplots()
    ax.imshow(data, origin="lower", extent=(left, right, bottom, top))

    fig.savefig("imshow.png")


if __name__ == "__main__":
    main()
