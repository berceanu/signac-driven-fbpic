import numpy as np
from matplotlib import pyplot


def main():
    x, y = np.ogrid[0:6, 0:7]
    data = x + y

    print(data)
    print(data.shape)
    # [[ 0  1  2  3  4  5  6]
    # [ 1  2  3  4  5  6  7]
    # [ 2  3  4  5  6  7  8]
    # [ 3  4  5  6  7  8  9]
    # [ 4  5  6  7  8  9 10]
    # [ 5  6  7  8  9 10 11]]
    # (6, 7)

    fig, ax = pyplot.subplots()
    ax.imshow(data, origin="lower", extent=(-0.5, 6.5, -0.5, 5.5))
    # extent=(left, right, bottom, top)

    fig.savefig("imshow.png")


if __name__ == "__main__":
    main()
