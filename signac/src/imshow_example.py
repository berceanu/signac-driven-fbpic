import numpy as np
from matplotlib import pyplot, colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# TODO eventually delete file
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
    img = ax.imshow(
        data,
        origin="lower",
        extent=(left, right, bottom, top),
        norm=colors.Normalize(vmin=data.min(), vmax=data.max()),
        cmap=cm.get_cmap("cividis"),
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.02)
    cbar = ax.figure.colorbar(
        img,
        cax=cax,
    )
    cbar.set_label(r"E ($\mathrm{MeV}$)")

    ax.set_xlabel(r"$n_e$ ($10^{18}\,\mathrm{electrons\,cm^{-3}}$)")
    ax.set_ylabel(r"$a_0$")

    fig.savefig("imshow.png")
    pyplot.close(fig)


if __name__ == "__main__":
    main()
