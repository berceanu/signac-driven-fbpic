"""Module containing useful matplotlib-related functionality."""
import matplotlib
from matplotlib import pyplot, ticker

def mpl_publication_style():
    matplotlib.rcParams['axes.facecolor'] = 'red'

def add_grid(ax, linewidth=0.5, linecolor="0.5"):
    lw = dict(major=linewidth, minor=linewidth / 2)
    for xy in "x", "y":
        for ticks in "major", "minor":
            ax.grid(
                which=ticks,
                axis=xy,
                linewidth=lw[ticks],
                linestyle="dotted",
                color=linecolor,
            )


def add_ticks(ax, major_x_every=25.0, major_y_every=10.0, alpha=0.75):

    ax.yaxis.set_ticks_position("both")
    ax.xaxis.set_ticks_position("both")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(major_x_every))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(major_y_every))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    for ticks, length, width in zip(("major", "minor"), (6, 3), (2, 1)):
        ax.tick_params(
            which=ticks,
            direction="in",
            length=length,
            width=width,
            grid_alpha=alpha,
        )


def main():
    """Main entry point."""
    


if __name__ == '__main__':
    main()
