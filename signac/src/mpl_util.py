"""Module containing useful matplotlib-related functionality."""
import matplotlib
from matplotlib import pyplot, ticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable


class LabelOffset:
    def __init__(self, ax, label="", axis="y"):
        self.axis = {"y": ax.yaxis, "x": ax.xaxis}[axis]
        self.label = label
        ax.callbacks.connect(axis + "lim_changed", self.update)
        ax.figure.canvas.draw()
        self.update(None)

    def update(self, lim):
        fmt = self.axis.get_major_formatter()
        self.axis.offsetText.set_visible(False)
        self.axis.set_label_text(self.label + " " + fmt.get_offset())


def mpl_publication_style(extension="pdf"):
    """
    https://turnermoni.ca/python3.html

    Usage:
    >>> from matplotlib import rc_context
    >>> with rc_context():
    >>>     mpl_publication_style()
    >>>     <plot your figure>
    """
    matplotlib.rcParams["savefig.dpi"] = 1200  # default 1200
    matplotlib.rcParams["savefig.format"] = extension  # default pdf

    # Instead of individually increasing font sizes, point sizes, and line
    # thicknesses, I found it easier to just decrease the figure size so
    # that the line weights of various components still agree
    matplotlib.rcParams["figure.figsize"] = (3.404, 2.104)
    matplotlib.rcParams["figure.dpi"] = 192

    # Turn on minor ticks, top and right axis ticks, and change the direction to "in"
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True
    matplotlib.rcParams["xtick.minor.visible"] = True
    matplotlib.rcParams["ytick.minor.visible"] = True
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.direction"] = "in"

    # Increase the major and minor tick-mark lengths
    matplotlib.rcParams["xtick.major.size"] = 6  # default 3.5
    matplotlib.rcParams["ytick.major.size"] = 6  # default 3.5
    matplotlib.rcParams["xtick.minor.size"] = 3  # default 2
    matplotlib.rcParams["ytick.minor.size"] = 3  # default 2

    # Change the tick-mark and axes widths, as well as the widths of plotted lines,
    # to be consistent with the font weight
    matplotlib.rcParams["xtick.major.width"] = 0.6  # default 0.8
    matplotlib.rcParams["ytick.major.width"] = 0.6  # default 0.8
    matplotlib.rcParams["xtick.minor.width"] = 1  # default 0.6
    matplotlib.rcParams["ytick.minor.width"] = 1  # default 0.6

    matplotlib.rcParams["axes.linewidth"] = 0.6  # default 0.8
    matplotlib.rcParams["lines.linewidth"] = 1.0  # default 1.5
    matplotlib.rcParams["lines.markeredgewidth"] = 0.6  # default 1
    matplotlib.rcParams["lines.markersize"] = 3

    # The magic sauce
    matplotlib.rcParams["text.usetex"] = True  # default True
    matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
    matplotlib.rcParams[
        "pgf.preamble"
    ] = r"\usepackage[utf8x]{inputenc} \usepackage[T1]{fontenc} \usepackage{cmbright} \usepackage{amsmath}"

    # Increase the padding between the ticklabels and the axes, to prevent
    # overlap in the lower left-hand corner
    matplotlib.rcParams["xtick.major.pad"] = 4  # default 3.5
    matplotlib.rcParams["ytick.major.pad"] = 4  # default 3.5
    matplotlib.rcParams["xtick.minor.pad"] = 4  # default 3.5
    matplotlib.rcParams["ytick.minor.pad"] = 4  # default 3.5

    # Turn off the legend frame and reduce the space between the point and the label
    matplotlib.rcParams["legend.frameon"] = False

    # Font size
    matplotlib.rcParams["legend.fontsize"] = 6
    matplotlib.rcParams["xtick.labelsize"] = 8
    matplotlib.rcParams["ytick.labelsize"] = 8
    matplotlib.rcParams["axes.labelsize"] = 8
    matplotlib.rcParams["font.size"] = 8
    matplotlib.rcParams["figure.titlesize"] = 6
    matplotlib.rcParams["axes.titlesize"] = 6

    # Margins / spacing
    matplotlib.rcParams["figure.subplot.bottom"] = 0.15
    matplotlib.rcParams["figure.subplot.left"] = 0.14


def add_colorbar(ax, mappable, *, size="5%", position="right"):
    orientation = "horizontal" if (position == "top") else "vertical"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad="1%")
    cbar = ax.figure.colorbar(
        mappable,
        cax=cax,
        orientation=orientation,
    )
    cbar.outline.set_visible(False)
    # restore default tick length and width
    for ticks, length, width in zip(("major", "minor"), (3.5, 2), (0.8, 0.6)):
        cbar.ax.tick_params(
            which=ticks,
            length=length,
            width=width,
        )
    return cbar


def main():
    """Main entry point."""


if __name__ == "__main__":
    main()
