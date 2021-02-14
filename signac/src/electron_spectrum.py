"""
Module for analysis and visualization of electron spectra.
All energies are expressed in MeV, and charges in pC.
"""
from dataclasses import dataclass, field
import numpy as np
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.ndimage import gaussian_filter1d
from cycler import cycler
from collections import defaultdict

C_LS = cycler(color=["C1", "C2", "C3"]) + cycler(linestyle=["--", ":", "-."])
C_LS_ITER = C_LS()
STYLE = defaultdict(lambda: next(C_LS_ITER))


@dataclass
class Window:
    __slots__ = ["low", "high"]
    low: float
    high: float


@dataclass
class EnergyWindow(Window):
    pass


@dataclass
class ChargeWindow(Window):
    pass


@dataclass
class ElectronSpectrum:
    """Keeps track of the spectrum."""

    fname: str
    differential_charge: np.ndarray = field(init=False, repr=False)
    smooth_differential_charge: np.ndarray = field(init=False, repr=False)
    energy: np.ndarray = field(init=False, repr=False)
    fig: Figure = field(init=False, repr=False)
    ax: Axes = field(init=False, repr=False)
    xlabel: str = "$E$ (MeV)"
    xlim: EnergyWindow = EnergyWindow(50.0, 350.0)
    hatch_window: EnergyWindow = EnergyWindow(100.0, 300.0)
    ylabel: str = "$\\frac{\\mathrm{d} Q}{\\mathrm{d} E}$ (pC/MeV)"
    ylim: ChargeWindow = ChargeWindow(0.0, 50.0)
    linewidth: float = 0.5
    linecolor: str = "0.5"
    alpha: float = 0.75

    def __post_init__(self):
        self.differential_charge, self.energy = self.loadf()

    def loadf(self):
        f = np.load(self.fname)
        return f["counts"], f["edges"][:-1]

    def prepare_figure(self):
        self.fig, self.ax = pyplot.subplots(figsize=(10, 4), facecolor="white")

        self.ax.set_xlim(self.xlim.low, self.xlim.high)
        self.ax.set_ylim(self.ylim.low, self.ylim.high)

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def add_histogram(self):
        self.ax.hist(
            self.energy,
            self.energy,
            weights=self.differential_charge,
            histtype="step",
            color=self.linecolor,
            linewidth=self.linewidth,
            label="original electron spectrum",
        )

    def add_gaussian_filter(self, sigma=3.0):
        y = gaussian_filter1d(self.differential_charge, sigma)
        label = f"filtered, $\\sigma={sigma}$"
        self.ax.plot(
            self.energy,
            y,
            label=label,
            color=STYLE[label]["color"],
            linestyle=STYLE[label]["linestyle"],
            linewidth=2 * self.linewidth,
        )
        return y

    def add_ticks(self, major_x_every=25.0, major_y_every=10.0):

        self.ax.yaxis.set_ticks_position("both")
        self.ax.xaxis.set_ticks_position("both")

        self.ax.xaxis.set_major_locator(MultipleLocator(major_x_every))
        self.ax.yaxis.set_major_locator(MultipleLocator(major_y_every))
        self.ax.xaxis.set_minor_locator(AutoMinorLocator())
        self.ax.yaxis.set_minor_locator(AutoMinorLocator())

        for ticks, length, width in zip(("major", "minor"), (6, 3), (2, 1)):
            self.ax.tick_params(
                which=ticks,
                direction="in",
                length=length,
                width=width,
                grid_alpha=self.alpha,
            )

    def add_grid(self):
        lw = dict(major=self.linewidth, minor=self.linewidth / 2)
        for ax in "x", "y":
            for ticks in "major", "minor":
                self.ax.grid(
                    which=ticks,
                    axis=ax,
                    linewidth=lw[ticks],
                    linestyle="dotted",
                    color=self.linecolor,
                )

    def add_hatch(self):
        self.ax.fill_between(
            self.energy,
            self.differential_charge,
            where=(self.energy >= self.hatch_window.low)
            & (self.energy <= self.hatch_window.high),
            facecolor="none",
            hatch="///",
            edgecolor=self.linecolor,
            linewidth=0.0,
            alpha=self.alpha,
        )

    def plot(self):
        self.prepare_figure()
        self.add_histogram()
        for sigma in 3, 6, 10:
            self.add_gaussian_filter(sigma=sigma)
        self.smooth_differential_charge = self.add_gaussian_filter(sigma=10)
        self.add_ticks()
        self.add_grid()
        self.add_hatch()
        self.ax.legend()

    def save(self, fname="electron_spectrum.png", dpi=192):
        self.fig.savefig(fname, dpi=dpi)
        pyplot.close(self.fig)


def main():
    """Main entry point."""
    import random
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    job = random.choice(list(iter(proj)))
    print(job)

    es = ElectronSpectrum(job.fn("final_histogram.npz"))

    es.plot()
    es.save()


if __name__ == "__main__":
    main()
