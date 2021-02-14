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
from typing import Tuple

C_LS = cycler(color=["C1", "C2", "C3"]) + cycler(linestyle=["--", ":", "-."])
C_LS_ITER = C_LS()
STYLE = defaultdict(lambda: next(C_LS_ITER))


@dataclass
class EnergyWindow:
    low: float
    high: float
    peak_position: float = field(init=False)
    total_charge: float = field(init=False)
    mask: np.ndarray = field(init=False, repr=False)

    def create_boolean_mask(self, energy):
        return (energy >= self.low) & (energy <= self.high)

    def find_peak_position(self, energy, charge):
        return energy[self.mask][np.argmax(charge[self.mask])]

    def integrate_charge(self, energy, charge):
        delta_energy = np.diff(energy)
        return np.sum(delta_energy[self.mask] * charge[self.mask])


@dataclass
class ElectronSpectrum:
    """Keeps track of the spectrum."""

    fname: str
    iteration: int = field(init=False)
    iteration_time_ps: float = field(init=False)
    differential_charge: np.ndarray = field(init=False, repr=False)
    smooth_differential_charge: np.ndarray = field(init=False, repr=False)
    energy: np.ndarray = field(init=False, repr=False)
    fig: Figure = field(init=False, repr=False)
    ax: Axes = field(init=False, repr=False)
    xlabel: str = r"$E\, (\mathrm{MeV})$"
    xlim: Tuple[float] = (50.0, 350.0)
    hatch_window: EnergyWindow = EnergyWindow(100.0, 300.0)
    sigma: float = 10.0  # std of Gaussian Kernel
    ylabel: str = r"$\frac{\mathrm{d} Q}{\mathrm{d} E}\, \left(\frac{\mathrm{pC}}{\mathrm{MeV}}\right)$"
    ylim: Tuple[float] = (0.0, 50.0)
    linewidth: float = 0.5
    linecolor: str = "0.5"
    alpha: float = 0.75
    title: str = field(init=False, repr=False)

    def __post_init__(self):
        (
            self.differential_charge,
            energy,
            self.iteration,
            self.iteration_time_ps,
        ) = self.loadf()
        self.hatch_window.mask = self.hatch_window.create_boolean_mask(energy[:-1])
        self.hatch_window.total_charge = self.hatch_window.integrate_charge(
            energy, self.differential_charge
        )
        self.energy = energy[:-1]
        self.smooth_differential_charge = gaussian_filter1d(
            self.differential_charge, self.sigma
        )
        self.hatch_window.peak_position = self.hatch_window.find_peak_position(
            self.energy, self.smooth_differential_charge
        )
        self.title = f"t = {self.iteration_time_ps:.2f} ps (iteration {self.iteration})"

    def loadf(self):
        f = np.load(self.fname)
        return f["counts"], f["edges"], f["iteration"], f["iteration_time_ps"]

    def prepare_figure(self, figsize=(10, 3.5)):
        self.fig, self.ax = pyplot.subplots(figsize=figsize, facecolor="white")

        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

    def add_histogram(self):
        self.ax.hist(
            x=self.energy,
            bins=self.energy,
            weights=self.differential_charge,
            histtype="step",
            color=self.linecolor,
            linewidth=self.linewidth,
        )

    def add_gaussian_filter(self, sigma=None):
        if sigma is None:
            y = self.smooth_differential_charge
            sigma = self.sigma
        else:
            y = gaussian_filter1d(self.differential_charge, sigma)

        label = f"Gaussian filter, $\\sigma={sigma:.0f}$"
        self.ax.plot(
            self.energy,
            y,
            label=label,
            color=STYLE[label]["color"],
            linestyle=STYLE[label]["linestyle"],
            linewidth=2 * self.linewidth,
        )

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
            x=self.energy,
            y1=self.differential_charge,
            where=(self.energy >= self.hatch_window.low)
            & (self.energy <= self.hatch_window.high),
            facecolor="none",
            hatch="///",
            edgecolor=self.linecolor,
            linewidth=0.0,
            alpha=self.alpha,
        )

    def annotate_peak(self):
        self.ax.axvline(
            x=self.hatch_window.peak_position,
            color=self.linecolor,
            linestyle="solid",
            linewidth=2 * self.linewidth,
            label=f"{self.hatch_window.peak_position:.0f} MeV, {self.hatch_window.total_charge:.0f} pC",
        )

    def plot(self):
        self.prepare_figure()
        self.add_histogram()
        self.add_gaussian_filter()
        self.add_ticks()
        self.add_grid()
        self.add_hatch()
        self.annotate_peak()
        self.ax.legend(
            bbox_to_anchor=(0, 1, 1, 0.1),
            ncol=2,
            mode="expand",
            loc="lower left",
            frameon=False,
        )
        self.ax.set_title(self.title)

    def savefig(self, fname="electron_spectrum.png", dpi=192):
        self.fig.savefig(fname, dpi=dpi)
        pyplot.close(self.fig)


def main():
    """Main entry point."""
    import random
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    job = random.choice(list(iter(proj)))

    es = ElectronSpectrum("final_histogram.npz")
    es.plot()
    es.savefig()


if __name__ == "__main__":
    main()
