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
import pathlib
from openpmd_viewer.addons import LpaDiagnostics
from simulation_diagnostics import particle_energy_histogram

C_LS = cycler(color=["C1", "C2", "C3"]) + cycler(linestyle=["--", ":", "-."])
C_LS_ITER = C_LS()
STYLE = defaultdict(lambda: next(C_LS_ITER))


def get_time_series_from(job):
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path)
    return time_series


def get_iteration_time_from(time_series, iteration=None):
    if iteration is None:  # use final iteration
        index = -1
    else:
        index = np.where(time_series.iterations == iteration)

    final_iteration = time_series.iterations[-1]

    iteration_time_in_s = time_series.t[index]

    try:
        time_in_s = iteration_time_in_s.item()
    except ValueError:
        print(
            f"Iteration {iteration:d} not found, available iterations: {time_series.iterations.tolist()}"
        )
        raise

    return time_in_s, final_iteration
    # TODO add z position, see project.py L460

def get_time_series_and_iteration_time_from(job, iteration=None):
    time_series = get_time_series_from(job)
    iteration_time_in_s, final_iteration = get_iteration_time_from(
        time_series, iteration
    )

    if iteration is None:
        iteration = final_iteration

    iteration_time_ps = iteration_time_in_s * 1.0e12
    return time_series, iteration, iteration_time_ps


def save_energy_histogram(job, iteration=None):
    if iteration is None:
        fn_hist = pathlib.Path(job.fn("final_histogram.npz"))
    else:
        fn_hist = pathlib.Path(job.fn(f"histogram{iteration:06d}.npz"))

    time_series, iteration, iteration_time_ps = get_time_series_and_iteration_time_from(
        job, iteration
    )

    # no cutoff
    hist, bins, nbins = particle_energy_histogram(
        tseries=time_series,
        iteration=iteration,
        species="electrons",
        cutoff=np.inf,
    )
    np.savez(
        fn_hist,
        counts=hist,
        edges=bins,
        iteration=iteration,
        iteration_time_ps=iteration_time_ps,
        jobid=job.id,
    )
    return fn_hist


def construct_electron_spectrum(job, iteration=None):
    fn_hist = save_energy_histogram(job, iteration)
    fig_fname = fn_hist.with_suffix(".png")

    return ElectronSpectrum(fn_hist, fig_fname)


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
    fig_fname: str
    iteration: int = field(init=False)
    iteration_time_ps: float = field(init=False)
    jobid: str = field(init=False)
    differential_charge: np.ndarray = field(init=False, repr=False)
    smooth_differential_charge: np.ndarray = field(init=False, repr=False)
    energy: np.ndarray = field(init=False, repr=False)
    fig: Figure = field(init=False, repr=False)
    ax: Axes = field(init=False, repr=False)
    xlabel: str = r"$E\, (\mathrm{MeV})$"
    xlim: Tuple[float] = (50.0, 350.0)
    hatch_window: EnergyWindow = EnergyWindow(100.0, 300.0)
    sigma: float = 11.0  # std of Gaussian Kernel
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
            self.jobid,
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
        return (
            f["counts"],
            f["edges"],
            f["iteration"],
            f["iteration_time_ps"],
            np.array_str(f["jobid"]),
        )

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

    def add_job_id(self):
        self.ax.annotate(
            text=f"{self.jobid}",
            xycoords="axes fraction",
            xy=(0.9, 0.9),
            color=self.linecolor,
            xytext=(10, 0),
            textcoords="offset points",
            size="small",
            ha="right",
            va="baseline",
        )

    def plot(self):
        self.prepare_figure()
        self.add_histogram()
        self.add_gaussian_filter()
        self.add_ticks()
        self.add_grid()
        self.add_hatch()
        self.annotate_peak()
        self.add_job_id()
        self.ax.legend(
            bbox_to_anchor=(0, 1, 1, 0.1),
            ncol=2,
            mode="expand",
            loc="lower left",
            frameon=False,
        )
        self.ax.set_title(self.title)

    def savefig(self, fname=None, dpi=192):
        if fname is None:
            fname = self.fig_fname
        self.fig.savefig(fname, dpi=dpi)
        pyplot.close(self.fig)


def main():
    """Main entry point."""
    import random
    import signac

    random.seed(24)

    proj = signac.get_project(search=False)
    job = random.choice(list(iter(proj)))

    es = construct_electron_spectrum(job)
    es.plot()
    es.savefig()

    print(f"Read {es.fname}")
    print(f"Wrote {es.fig_fname}")

    print(f"Peak position at {es.hatch_window.peak_position:.1f} MeV")
    print(f"Total integrated charge {es.hatch_window.total_charge:.1f}  pC")


if __name__ == "__main__":
    main()
