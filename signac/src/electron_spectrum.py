"""
Module for analysis and visualization of electron spectra.
All energies are expressed in MeV, and charges in pC.
"""
import pathlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import ClassVar, Tuple, List
import collections.abc


import numpy as np
from cycler import cycler
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.constants import c
from scipy.ndimage import gaussian_filter1d

from simulation_diagnostics import particle_energy_histogram
import mpl_util
import util, job_util

mpl_util.mpl_publication_style()


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


def get_time_series_and_iteration_time_from(job, iteration=None):
    time_series = job_util.get_time_series_from(job)
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
        total_iterations=job.sp.N_step - 1,
    )
    return fn_hist


def construct_electron_spectrum(job, iteration=None):
    fn_hist = save_energy_histogram(job, iteration)
    fig_fname = fn_hist.with_suffix(".png")

    return ElectronSpectrum(fn_hist, fig_fname)


def multiple_jobs_single_iteration(jobs, iteration=None, label=None):
    spectra = list()
    for job in sorted(
        jobs, key=lambda job: job.sp[label] if label is not None else job.id
    ):
        spectrum = construct_electron_spectrum(job, iteration)
        if label is not None:
            spectrum.label = job.sp[label]
        spectra.append(spectrum)

    return MultipleJobsMultipleSpectra(spectra=spectra, label=label)


def multiple_iterations_single_job(job, iterations):
    spectra = list()
    for iteration in sorted(iterations):
        spectrum = construct_electron_spectrum(job, iteration)
        spectrum.label = f"iteration = {iteration}"
        spectra.append(spectrum)

    return SingleJobMultipleSpectra(spectra=spectra)


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
class ElectronSpectrum(collections.abc.Hashable):
    """Keeps track of the spectrum."""

    fname: str
    fig_fname: str
    iteration: int = field(init=False)
    iteration_time_ps: float = field(init=False)
    z_position: float = field(init=False)
    total_iterations: int = field(init=False)
    c_um_per_ps: ClassVar[float] = c * 1.0e-6
    jobid: str = field(init=False)
    differential_charge: np.ndarray = field(init=False, repr=False)
    smooth_differential_charge: np.ndarray = field(init=False, repr=False)
    energy: np.ndarray = field(init=False, repr=False)
    fig: Figure = field(init=False, repr=False)
    ax: Axes = field(init=False, repr=False)
    title: str = field(init=False, repr=False)
    label: str = field(init=False)
    xlabel: str = r"$E\, (\mathrm{MeV})$"
    xlim: Tuple[float] = (50.0, 350.0)
    hatch_window: EnergyWindow = EnergyWindow(100.0, 300.0)
    sigma: int = 16  # std of Gaussian Kernel
    ylabel: str = r"$\frac{\mathrm{d} Q}{\mathrm{d} E}\, \left(\frac{\mathrm{pC}}{\mathrm{MeV}}\right)$"
    ylim: Tuple[float] = (0.0, 50.0)
    linewidth: float = 0.5
    linecolor: str = "0.5"
    linestyle: str = "solid"
    alpha: float = 0.75

    def __post_init__(self):
        (
            self.differential_charge,
            energy,
            self.iteration,
            self.iteration_time_ps,
            self.jobid,
            self.total_iterations,
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
        self.z_position = self.iteration_time_ps * self.c_um_per_ps
        self.title = self.generate_title()

    def __hash__(self):
        return hash((self.iteration, self.total_iterations, self.jobid))

    def loadf(self):
        f = np.load(self.fname)
        return (
            f["counts"],
            f["edges"],
            f["iteration"].item(),
            f["iteration_time_ps"].item(),
            np.array_str(f["jobid"]),
            f["total_iterations"].item(),
        )

    def generate_title(self):
        percentage = self.iteration / self.total_iterations
        title = (
            f"t = {self.iteration_time_ps:.2f} ps, z = {self.z_position:.0f} $\mathrm{{\mu m}}$"
            f"(iteration {self.iteration}, {percentage:.0%})"
        )
        return title

    def prepare_figure(self, figsize=(10, 3.5)):
        self.fig, self.ax = pyplot.subplots(figsize=figsize, facecolor="white")

        self.ax.set_title(self.title, fontsize=10)

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)

        self.ax.set_xlim(*self.xlim)
        self.ax.set_ylim(*self.ylim)

    def add_histogram(
        self, ax=None, linecolor=None, linestyle=None, linewidth=None, label=None
    ):
        if ax is None:
            ax = self.ax
        if linecolor is None:
            linecolor = self.linecolor
        if linestyle is None:
            linestyle = self.linestyle
        if linewidth is None:
            linewidth = self.linewidth

        ax.hist(
            x=self.energy,
            bins=self.energy,
            weights=self.differential_charge,
            histtype="step",
            color=linecolor,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )

    def gaussian_filter(self):
        combined_cycler = cycler(color=["C1", "C2", "C3"]) + cycler(
            linestyle=["dashed", "dotted", "dashdot"]
        )
        combined_cycler_iterator = combined_cycler()
        cycler_dict = defaultdict(lambda: next(combined_cycler_iterator))

        def add_gaussian_filter(sigma=None):
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
                color=cycler_dict[label]["color"],
                linestyle=cycler_dict[label]["linestyle"],
                linewidth=2 * self.linewidth,
            )

        return add_gaussian_filter

    def add_ticks(self, major_x_every=25.0, major_y_every=10.0):
        mpl_util.add_ticks(
            self.ax,
            major_x_every=major_x_every,
            major_y_every=major_y_every,
            alpha=self.alpha,
        )

    def add_grid(self):
        mpl_util.add_grid(self.ax, linewidth=self.linewidth, linecolor=self.linecolor)

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

    def add_legend(self):
        self.ax.legend(
            bbox_to_anchor=(0, 1, 1, 0.1),
            ncol=2,
            mode="expand",
            loc="lower left",
            frameon=False,
            handlelength=1,
        )

    def plot(self):
        self.prepare_figure()
        self.add_histogram()
        self.gaussian_filter()()
        self.add_ticks()
        self.add_grid()
        self.add_hatch()
        self.annotate_peak()
        self.add_job_id()
        self.add_legend()

    def savefig(self, fname=None, dpi=192):
        if fname is None:
            fname = self.fig_fname
        self.fig.savefig(fname, dpi=dpi)
        pyplot.close(self.fig)


@dataclass
class MultipleSpectra(collections.abc.Sequence):
    """Base class for list of ElectronSpectrum objects."""

    spectra: List[ElectronSpectrum]
    fig_fname: str = field(init=False)

    energy: np.ndarray = field(init=False, repr=False)

    fig: Figure = field(init=False, repr=False)
    ax: Axes = field(init=False, repr=False)

    xlabel: str = r"$E\, (\mathrm{MeV})$"
    xlim: Tuple[float] = (50.0, 350.0)
    ylabel: str = r"$\frac{\mathrm{d} Q}{\mathrm{d} E}\, \left(\frac{\mathrm{pC}}{\mathrm{MeV}}\right)$"
    ylim: Tuple[float] = (0.0, 50.0)

    linewidth: float = 0.5
    linecolor: str = "0.5"
    alpha: float = 0.75

    def __post_init__(self):
        assert util.all_equal(
            (spectrum.energy for spectrum in self)
        ), "Spectra have different energy ranges."
        self.energy = self[0].energy

    def __getitem__(self, key):
        return self.spectra.__getitem__(key)

    def __len__(self):
        return self.spectra.__len__()

    def prepare_figure(self, figsize=(10, 3.5)):
        self.fig, self.ax = pyplot.subplots(figsize=figsize, facecolor="white")

        self.ax.set_xlabel(self.xlabel)
        self.ax.set_xlim(*self.xlim)

        self.ax.set_ylabel(self.ylabel)
        self.ax.set_ylim(*self.ylim)

    def add_histograms(self):
        combined_cycler = cycler(color=["C0", "C1", "C2", "C3"]) + cycler(
            linestyle=["solid", "dashed", "dotted", "dashdot"]
        )
        combined_cycler_iterator = combined_cycler()
        cycler_dict = defaultdict(lambda: next(combined_cycler_iterator))

        for spectrum in self:
            label = spectrum.label
            spectrum.add_histogram(
                self.ax,
                linecolor=cycler_dict[label]["color"],
                linestyle=cycler_dict[label]["linestyle"],
                linewidth=2 * self.linewidth,
                label=label,
            )

        self.ax.legend(
            frameon=False,
            handlelength=1,
        )

    def add_grid(self):
        mpl_util.add_grid(self.ax, linewidth=self.linewidth, linecolor=self.linecolor)

    def add_ticks(self, major_x_every=25.0, major_y_every=10.0):
        mpl_util.add_ticks(
            self.ax,
            major_x_every=major_x_every,
            major_y_every=major_y_every,
            alpha=self.alpha,
        )

    def plot(self):
        self.prepare_figure()
        self.add_histograms()
        self.add_grid()
        self.add_ticks()

    def savefig(self, fname=None, dpi=192):
        if fname is None:
            fname = self.fig_fname
        self.fig.savefig(fname, dpi=dpi)
        pyplot.close(self.fig)


@dataclass
class SingleJobMultipleSpectra(MultipleSpectra):
    jobid: str = field(init=False)
    title: str = field(init=False, repr=False)

    def __post_init__(self):
        assert util.all_equal(
            (spectrum.jobid for spectrum in self)
        ), "Spectra belong to different jobs."
        self.jobid = self[0].jobid

        self.title = self.jobid
        self.fig_fname = self.create_fig_fname()

    def create_fig_fname(self):
        fig_fname = f"{self.jobid:.6}_"

        its = sorted(spectrum.iteration for spectrum in self)
        its = (str(it) for it in its)
        fig_fname += "_".join(its) + ".png"
        return fig_fname

    def prepare_figure(self, figsize=(10, 3.5)):
        super().prepare_figure(figsize=figsize)
        self.ax.set_title(self.title, fontsize=10)


@dataclass
class MultipleJobsMultipleSpectra(MultipleSpectra):
    iteration: int = field(init=False)
    title: str = field(init=False, repr=False)
    label: str = field(default_factory=str)

    def __post_init__(self):
        assert util.all_equal(
            (spectrum.iteration for spectrum in self)
        ), "Spectra have different iteration numbers."
        self.iteration = self[0].iteration
        self.title = f"iteration {self.iteration}"
        self.create_labels()
        self.fig_fname = self.create_fig_fname()

    def create_labels(self):
        for spectrum in self:
            if not self.label:
                spectrum.label = spectrum.jobid
            else:
                spectrum.label = (
                    f"{self.label} = {spectrum.label} — {spectrum.jobid:.8}"
                )

    def create_fig_fname(self):
        ids = sorted(f"{spectrum.jobid:.6}" for spectrum in self)
        fig_fname = "_".join(ids) + ".png"
        return fig_fname

    def prepare_figure(self, figsize=(10, 3.5)):
        super().prepare_figure(figsize=figsize)
        self.ax.set_title(self.title, fontsize=10)


def main():
    """Main entry point."""
    import random
    import signac

    random.seed(24)

    proj = signac.get_project(search=False)

    job = random.choice(list(iter(proj)))
    # es = construct_electron_spectrum(job)
    # es.plot()
    # es.savefig()
    # print(f"Read {es.fname}")
    # print(f"Wrote {es.fig_fname}")

    spectra = multiple_jobs_single_iteration(proj.find_jobs(), label="Nm")
    spectra.plot()
    spectra.savefig()

    time_series = job_util.get_time_series_from(job)
    print("Available iterations:")
    print(time_series.iterations)

    per_job_spectra = multiple_iterations_single_job(job, (52844, 79266, 105688))
    per_job_spectra.plot()
    per_job_spectra.savefig()


if __name__ == "__main__":
    main()
