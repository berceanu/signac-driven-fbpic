#!/usr/bin/env python3
"""This module contains the operation functions for this project.

The workflow defined in this file can be executed from the command
line with

    $ python src/project.py run [job_id [job_id ...]]

See also: $ python src/project.py --help

Note: All the lines marked with the CHANGEME comment contain customizable parameters.
"""
import logging
import math
import sys
import pathlib
from multiprocessing import Pool
from functools import partial

import numpy as np
import sliceplots
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from matplotlib import pyplot
from openpmd_viewer.addons import LpaDiagnostics
import unyt as u
from peak_detection import (
    plot_electron_energy_spectrum,
    integrated_charge,
    peak_position,
)
from util import ffmpeg_command, shell_run
from simulation_diagnostics import (
    particle_energy_histogram,
    laser_density_plot,
    phase_space_plot,
)
from density_functions import plot_density_profile, make_gaussian_dens_func
from laser_profiles import make_flat_laser_profile, plot_laser_intensity
from render_lwfa_script import write_lwfa_script
from timer import Timer

logger = logging.getLogger(__name__)
log_file_name = "fbpic-project.log"


class OdinEnvironment(DefaultSlurmEnvironment):
    """Environment profile for the LGED cluster.
    https://docs.signac.io/projects/flow/en/latest/supported_environments/comet.html#flow.environments.xsede.CometEnvironment
    """

    hostname_pattern = r".*\.ra5\.eli-np\.ro$"
    template = "odin.sh"
    cores_per_node = 16
    mpi_cmd = "mpiexec"

    @classmethod
    def add_args(cls, parser):
        super(OdinEnvironment, cls).add_args(parser)
        parser.add_argument(
            "--partition",
            choices=["cpu", "gpu"],
            default="gpu",
            help="Specify the partition to submit to.",
        )
        parser.add_argument(
            "-w",
            "--walltime",
            type=float,
            default=72,
            help="The wallclock time in hours.",
        )
        parser.add_argument(
            "--job-output",
            help=(
                "What to name the job output file. "
                "If omitted, uses the system default "
                '(slurm default is "slurm-%%j.out").'
            ),
        )


class Project(FlowProject):
    """
    Placeholder for ``FlowProject`` class.
    """


ex = Project.make_group(name="ex")


@Project.label
def progress(job):
    """Show progress of fbpic simulation, based on completed/total .h5 files."""
    # get last iteration based on input parameters
    number_of_iterations = math.ceil((job.sp.N_step - 0) / job.sp.diag_period)

    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    if not h5_path.is_dir():
        # {job_dir}/diags/hdf5 not present, ``fbpic`` didn't run
        return "0/%s" % number_of_iterations

    h5_files = list(h5_path.glob("*.h5"))

    return f"{len(h5_files)}/{number_of_iterations}"


def fbpic_ran(job):
    """
    Check if ``fbpic`` produced all the output .h5 files.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: True if all output files are in {job_dir}/diags/hdf5, False otherwise
    """
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    if not h5_path.is_dir():
        # {job_dir}/diags/hdf5 not present, ``fbpic`` didn't run
        did_it_run = False
        return did_it_run

    time_series = LpaDiagnostics(h5_path)
    iterations: np.ndarray = time_series.iterations

    # estimate iteration array based on input parameters
    estimated_iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=np.int)

    # check if iterations array corresponds to input params
    did_it_run = np.array_equal(estimated_iterations, iterations)

    return did_it_run


def are_pngs(job, stem):
    """
    Check if all the {job_dir}/`stem`s/`stem`{it:06d}.png files are present.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: True if .png files are there, False otherwise
    """
    p = pathlib.Path(job.ws) / f"{stem}s"
    files = [fn.name for fn in p.glob("*.png")]

    # estimate iteration array based on input parameters
    iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=np.int)

    pngs = (f"{stem}{it:06d}.png" for it in iterations)

    return set(files) == set(pngs)


def are_rho_pngs(job):
    return are_pngs(job, "rho")


def are_phasespace_pngs(job):
    return are_pngs(job, "phasespace")


@ex
@Project.operation
@Project.post.isfile("lwfa_script.py")
def lwfa_script(job):
    """Write lwfa_script.py in the job's workspace folder."""
    write_lwfa_script(job)


@ex
@Project.operation
@Project.post.isfile("initial_density_profile.png")
def plot_initial_density_profile(job):
    """Plot the initial plasma density profile."""
    plot_density_profile(
        make_gaussian_dens_func, job.fn("initial_density_profile.png"), job
    )


@ex
@Project.operation
@Project.pre.after(plot_initial_density_profile)
@Project.post.isfile("laser_intensity.png")
def plot_laser(job):
    """Plot the laser intensity at focus and far from focus, in linear and log scale."""
    plot_laser_intensity(
        make_flat_laser_profile(job),
        rmax=job.sp.rmax,
        Nr=job.sp.Nr,
        zfoc=job.sp.zfoc,
        z0=job.sp.z0,
        zR=job.sp.zR,
        lambda0=job.sp.lambda0,
        w0=job.sp.w0,
        fn=job.fn("laser_intensity.png"),
    )


@ex.with_directives(directives=dict(ngpu=1))
@directives(ngpu=1)
@Project.operation
@Project.pre.after(plot_laser)
@Project.post(fbpic_ran)
@Project.post.true("runtime")
def run_fbpic(job):
    """
    This ``signac-flow`` operation runs a ``fbpic`` simulation.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    from fbpic.main import Simulation
    from fbpic.lpa_utils.laser import add_laser_pulse
    from fbpic.openpmd_diag import (
        FieldDiagnostic,
        ParticleDiagnostic,
        ParticleChargeDensityDiagnostic,
    )

    # redirect stdout to "stdout.txt"
    orig_stdout = sys.stdout
    f = open(job.fn("stdout.txt"), "w")
    sys.stdout = f

    # Initialize the simulation object
    sim = Simulation(
        Nz=job.sp.Nz,
        zmax=job.sp.zmax,
        Nr=job.sp.Nr,
        rmax=job.sp.rmax,
        Nm=job.sp.Nm,
        dt=job.sp.dt,
        zmin=job.sp.zmin,
        boundaries={
            "z": "open",
            "r": job.sp.r_boundary_conditions,
        },
        n_order=-1,
        use_cuda=True,
        verbose_level=2,
    )
    # Add the plasma electrons
    plasma_elec = sim.add_new_species(
        q=u.electron_charge.to_value("C"),
        m=u.electron_mass.to_value("kg"),
        n=job.sp.n_e,
        dens_func=make_gaussian_dens_func(job),
        p_zmin=job.sp.p_zmin,
        p_zmax=job.sp.p_zmax,
        p_rmax=job.sp.p_rmax,
        p_nz=job.sp.p_nz,
        p_nr=job.sp.p_nr,
        p_nt=job.sp.p_nt,
    )

    add_laser_pulse(
        sim=sim,
        laser_profile=make_flat_laser_profile(job),
    )
    sim.set_moving_window(v=u.clight.to_value("m/s"))

    # Add diagnostics
    write_dir = pathlib.Path(job.ws) / "diags"
    sim.diags = [
        FieldDiagnostic(
            period=job.sp.diag_period,
            fldobject=sim.fld,
            comm=sim.comm,
            write_dir=write_dir,
            fieldtypes=["rho", "E"],
        ),
        ParticleDiagnostic(
            period=job.sp.diag_period,
            species={"electrons": plasma_elec},
            comm=sim.comm,
            write_dir=write_dir,
        ),
        ParticleChargeDensityDiagnostic(
            period=job.sp.diag_period,
            sim=sim,
            species={"electrons": plasma_elec},
            write_dir=write_dir,
        ),
    ]
    # TODO add electron tracking

    # set deterministic random seed
    np.random.seed(0)

    # time the fbpic run
    t = Timer()
    t.start()

    # Run the simulation
    sim.step(job.sp.N_step, show_progress=False)

    # stop the timer
    runtime = t.stop()
    job.doc.setdefault("runtime", runtime.split(".")[0])

    # redirect stdout back and close "stdout.txt"
    sys.stdout = orig_stdout
    f.close()


@ex.with_directives(directives=dict(np=3))
@directives(np=3)
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post(are_rho_pngs)
@Project.post(are_phasespace_pngs)
def save_pngs(job):
    """
    Loop through a whole simulation and, for *each ``fbpic`` iteration*:
    * save a snapshot of the plasma density field ``rho`` to {job_dir}/rhos/rho{it:06d}.png

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    rho_path = pathlib.Path(job.ws) / "rhos"
    phasespace_path = pathlib.Path(job.ws) / "phasespaces"
    time_series = LpaDiagnostics(h5_path)

    it_laser_density_plot = partial(
        laser_density_plot,
        tseries=time_series,
        rho_field_name="rho_electrons",
        save_path=rho_path,
        n_c=job.sp.n_c,
        E0=job.sp.E0,
    )
    it_phase_space_plot = partial(
        phase_space_plot,
        tseries=time_series,
        uzmax=1.5e3,
        vmax=1.0e8,
        save_path=phasespace_path,
    )

    with Pool(3) as pool:
        pool.map(it_phase_space_plot, time_series.iterations.tolist())
        pool.map(it_laser_density_plot, time_series.iterations.tolist())


def generate_movie(job, stem):
    """
    Generate a movie from all the .png files in {job_dir}/`stem`s/
    :param job: the job instance is a handle to the data of a unique statepoint
    """
    command = ffmpeg_command(
        input_files=pathlib.Path(job.ws) / f"{stem}s" / f"{stem}*.png",
        output_file=job.fn(f"{stem}.mp4"),
    )
    shell_run(command, shell=True)


@ex
@Project.operation
@Project.pre.after(save_pngs)
@Project.post.isfile("rho.mp4")
def generate_rho_movie(job):
    generate_movie(job, stem="rho")


@ex
@Project.operation
@Project.pre.after(save_pngs)
@Project.post.isfile("phasespace.mp4")
def generate_phasespace_movie(job):
    generate_movie(job, stem="phasespace")


@ex
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post.isfile("final_histogram.npz")
def save_final_histogram(job):
    """Save the histogram corresponding to the last iteration."""

    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path)
    last_iteration = time_series.iterations[-1]

    current_time = (time_series.t[-1] * u.second).to(u.picosecond)
    ax_title = f"t = {current_time:.2f} (iteration {last_iteration:,g})"
    job.doc.setdefault("ax_title", ax_title)

    # compute 1D histogram
    energy_hist, bin_edges, _ = particle_energy_histogram(
        tseries=time_series,
        iteration=last_iteration,
        cutoff=np.inf,  # no cutoff
    )
    np.savez(job.fn("final_histogram.npz"), edges=bin_edges, counts=energy_hist)


@ex
@Project.operation
@Project.pre.after(save_final_histogram)
@Project.post.isfile("final_histogram.png")
def plot_final_histogram(job):
    """Plot the electron spectrum corresponding to the last iteration."""

    plot_electron_energy_spectrum(
        job.fn("final_histogram.npz"),
        job.fn("final_histogram.png"),
        ax_title=job.doc.ax_title,
    )


# TODO replace ax_title hack with propper code
@ex
@Project.operation
@Project.pre.after(plot_final_histogram)
@Project.post(lambda job: "ax_title" not in job.doc)
def remove_ax_title(job):
    """Remove the `ax_title` key form the job's document."""
    del job.doc["ax_title"]


@ex
@Project.operation
@Project.pre.after(save_final_histogram)
@Project.post.true("peak_charge")
@Project.post.true("peak_position")
def get_peak_charge_and_position(job):
    energy_low = 100
    energy_high = 300

    int_charge = integrated_charge(
        job.fn("final_histogram.npz"), from_energy=energy_low, to_energy=energy_high
    )
    peak_pos = peak_position(
        job.fn("final_histogram.npz"), from_energy=energy_low, to_energy=energy_high
    )

    job.doc["peak_position"] = float("{:.1f}".format(peak_pos))  # MeV
    job.doc["peak_charge"] = float("{:.1f}".format(int_charge))  # pC


@ex
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post.isfile("all_hist.txt")
@Project.post.isfile("hist_edges.txt")
def save_histograms(job):
    """
    Loop through a whole simulation and, for *each ``fbpic`` iteration*:

    compute the weighted particle energy histogram and save it to "all_hist.txt",
    and the histogram bins to "hist_edges.txt"

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path)
    number_of_iterations: int = time_series.iterations.size

    # Do a mock histogram in order to get the number of bins
    _, _, nrbins = particle_energy_histogram(
        tseries=time_series,
        iteration=0,
    )
    all_hist = np.empty(shape=(number_of_iterations, nrbins), dtype=np.float64)
    hist_edges = np.empty(shape=(nrbins + 1,), dtype=np.float64)

    # loop through all the iterations in the job's time series
    for idx, it in enumerate(time_series.iterations):
        # generate 1D energy histogram
        energy_hist, bin_edges, _ = particle_energy_histogram(
            tseries=time_series,
            iteration=it,
        )
        # build up arrays for 2D energy histogram
        all_hist[idx, :] = energy_hist
        if idx == 0:  # only save the first one
            hist_edges[:] = bin_edges

    np.savetxt(
        job.fn("all_hist.txt"),
        all_hist,
        header="One iteration per row, containing the energy histogram.",
    )
    np.savetxt(job.fn("hist_edges.txt"), hist_edges, header="Energy histogram bins.")


@ex
@Project.operation
@Project.pre.after(save_histograms)
@Project.post.isfile("hist2d.png")
def plot_2d_hist(job):
    """
    Plot the 2D histogram, composed of the 1D slices for each iteration.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    all_hist = np.loadtxt(
        job.fn("all_hist.txt"), dtype=np.float64, comments="#", ndmin=2
    )
    hist_edges = np.loadtxt(
        job.fn("hist_edges.txt"), dtype=np.float64, comments="#", ndmin=1
    )
    # compute moving window position for each iteration
    iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=np.int)
    times = iterations * job.sp.dt * u.second
    positions = times * u.clight

    z_0 = positions.to_value("micrometer")
    all_z = positions.to_value("meter")

    dens = make_gaussian_dens_func(job)(all_z, 0.0)

    # rescale for visibility, 1/5th of the histogram y axis
    v_axis_size = hist_edges[-1] - hist_edges[1]
    dens *= v_axis_size / 5
    # upshift density to start from lower limit of histogram y axis
    dens += hist_edges[1] - dens.min()

    fig = pyplot.figure(figsize=(2 * 8, 8))

    # plot 2D energy-charge histogram
    hist2d = sliceplots.Plot2D(
        fig=fig,
        arr2d=all_hist.T,  # 2D data
        h_axis=z_0,  # x-axis
        v_axis=hist_edges[1:],  # y-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"E (MeV)",
        zlabel=r"dQ/dE (pC/MeV)",
        vslice_val=z_0[-1],  # can be changed to z_0[iteration]
        extent=(z_0[0], z_0[-1], hist_edges[1], hist_edges[-1]),
    )
    hist2d.ax0.plot(all_z * 1e6, dens, linewidth=2.5, linestyle="dashed", color="0.75")
    hist2d.canvas.print_figure(job.fn("hist2d.png"))


if __name__ == "__main__":
    logging.basicConfig(
        filename=log_file_name,
        format="%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("==RUN STARTED==")

    Project().main()  # run the whole signac project workflow

    logger.info("==RUN FINISHED==")
