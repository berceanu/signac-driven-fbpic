#!/usr/bin/env python3
"""This module contains the operation functions for this preproject.

The workflow defined in this file can be executed from the command
line with

    $ python src/preproject.py run [job_id [job_id ...]]

See also: $ python src/preproject.py --help
"""
import logging
import os
import subprocess
import sys
from matplotlib import pyplot
from typing import List, Optional, Tuple, Union, Callable, Iterable

import numpy as np
import pandas as pd
import sliceplots
from flow import FlowProject
from opmd_viewer import OpenPMDTimeSeries
from scipy.constants import physical_constants
from scipy.signal import hilbert
from signac.contrib.job import Job

logger = logging.getLogger(__name__)
log_file_name = "fbpic-preproject.log"

c_light = physical_constants["speed of light in vacuum"][0]
m_e = physical_constants["electron mass"][0]
q_e = physical_constants["elementary charge"][0]
mc2 = m_e * c_light ** 2 / (q_e * 1e6)  # 0.511 MeV

#####################
# UTILITY FUNCTIONS #
#####################


def nstep(job):
    return int((job.sp.zmax - job.sp.zmin) / 2 / c_light / job.sp.dt * 8)


def are_files(file_names: Iterable[str]) -> Callable[[Job], bool]:
    """Check if given file names are in the ``job`` dir.
    Useful for pre- and post- operation conditions.

    :param file_names: iterable containing file names
    :return: anonymous function that does the check
    """
    return lambda job: all(job.isfile(fn) for fn in file_names)


def path_exists(path: str) -> Callable[[Job], bool]:
    """
    Checks if relative ``path`` exists inside the job's workspace folder.
    Useful for pre- and post- operation conditions.

    :param path: relative path, eg. dir1/dir2
    :return: anonymous function that does the check
    """
    return lambda job: os.path.isdir(os.path.join(job.ws, path))


def sh(*cmd, **kwargs) -> str:
    """
    Run the command ``cmd`` in the shell.

    :param cmd: the command to be run, with separate arguments
    :param kwargs: optional keyword arguments for ``Popen``, eg. shell=True
    :return: the shell STDOUT and STDERR
    """
    logger.info(cmd[0])
    stdout = (
        subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs
        )
        .communicate()[0]
        .decode("utf-8")
    )
    logger.info(stdout)
    return stdout


def ffmpeg_command(
    frame_rate: float = 4.0,
    input_files: str = "pic%04d.png",  # pic0001.png, pic0002.png, ...
    output_file: str = "test.mp4",
) -> str:
    """
    Build up the command string for running ``ffmpeg``.
    For details, see `this blog post <http://hamelot.io/visualization/using-ffmpeg-to-convert-a-set-of-images-into-a-video/>`_

    :param frame_rate: desired video framerate, in fps
    :param input_files: shell-like wildcard pattern (globbing)
    :param output_file: name of video output
    :return: the command that needs to be executed in the shell for producing the video from the input files
    """
    return (
        rf"ffmpeg -framerate {frame_rate} -pattern_type glob -i '{input_files}' "
        rf"-c:v libx264 -vf fps=25 -pix_fmt yuv420p {output_file}"
    )


class Project(FlowProject):
    """
    Placeholder for ``FlowProject`` class.
    """

    pass


###############################
# INITIALIZE & RUN SIMULATION #
###############################


def fbpic_ran(job: Job) -> bool:
    """
    Check if ``fbpic`` produced all the output .h5 files.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: True if all output files are in {job_dir}/prediags/hdf5, False otherwise
    """
    h5_path: Union[bytes, str] = os.path.join(job.ws, "prediags", "hdf5")
    if not os.path.isdir(h5_path):
        # {job_dir}/prediags/hdf5 not present, ``fbpic`` didn't run
        did_it_run = False
        return did_it_run

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=True)
    iterations: np.ndarray = time_series.iterations

    # estimate iteration array based on input parameters
    estimated_iterations = np.arange(0, nstep(job), job.sp.diag_period, dtype=np.int)

    # check if iterations array corresponds to input params
    did_it_run = np.array_equal(estimated_iterations, iterations)

    return did_it_run


def are_rho_pngs(job: Job) -> bool:
    """
    Check if all the {job_dir}/prediags/rhos/rho{it:06d}.png files are present.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: True if .png files are there, False otherwise
    """
    files = os.listdir(os.path.join(job.ws, "prediags", "rhos"))

    # estimate iteration array based on input parameters
    iterations = np.arange(0, nstep(job), job.sp.diag_period, dtype=np.int)

    pngs = (f"rho{it:06d}.png" for it in iterations)

    return set(files) == set(pngs)


@Project.operation
@Project.post(fbpic_ran)
def run_fbpic(job: Job) -> None:
    """
    This ``signac-flow`` operation runs a ``fbpic`` simulation.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    from fbpic.main import Simulation
    from fbpic.lpa_utils.laser import add_laser
    from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

    # The density profile
    def dens_func(z: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Returns relative density at position z and r.

        :param z: longitudinal positions, 1d array
        :param r: radial positions, 1d array
        :return: a 1d array ``n`` containing the density (between 0 and 1) at the given positions (z, r)
        """
        # Allocate relative density
        n = np.ones_like(z)

        # Make linear ramp
        n = np.where(
            z < job.sp.ramp_start + job.sp.ramp_length,
            (z - job.sp.ramp_start) / job.sp.ramp_length,
            n,
        )

        # Supress density before the ramp
        n = np.where(z < job.sp.ramp_start, 0.0, n)

        return n

    # plot density profile for checking
    all_z = np.linspace(job.sp.zmin, job.sp.p_zmax, 1000)
    dens = dens_func(all_z, 0.0)

    width_inch = job.sp.p_zmax / 1e-5
    major_locator = pyplot.MultipleLocator(10)
    minor_locator = pyplot.MultipleLocator(5)
    major_locator.MAXTICKS = 10000
    minor_locator.MAXTICKS = 10000

    def mark_on_plot(*, ax, parameter: str, y=1.1):
        ax.annotate(s=parameter, xy=(job.sp[parameter]*1e6, y), xycoords="data")
        ax.axvline(x=job.sp[parameter]*1e6, linestyle="--", color="red")
        return ax

    fig, ax = pyplot.subplots(figsize=(width_inch, 4.8))
    ax.plot(all_z*1e6, dens)
    ax.set_xlabel(r"$%s \;(\mu m)$" % "z")
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(job.sp.zmin*1e6 - 20, job.sp.p_zmax*1e6 + 20)
    ax.set_ylabel("Density profile $n$")
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)

    mark_on_plot(ax=ax, parameter="zmin")
    mark_on_plot(ax=ax, parameter="zmax")
    mark_on_plot(ax=ax, parameter="p_zmin", y=0.9)
    mark_on_plot(ax=ax, parameter="z0", y=0.8)
    mark_on_plot(ax=ax, parameter="ramp_start", y=0.7)
    mark_on_plot(ax=ax, parameter="L_interact")
    mark_on_plot(ax=ax, parameter="p_zmax")

    ax.annotate(s="ramp_start + ramp_length", xy=(job.sp.ramp_start*1e6 + job.sp.ramp_length*1e6, 1.1), xycoords="data")
    ax.axvline(x=job.sp.ramp_start*1e6 + job.sp.ramp_length*1e6, linestyle="--", color="red")

    ax.fill_between(all_z*1e6, dens, alpha=0.5)

    fig.savefig(job.fn("check_density.png"))

    # redirect stdout to "stdout.txt"
    orig_stdout = sys.stdout
    f = open(job.fn("stdout.txt"), "w")
    sys.stdout = f

    # Initialize the simulation object
    sim = Simulation(
        job.sp.Nz//8,
        job.sp.zmax,
        job.sp.Nr//8,
        job.sp.rmax,
        job.sp.Nm,
        job.sp.dt*8,
        n_e=None,  # no electrons
        zmin=job.sp.zmin,
        boundaries="open",
        n_order=-1,
        use_cuda=False,
        verbose_level=2,
    )

    # Add a laser to the fields of the simulation
    add_laser(sim, job.sp.a0, job.sp.w0, job.sp.ctau, job.sp.z0, lambda0=job.sp.lambda0)

    # Configure the moving window
    sim.set_moving_window(v=c_light)

    # Add diagnostics
    write_dir = os.path.join(job.ws, "prediags")
    sim.diags = [
        FieldDiagnostic(
            job.sp.diag_period, sim.fld, comm=sim.comm, write_dir=write_dir
        ),
    ]

    # set deterministic random seed
    np.random.seed(0)

    # Run the simulation for the length of the moving window box
    n_step = nstep(job)
    sim.step(n_step, show_progress=True)

    # redirect stdout back and close "stdout.txt"
    sys.stdout = orig_stdout
    f.close()

############
# PLOTTING #
############


def electric_field_amplitude_norm(lambda0=0.8e-6):
    """
    Computes the laser electric field amplitude for :math:`a_0=1`.

    :param lambda0: laser wavelength (meters)
    """
    # wavevector
    k0 = 2 * np.pi / lambda0

    # field amplitude
    e0 = m_e * c_light ** 2 * k0 / q_e

    return e0


def field_snapshot(
    tseries: OpenPMDTimeSeries,
    it: int,
    field_name: str,
    normalization_factor: float,
    coord: Optional[str] = None,
    m="all",
    theta=0.0,
    chop: Optional[List[float]] = None,
    path="./",
    **kwargs,
) -> None:
    """
    Plot the ``field_name`` field from ``tseries`` at step ``iter``.

    :param path: path to output file
    :param tseries: whole simulation time series
    :param it: time step in the simulation
    :param field_name: which field to extract, eg. 'rho', 'E', 'B' or 'J'
    :param normalization_factor: normalization factor for the extracted field
    :param coord: which component of the field to extract, eg. 'r', 't' or 'z'
    :param m: 'all' for extracting the sum of all the azimuthal modes
    :param theta: the angle of the plane of observation, with respect to the 'x' axis
    :param chop: adjusting extent of simulation box plot
    :param kwargs: extra plotting arguments, eg. labels, data limits etc.
    :return: saves field plot image to disk
    """
    if chop is None:
        chop = [0.0, 0.0, 0.0, 0.0]

    field, info = tseries.get_field(
        field=field_name, coord=coord, iteration=it, m=m, theta=theta
    )

    field *= normalization_factor

    plot = sliceplots.Plot2D(
        arr2d=field,
        h_axis=info.z * 1e6,
        v_axis=info.r * 1e6,
        xlabel=r"${} \;(\mu m)$".format(info.axes[1]),
        ylabel=r"${} \;(\mu m)$".format(info.axes[0]),
        extent=(
            info.zmin * 1e6 + chop[0],
            info.zmax * 1e6 + chop[1],
            info.rmin * 1e6 + chop[2],
            info.rmax * 1e6 + chop[3],
        ),
        cbar=True,
        text=f"iteration {it}",
        **kwargs,
    )

    filename = os.path.join(path, f"{field_name}{it:06d}.png")
    plot.canvas.print_figure(filename)


def get_a0(
    tseries: OpenPMDTimeSeries,
    t: Optional[float] = None,
    it: Optional[int] = None,
    coord="x",
    m="all",
    slicing_dir="y",
    theta=0.0,
    lambda0=0.8e-6,
) -> Tuple[float, float, float, float]:
    """
    Compute z₀, a₀, w₀, cτ.

    :param tseries: whole simulation time series
    :param t: time (in seconds) at which to obtain the data
    :param it: time step at which to obtain the data
    :param coord: which component of the field to extract
    :param m: 'all' for extracting the sum of all the modes
    :param slicing_dir: the direction along which to slice the data eg., 'x', 'y' or 'z'
    :param theta: the angle of the plane of observation, with respect to the 'x' axis
    :param lambda0: laser wavelength (meters)
    :return: z₀, a₀, w₀, cτ
    """
    # get E_x field in V/m
    electric_field_x, info_electric_field_x = tseries.get_field(
        field="E",
        coord=coord,
        t=t,
        iteration=it,
        m=m,
        theta=theta,
        slicing_dir=slicing_dir,
    )

    # normalized vector potential
    e0 = electric_field_amplitude_norm(lambda0=lambda0)
    a0 = electric_field_x / e0

    # get pulse envelope
    envelope = np.abs(hilbert(a0, axis=1))
    envelope_z = envelope[envelope.shape[0] // 2, :]

    a0_max = np.amax(envelope_z)

    # index of peak
    z_idx = np.argmax(envelope_z)
    # pulse peak position
    z0 = info_electric_field_x.z[z_idx]

    # FWHM perpendicular size of beam, proportional to w0
    fwhm_a0_w0 = (
        np.sum(np.greater_equal(envelope[:, z_idx], a0_max / 2))
        * info_electric_field_x.dr
    )

    # FWHM longitudinal size of the beam, proportional to ctau
    fwhm_a0_ctau = (
        np.sum(np.greater_equal(envelope_z, a0_max / 2)) * info_electric_field_x.dz
    )

    return z0, a0_max, fwhm_a0_w0, fwhm_a0_ctau


@Project.operation
@Project.pre(path_exists(os.path.join("prediags", "rhos")))
@Project.pre.after(run_fbpic)
@Project.post(are_files(("diags.txt", "all_hist.txt", "hist_edges.txt")))
def post_process_results(job: Job) -> None:
    """
    Loop through a whole simulation and, for *each ``fbpic`` iteration*:

    a. compute

        1. the iteration time ``it_time``
        2. position of the laser pulse peak ``z_0``
        3. normalized vector potential ``a_0``
        4. beam waist ``w_0``
        5. spatial pulse length ``c_tau``

    and write results to "diags.txt".

    b. compute the weighted particle energy histogram and save it to "all_hist.txt",
    and the histogram bins to "hist_edges.txt"

    c. save a snapshot of the plasma density field ``rho`` to {job_dir}/prediags/rhos/rho{it:06d}.png

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    h5_path: Union[bytes, str] = os.path.join(job.ws, "prediags", "hdf5")

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=False)

    diags_file = open(job.fn("diags.txt"), "w")
    diags_file.write("iteration,time[fs],z₀[μm],a₀,w₀[μm],cτ[μm]\n")

    # loop through all the iterations in the job's time series
    for idx, it in enumerate(time_series.iterations):
        it_time = it * job.sp.dt*8

        z_0, a_0, w_0, c_tau = get_a0(
            time_series, it=it, lambda0=job.sp.lambda0
        )

        diags_file.write(
            f"{it:06d},{it_time * 1e15:.3e},{z_0 * 1e6:.3e},{a_0:.3e},{w_0 * 1e6:.3e},{c_tau * 1e6:.3e}\n"
        )

    diags_file.close()


def add_create_dir_workflow(path: str) -> None:
    """
    Adds ``create_dir`` function(s) to the project workflow, for each value of ``path``.
    Can be called inside a loop, for multiple values of ``path``.
    For details, see `this recipe <https://github.com/glotzerlab/signac-docs/blob/master/docs/source/recipes.rst#how-to-define-parameter-dependent-operations>`_

    :param path: path to pass to ``create_dir``, and its pre- and post-conditions
    """
    # Make sure to make the operation-name a function of the parameter(s)!
    @Project.operation(f"create_dir_{path}".replace("/", "_"))
    @Project.pre(path_exists(os.path.dirname(path)))
    @Project.post(path_exists(path))
    def create_dir(job: Job) -> None:
        """
        Creates new ``path`` inside the job directory.

        :param job: the job instance is a handle to the data of a unique statepoint
        """
        full_path = os.path.join(job.ws, path)
        try:
            os.mkdir(full_path)
        except OSError:
            logger.warning("Creation of the directory %s failed" % full_path)
        else:
            logger.info("Successfully created the directory %s " % full_path)


add_create_dir_workflow(path=os.path.join("prediags", "rhos"))


@Project.operation
@Project.pre.isfile("diags.txt")
@Project.post(are_files(("a0.png", "w0.png", "ctau.png")))
def plot_1d_diags(job: Job) -> None:
    """
    Plot the 1D diagnostics, ``a_0``, ``w_0`` and ``c_tau`` vs ``z_0``.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    df_diags = pd.read_csv(job.fn("diags.txt"), header=0, index_col=0, comment="#")

    z_0 = df_diags.loc[:, "z₀[μm]"].values
    a_0 = df_diags.loc[:, "a₀"].values
    w_0 = df_diags.loc[:, "w₀[μm]"].values
    c_tau = df_diags.loc[:, "cτ[μm]"].values

    fig, ax = pyplot.subplots(figsize=(10, 6))
    sliceplots.plot1d(
        ax=ax,
        v_axis=a_0,  # y-axis
        h_axis=z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$%s$" % "a_0",
        xlim=[0, 900],  # TODO: hard-coded magic number
        ylim=[0, 10],  # TODO: hard-coded magic number
    )
    fig.savefig(job.fn("a0.png"))

    fig, ax = pyplot.subplots(figsize=(10, 6))
    sliceplots.plot1d(
        ax=ax,
        v_axis=w_0,  # y-axis
        h_axis=z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$%s \;(\mu m)$" % "w_0",
    )
    fig.savefig(job.fn("w0.png"))

    fig, ax = pyplot.subplots(figsize=(10, 6))
    sliceplots.plot1d(
        ax=ax,
        v_axis=c_tau,  # y-axis
        h_axis=z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$c \tau \;(\mu m)$",
    )
    fig.savefig(job.fn("ctau.png"))


@Project.operation
@Project.pre(path_exists(os.path.join("prediags", "rhos")))
@Project.pre(are_rho_pngs)
@Project.post.isfile("rho.mp4")
def generate_movie(job: Job) -> None:
    """
    Generate a movie from all the .png files in {job_dir}/prediags/rhos/

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    command = ffmpeg_command(
        frame_rate=10.0,  # TODO: hard-coded magic number
        input_files=os.path.join(job.ws, "prediags", "rhos", "rho*.png"),
        output_file=job.fn("rho.mp4"),
    )

    sh(command, shell=True)


def add_plot_snapshots_workflow(iteration: int) -> None:
    """
    Adds ``plot_snapshots`` function(s) to the project workflow, for each value of ``iteration``.

    :param iteration: iteration number to pass to ``plot_snapshots`` and its conditions
    """
    @Project.operation(f"plot_snapshots_{iteration:06d}")
    @Project.pre.after(run_fbpic)
    @Project.post(
        are_files(
            (
                f"E{iteration:06d}.png",
            )
        )
    )
    def plot_snapshots(job: Job) -> None:
        """
        Plot a snapshot of

        a. the electric field Ex

        corresponding to ``iteration``.

        :param job: the job instance is a handle to the data of a unique statepoint
        """
        e0 = electric_field_amplitude_norm(
            lambda0=job.sp.lambda0
        )

        h5_path: Union[bytes, str] = os.path.join(job.ws, "prediags", "hdf5")
        time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(
            h5_path, check_all_files=False
        )

        # plot electric field and save to disk
        field_snapshot(
            tseries=time_series,
            it=iteration,
            field_name="E",
            coord="x",
            normalization_factor=1.0 / e0,
            chop=[40, -20, 15, -15],  # TODO: hard-coded magic number
            path=job.ws,
            zlabel=r"$E_x/E_0$",
            vmin=-8,  # TODO: hard-coded magic number
            vmax=8,  # TODO: hard-coded magic number
            hslice_val=0.0,  # do a 1D slice through the middle of the simulation box
        )


for iteration_number in (16300,):  # TODO remove magic
    add_plot_snapshots_workflow(iteration=iteration_number)


if __name__ == "__main__":
    logging.basicConfig(
        filename=log_file_name,
        format="%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("==RUN STARTED==")

    Project().main()  # run the whole signac preproject workflow

    logger.info("==RUN FINISHED==")
