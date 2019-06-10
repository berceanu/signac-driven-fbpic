#!/usr/bin/env python3
"""This module contains the operation functions for this project.

The workflow defined in this file can be executed from the command
line with

    $ python src/project.py run [job_id [job_id ...]]

See also: $ python src/project.py --help
"""
import logging
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple, Union, Callable, Iterable

import numpy as np
import pandas as pd
import postproc.plotz as plotz
from flow import FlowProject
from opmd_viewer import OpenPMDTimeSeries
from scipy.constants import physical_constants
from scipy.signal import hilbert
from signac.contrib.job import Job

logger = logging.getLogger(__name__)
log_file_name = "fbpic-minimal-project.log"

c_light = physical_constants["speed of light in vacuum"][0]
m_e = physical_constants["electron mass"][0]
q_e = physical_constants["elementary charge"][0]
mc2 = m_e * c_light ** 2 / (q_e * 1e6)  # 0.511 MeV


#####################
# UTILITY FUNCTIONS #
#####################


def read_last_line(file_name: str) -> str:
    """
    Returns the last line of ``file_name``.
    See: `Stack Overflow <https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file/18603065#18603065>`_.

    :param file_name: name of input file
    :return: last line of text in the input file
    """
    with open(file_name, "rb") as f:
        _ = f.readline()  # Read the first line.
        f.seek(-2, os.SEEK_END)  # Jump to the second last byte.

        while f.read(1) != b"\n":  # Until EOL is found...
            # ...jump back the read byte plus one more.
            f.seek(-2, os.SEEK_CUR)

        last = f.readline()  # Read last line.

    return last.decode("utf-8")


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
    return lambda job: os.path.exists(os.path.join(job.ws, path))


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
    pass


####################
# OPERATION LABELS #
####################


def data_to_ascii(file_name: str) -> str:
    """
    Runs a series of ``sed`` substitions on ``file_name``, saving it as ``output_file_name``.

    :param file_name: input file name
    :return: output file name, poiting to modified file
    """
    stem, _ = os.path.splitext(file_name)
    output_file_name = stem + ".progress"

    shutil.copy(file_name, output_file_name)

    # replace '\r' by '\n'
    sh(rf"sed -i 's/\x0d/\x0a/g' {output_file_name}", shell=True)

    # replace '\u2588'(█) by '-'
    sh(rf"sed -i 's/\xe2\x96\x88/-/g' {output_file_name}", shell=True)

    # remove '<ESC>[K'
    sh(
        rf"sed -i -e $(echo -e 's/\033\[K//g') {output_file_name}",
        shell=True,
        executable="/bin/bash",
    )

    return output_file_name


@Project.label
def progress(job: Job) -> str:
    """
    Prints a progress bar showing the completion status of a certain ``fbpic`` run.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: |--------------              |
    """
    stdout_file = "stdout.txt"
    if job.isfile(stdout_file):
        output_file_name = data_to_ascii(job.fn(stdout_file))

        last_line = read_last_line(output_file_name)
        if last_line.startswith("|"):
            percentage = last_line
        else:  # already finished
            percentage = "100.00"
    else:  # didn't yet start
        percentage = "0.00"

    return percentage


###############################
# INITIALIZE & RUN SIMULATION #
###############################


def fbpic_ran(job: Job) -> bool:
    """
    Check if ``fbpic`` produced all the output .h5 files.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: True if all output files are in {job_dir}/diags/hdf5, False otherwise
    """
    h5_path: Union[bytes, str] = os.path.join(job.ws, "diags", "hdf5")
    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=True)
    iterations: np.ndarray = time_series.iterations

    # estimate iteration array based on input parameters
    estimated_iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=np.int)

    # check if iterations array corresponds to input params
    return np.array_equal(estimated_iterations, iterations)


def are_rho_pngs(job: Job) -> bool:
    """
    Check if all the {job_dir}/diags/rhos/rho{it:06d}.png files are present.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: True if .png files are there, False otherwise
    """
    files = os.listdir(os.path.join(job.ws, "diags", "rhos"))

    # estimate iteration array based on input parameters
    iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=np.int)

    pngs = (f"rho{it:06d}.png" for it in iterations)

    return set(files) == set(pngs)


@Project.operation
@Project.post(fbpic_ran)
def run_fbpic(job: Job) -> None:
    """
    This ``signac-flow`` operation runs a ``fbpic`` simulation.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    from fbpic.lpa_utils.laser import add_laser
    from fbpic.main import Simulation
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

    # redirect stdout to "stdout.txt"
    orig_stdout = sys.stdout
    f = open(job.fn("stdout.txt"), "w")
    sys.stdout = f

    # Initialize the simulation object
    sim = Simulation(
        job.sp.Nz,
        job.sp.zmax,
        job.sp.Nr,
        job.sp.rmax,
        job.sp.Nm,
        job.sp.dt,
        job.sp.p_zmin,
        job.sp.p_zmax,
        job.sp.p_rmin,
        job.sp.p_rmax,
        job.sp.p_nz,
        job.sp.p_nr,
        job.sp.p_nt,
        job.sp.n_e,
        dens_func=dens_func,
        zmin=job.sp.zmin,
        boundaries="open",
        n_order=-1,
        use_cuda=True,
        verbose_level=2,
    )

    # Add a laser to the fields of the simulation
    add_laser(sim, job.sp.a0, job.sp.w0, job.sp.ctau, job.sp.z0)

    # Configure the moving window
    sim.set_moving_window(v=c_light)

    # Add diagnostics
    write_dir = os.path.join(job.ws, "diags")
    sim.diags = [
        FieldDiagnostic(
            job.sp.diag_period, fldobject=sim.fld, comm=sim.comm, write_dir=write_dir
        ),
        ParticleDiagnostic(
            job.sp.diag_period,
            species={"electrons": sim.ptcl[0]},
            select={"uz": [1.0, None]},
            comm=sim.comm,
            write_dir=write_dir,
        ),
    ]

    # set deterministic random seed
    np.random.seed(0)

    # Run the simulation
    sim.step(job.sp.N_step, show_progress=True)

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


def particle_energy_histogram(
        tseries: OpenPMDTimeSeries, it: int, energy_min=1.0, energy_max=300.0, nbins=100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the weighted particle energy histogram from ``tseries`` at step ``iteration``.

    :param tseries: whole simulation time series
    :param it: time step in the simulation
    :param energy_min: lower energy threshold
    :param energy_max: upper energy threshold
    :param nbins: number of bins
    :return: histogram values and bin edges
    """
    delta_energy = (energy_max - energy_min) / nbins
    energy_bins = np.linspace(start=energy_min, stop=energy_max, num=nbins + 1)

    ux, uy, uz, w = tseries.get_particle(["ux", "uy", "uz", "w"], iteration=it)
    energy = mc2 * np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)

    hist, bin_edges = np.histogram(
        energy, bins=energy_bins, weights=q_e * 1e12 / delta_energy * w
    )

    return hist, bin_edges


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

    plot = plotz.Plot2D(
        field,
        info.z * 1e6,
        info.r * 1e6,
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
@Project.pre(path_exists(os.path.join("diags", "rhos")))
@Project.pre.after(run_fbpic)
@Project.post(are_files(("diags.txt", "all_hist.txt", "hist_edges.txt")))
@Project.post(are_rho_pngs)
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

    c. save a snapshot of the plasma density field ``rho`` to {job_dir}/diags/rhos/rho{it:06d}.png

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    h5_path: Union[bytes, str] = os.path.join(job.ws, "diags", "hdf5")
    rho_path: Union[bytes, str] = os.path.join(job.ws, "diags", "rhos")

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=False)
    number_of_iterations: int = time_series.iterations.size

    nbins = 349  # TODO: hard-coded magic number
    all_hist = np.empty(shape=(number_of_iterations, nbins), dtype=np.float64)
    hist_edges = np.empty(shape=(nbins + 1,), dtype=np.float64)

    diags_file = open(job.fn("diags.txt"), "w")
    diags_file.write("iteration,time[fs],z₀[μm],a₀,w₀[μm],cτ[μm]\n")

    # loop through all the iterations in the job's time series
    for idx, it in enumerate(time_series.iterations):
        it_time = it * job.sp.dt

        z_0, a_0, w_0, c_tau = get_a0(
            time_series, it=it, lambda0=0.8e-6
        )  # TODO: hard-coded magic number

        diags_file.write(
            f"{it:06d},{it_time * 1e15:.3e},{z_0 * 1e6:.3e},{a_0:.3e},{w_0 * 1e6:.3e},{c_tau * 1e6:.3e}\n"
        )

        # generate 1D energy histogram
        energy_hist, bin_edges = particle_energy_histogram(
            tseries=time_series,
            it=it,
            energy_min=1.0,  # TODO hard-coded magic number
            energy_max=350.0,  # TODO hard-coded magic number
            nbins=nbins,
        )
        # build up arrays for 2D energy histogram
        all_hist[idx, :] = energy_hist
        if idx == 0:  # only save the first one
            hist_edges[:] = bin_edges

        # save "rho{it:06d}.png"
        field_snapshot(
            tseries=time_series,
            it=it,
            field_name="rho",
            normalization_factor=1.0 / (-q_e * job.sp.n_e),
            chop=[40, -20, 15, -15],  # TODO hard-coded magic number
            path=rho_path,
            zlabel=r"$n/n_e$",
            vmin=0,  # TODO hard-coded magic number
            vmax=3,  # TODO hard-coded magic number
        )

    diags_file.close()

    np.savetxt(
        job.fn("all_hist.txt"),
        all_hist,
        header="One iteration per row, containing the energy histogram.",
    )
    np.savetxt(job.fn("hist_edges.txt"), hist_edges, header="Energy histogram bins.")


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


add_create_dir_workflow(path=os.path.join("diags", "rhos"))

# TODO move .png and .mp4 from {job_dir} to {job_dir}/diags
# TODO bash script that allows single project.py run for GPU + postproc

@Project.operation
@Project.pre(are_files(("diags.txt", "all_hist.txt", "hist_edges.txt")))
@Project.post.isfile("hist2d.png")
def plot_2d_hist(job: Job) -> None:
    """
    Plot the 2D histogram, composed of the 1D slices for each iteration.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    df_diags = pd.read_csv(job.fn("diags.txt"), header=0, index_col=0, comment="#")
    all_hist = np.loadtxt(
        job.fn("all_hist.txt"), dtype=np.float64, comments="#", ndmin=2
    )
    hist_edges = np.loadtxt(
        job.fn("hist_edges.txt"), dtype=np.float64, comments="#", ndmin=1
    )

    z_0 = df_diags.loc[:, "z₀[μm]"]

    # plot 2D energy-charge histogram
    hist2d = plotz.Plot2D(
        all_hist.T,  # 2D data
        z_0.values,  # x-axis
        hist_edges[1:],  # y-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"E (MeV)",
        zlabel=r"dQ/dE (pC/MeV)",
        vslice_val=z_0.loc[35100],  # TODO: hard-coded magic number
        extent=(z_0.iloc[0], z_0.iloc[-1], hist_edges[1], hist_edges[-1]),
        vmin=0,  # TODO: hard-coded magic number
        vmax=10,  # TODO: hard-coded magic number
    )
    hist2d.canvas.print_figure(job.fn("hist2d.png"))


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

    # plot a_0 vs z_0
    plot_1d = plotz.Plot1D(
        a_0,  # y-axis
        z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$%s$" % "a_0",
        xlim=[0, 900],  # TODO: hard-coded magic number
        ylim=[0, 10],  # TODO: hard-coded magic number
        figsize=(10, 6),
    )
    plot_1d.canvas.print_figure(job.fn("a0.png"))

    # plot w_0 vs z_0
    plot_1d = plotz.Plot1D(
        w_0,  # y-axis
        z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$%s \;(\mu m)$" % "w_0",
        figsize=(10, 6),
    )
    plot_1d.canvas.print_figure(job.fn("w0.png"))

    # plot c_tau vs z_0
    plot_1d = plotz.Plot1D(
        c_tau,  # x-axis
        z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$c \tau \;(\mu m)$",
        figsize=(10, 6),
    )
    plot_1d.canvas.print_figure(job.fn("ctau.png"))


@Project.operation
@Project.pre(path_exists(os.path.join("diags", "rhos")))
@Project.pre(are_rho_pngs)
@Project.post.isfile("rho.mp4")
def generate_movie(job: Job) -> None:
    """
    Generate a movie from all the .png files in {job_dir}/diags/rhos/

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    command = ffmpeg_command(
        frame_rate=10.0,  # TODO: hard-coded magic number
        input_files=os.path.join(job.ws, "diags", "rhos", "rho*.png"),
        output_file=job.fn("rho.mp4"),
    )

    sh(command, shell=True)


# TODO apply parametrization recipe
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post.isfile("E035100.png")  # TODO: hard-coded magic number
def plot_efield_snapshot(job: Job) -> None:
    """
    Plot a snapshot of the electric field from a given iteration number (35100 in this case).

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    e0 = electric_field_amplitude_norm(lambda0=0.8e-6)  # TODO: hard-coded magic number

    h5_path: Union[bytes, str] = os.path.join(job.ws, "diags", "hdf5")
    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=False)
    # TODO no need to load the whole time series, can extract directly the relevant iteration

    # plot electric field and save to disk
    field_snapshot(
        tseries=time_series,
        it=35100,  # TODO: hard-coded magic number
        field_name="E",
        coord="x",
        normalization_factor=1.0 / e0,
        chop=[40, -20, 15, -15],  # TODO: hard-coded magic number
        path=job.ws,
        zlabel=r"$E_x / E_0$",
        vmin=-8,  # TODO: hard-coded magic number
        vmax=8,  # TODO: hard-coded magic number
        hslice_val=0.0,  # do a 1D slice through the middle of the simulation box
    )
# TODO plot 1D energy histogram for certain iteration (see analysis.py)
# TODO plot rho field for certain iteration

# TODO animation of the whole plot panel: energy histogram, rho field, electric field

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
