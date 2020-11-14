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
import os
import subprocess
import glob
from typing import List, Optional, Union, Callable
from reader import read_density

import numpy as np
import sliceplots
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from matplotlib import pyplot
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import physical_constants
from signac.contrib.job import Job

logger = logging.getLogger(__name__)
log_file_name = "fbpic-project.log"

c_light = physical_constants["speed of light in vacuum"][0]
m_e = physical_constants["electron mass"][0]
m_p = physical_constants["proton mass"][0]
q_e = physical_constants["elementary charge"][0]
mc2 = m_e * c_light ** 2 / (q_e * 1e6)  # 0.511 MeV


class OdinEnvironment(DefaultSlurmEnvironment):
    """Environment profile for the LGED cluster.
    https://docs.signac.io/projects/flow/en/latest/supported_environments/comet.html#flow.environments.xsede.CometEnvironment
    """

    hostname_pattern = r".*\.ra5\.eli-np\.ro$"
    template = "odin.sh"
    cores_per_node = 48
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
            "--job-output",
            help=(
                "What to name the job output file. "
                "If omitted, uses the system default "
                '(slurm default is "slurm-%%j.out").'
            ),
        )


#####################
# UTILITY FUNCTIONS #
#####################


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
    frame_rate: float = 10.0,  # CHANGEME
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


####################
# OPERATION LABELS #
####################


@Project.label
def progress(job) -> str:
    # get last iteration based on input parameters
    number_of_iterations = math.ceil((job.sp.N_step - 0) / job.sp.diag_period)

    h5_path = os.path.join(job.ws, "diags", "hdf5")
    if not os.path.isdir(h5_path):
        # {job_dir}/diags/hdf5 not present, ``fbpic`` didn't run
        return "0/%s" % number_of_iterations

    h5_files = glob.glob(os.path.join(h5_path, "*.h5"))

    return f"{len(h5_files)}/{number_of_iterations}"


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
    if not os.path.isdir(h5_path):
        # {job_dir}/diags/hdf5 not present, ``fbpic`` didn't run
        did_it_run = False
        return did_it_run

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=True)
    iterations: np.ndarray = time_series.iterations

    # estimate iteration array based on input parameters
    estimated_iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=np.int)

    # check if iterations array corresponds to input params
    did_it_run = np.array_equal(estimated_iterations, iterations)

    return did_it_run


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
@directives(ngpu=1)
@Project.post(fbpic_ran)
def run_fbpic(job: Job) -> None:
    """
    This ``signac-flow`` operation runs a ``fbpic`` simulation.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    from fbpic.main import Simulation
    from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic
    from fbpic.lpa_utils.bunch import add_particle_bunch_file
    from scipy import interpolate

    position_m, norm_density = read_density("density_1_inlet_spacers.txt")

    interp_z_min = position_m.min()
    interp_z_max = position_m.max()

    rho = interpolate.interp1d(
        position_m, norm_density, bounds_error=False, fill_value=(0.0, 0.0)
    )

    # The density profile
    def dens_func(z: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Returns relative density at position z and r.

        :param z: longitudinal positions, 1d array
        :param r: radial positions, 1d array
        :return: a 1d array ``n`` containing the density (between 0 and 1) at the given positions (z, r)
        """
        # Allocate relative density
        n = np.ones_like(z)

        # only compute n if z is inside the interpolation bounds
        n = np.where(np.logical_and(z > interp_z_min, z < interp_z_max), rho(z), n)

        # Make linear ramp
        n = np.where(
            z < job.sp.ramp_start + job.sp.ramp_length,
            (z - job.sp.ramp_start) / job.sp.ramp_length * rho(interp_z_min),
            n,
        )

        # Supress density before the ramp
        n = np.where(z < job.sp.ramp_start, 0.0, n)

        return n

    # plot density profile for checking
    all_z = np.linspace(job.sp.zmin, job.sp.p_zmax, 1000)
    dens = dens_func(all_z, 0.0)

    def mark_on_plot(*, ax, parameter: str, y=1.1):
        ax.annotate(text=parameter, xy=(job.sp[parameter] * 1e6, y), xycoords="data")
        ax.axvline(x=job.sp[parameter] * 1e6, linestyle="--", color="red")
        return ax

    fig, ax = pyplot.subplots(figsize=(30, 4.8))
    ax.plot(all_z * 1e6, dens)
    ax.set_xlabel(r"$%s \;(\mu m)$" % "z")
    ax.set_ylim(-0.1, 1.2)
    ax.set_xlim(job.sp.zmin * 1e6 - 20, job.sp.p_zmax * 1e6 + 20)
    ax.set_ylabel("Density profile $n$")

    mark_on_plot(ax=ax, parameter="zmin")
    mark_on_plot(ax=ax, parameter="zmax")
    mark_on_plot(ax=ax, parameter="p_zmin", y=0.9)
    mark_on_plot(ax=ax, parameter="ramp_start", y=0.7)
    mark_on_plot(ax=ax, parameter="L_interact")
    mark_on_plot(ax=ax, parameter="p_zmax")

    ax.annotate(
        text="ramp_start + ramp_length",
        xy=(job.sp.ramp_start * 1e6 + job.sp.ramp_length * 1e6, 1.1),
        xycoords="data",
    )
    ax.axvline(
        x=job.sp.ramp_start * 1e6 + job.sp.ramp_length * 1e6,
        linestyle="--",
        color="red",
    )

    ax.fill_between(all_z * 1e6, dens, alpha=0.5)

    fig.savefig(job.fn("check_density.png"))

    # redirect stdout to "stdout.txt"
    # orig_stdout = sys.stdout
    # f = open(job.fn("stdout.txt"), "w")
    # sys.stdout = f
    # FIXME

    # Initialize the simulation object
    sim = Simulation(
        job.sp.Nz,
        job.sp.zmax,
        job.sp.Nr,
        job.sp.rmax,
        job.sp.Nm,
        job.sp.dt,
        zmin=job.sp.zmin,
        boundaries={"z": "open", "r": "open"},
        n_order=-1,
        use_cuda=True,
        verbose_level=2,
    )
    # 'r': 'open' can also be used, but is more computationally expensive

    # Add the plasma electron and plasma ions
    plasma_elec = sim.add_new_species(
        q=-q_e,
        m=m_e,
        n=job.sp.n_e,
        dens_func=dens_func,
        p_zmin=job.sp.p_zmin,
        p_zmax=job.sp.p_zmax,
        p_rmax=job.sp.p_rmax,
        p_nz=job.sp.p_nz,
        p_nr=job.sp.p_nr,
        p_nt=job.sp.p_nt,
    )
    plasma_ions = sim.add_new_species(
        q=q_e,
        m=m_p,
        n=job.sp.n_e,
        dens_func=dens_func,
        p_zmin=job.sp.p_zmin,
        p_zmax=job.sp.p_zmax,
        p_rmax=job.sp.p_rmax,
        p_nz=job.sp.p_nz,
        p_nr=job.sp.p_nr,
        p_nt=job.sp.p_nt,
    )

    # The electron beam
    # TODO should this be used?
    # L0 = 100.0e-6  # Position at which the beam should be "unfreezed"
    Qtot = 200.0e-12  # Charge in Coulomb

    # particles beam from txt file
    bunch = add_particle_bunch_file(
        sim=sim,
        q=-q_e,
        m=m_e,
        filename="exp_4deg.txt",
        n_physical_particles=Qtot / q_e,
        z_off=-1900e-6,
    )

    # Configure the moving window
    sim.set_moving_window(v=c_light)

    # Add diagnostics
    write_dir = os.path.join(job.ws, "diags")
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
            species={"electrons": plasma_elec, "bunch": bunch},
            comm=sim.comm,
            write_dir=write_dir,
        ),
    ]

    # set deterministic random seed
    np.random.seed(0)

    # Run the simulation
    sim.step(job.sp.N_step, show_progress=True)

    # redirect stdout back and close "stdout.txt"
    # sys.stdout = orig_stdout
    # f.close()
    # FIXME


############
# PLOTTING #
############


def field_snapshot(
    tseries: OpenPMDTimeSeries,
    it: int,
    field_name: str,
    normalization_factor=1,
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
    if chop is None:  # how much to cut out from simulation domain
        chop = [0, 0, 0, 0]  # CHANGEME

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


@Project.operation
@Project.pre(path_exists(os.path.join("diags", "rhos")))
@Project.pre.after(run_fbpic)
@Project.post(are_rho_pngs)
def post_process_results(job: Job) -> None:
    """
    Loop through a whole simulation and, for *each ``fbpic`` iteration*:

    a. save a snapshot of the plasma density field ``rho`` to {job_dir}/diags/rhos/rho{it:06d}.png

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    h5_path: Union[bytes, str] = os.path.join(job.ws, "diags", "hdf5")
    rho_path: Union[bytes, str] = os.path.join(job.ws, "diags", "rhos")

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=False)

    # loop through all the iterations in the job's time series
    for it in time_series.iterations:
        # save "rho{it:06d}.png"
        field_snapshot(
            tseries=time_series,
            it=it,
            field_name="rho",
            normalization_factor=1.0 / (-q_e * job.sp.n_e),
            path=rho_path,
            zlabel=r"$n/n_e$",
            vmin=0,
            vmax=1,  # CHANGEME
            hslice_val=0,
        )

    # the field "rho" has (SI) units of charge/volume (Q/V), C/(m^3)
    # the initial density n_e has units of N/V, N = electron number
    # multiply by electron charge -q_e to get (N e) / V
    # so we get Q / N e, which is C/C, i.e. dimensionless
    # Note: one can also normalize by the critical density n_c


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
        input_files=os.path.join(job.ws, "diags", "rhos", "rho*.png"),
        output_file=job.fn("rho.mp4"),
    )

    sh(command, shell=True)


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
    # TODO remove src/trash/
