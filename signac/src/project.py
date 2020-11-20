##!/usr/bin/env python3
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
import sys
import subprocess
import glob
from typing import List, Optional, Union

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


ex = Project.make_group(name="ex")


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
    Check if all the {job_dir}/rhos/rho{it:06d}.png files are present.

    :param job: the job instance is a handle to the data of a unique statepoint
    :return: True if .png files are there, False otherwise
    """
    files = os.listdir(os.path.join(job.ws, "rhos"))

    # estimate iteration array based on input parameters
    iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=np.int)

    pngs = (f"rho{it:06d}.png" for it in iterations)

    return set(files) == set(pngs)


@ex.with_directives(directives=dict(ngpu=1))
@directives(ngpu=1)
@Project.operation
@Project.post(fbpic_ran)
def run_fbpic(job: Job) -> None:
    """
    This ``signac-flow`` operation runs a ``fbpic`` simulation.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    from fbpic.main import Simulation
    from fbpic.lpa_utils.laser import add_laser_pulse, GaussianLaser
    from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

    # The density profile
    def dens_func(z, r):
        """
        User-defined function: density profile of the plasma

        It should return the relative density with respect to n_plasma,
        at the position x, y, z (i.e. return a number between 0 and 1)

        Parameters
        ----------
        z, r: 1darrays of floats
            Arrays with one element per macroparticle
        Returns
        -------
        n : 1d array of floats
            Array of relative density, with one element per macroparticles
        """

        def ramp(z, *, center, sigma, p):
            """Gaussian-like function."""
            return np.exp(-(((z - center) / sigma) ** p))

        # Allocate relative density
        n = np.ones_like(z)

        # before up-ramp
        n = np.where(z < 0.0, 0.0, n)

        # Make up-ramp
        n = np.where(
            z < job.sp.center_left,
            ramp(z, center=job.sp.center_left, sigma=job.sp.sigma_left, p=job.sp.power),
            n,
        )

        # Make down-ramp
        n = np.where(
            (z >= job.sp.center_right)
            & (z < job.sp.center_right + 2 * job.sp.sigma_right),
            ramp(
                z, center=job.sp.center_right, sigma=job.sp.sigma_right, p=job.sp.power
            ),
            n,
        )

        # after down-ramp
        n = np.where(z >= job.sp.center_right + 2 * job.sp.sigma_right, 0, n)

        return n

    # plot density profile for checking
    all_z = np.linspace(job.sp.zmin, job.sp.L_interact, 1000)
    dens = dens_func(all_z, 0.0)

    def mark_on_plot(*, ax, parameter: str, y=1.1):
        ax.annotate(text=parameter, xy=(job.sp[parameter] * 1e6, y), xycoords="data")
        ax.axvline(x=job.sp[parameter] * 1e6, linestyle="--", color="red")
        return ax

    fig, ax = pyplot.subplots(figsize=(30, 4.8))
    ax.plot(all_z * 1e6, dens)
    ax.set_xlabel(r"$%s \;(\mu m)$" % "z")
    ax.set_ylim(0.0, 1.2)
    ax.set_xlim(job.sp.zmin * 1e6 - 20, job.sp.L_interact * 1e6 + 20)
    ax.set_ylabel("Density profile $n$")

    mark_on_plot(ax=ax, parameter="zmin")
    mark_on_plot(ax=ax, parameter="zmax")
    mark_on_plot(ax=ax, parameter="p_zmin", y=0.9)
    mark_on_plot(ax=ax, parameter="center_left", y=0.7)
    mark_on_plot(ax=ax, parameter="center_right", y=0.7)
    mark_on_plot(ax=ax, parameter="L_interact", y=0.7)
    mark_on_plot(ax=ax, parameter="p_zmax")

    ax.fill_between(all_z * 1e6, dens, alpha=0.5)

    fig.savefig(job.fn("check_density.png"))
    pyplot.close(fig)

    # redirect stdout to "stdout.txt"
    #orig_stdout = sys.stdout
    #f = open(job.fn("stdout.txt"), "w")
    #sys.stdout = f

    # Initialize the simulation object
    sim = Simulation(
        job.sp.Nz,
        job.sp.zmax,
        job.sp.Nr,
        job.sp.rmax,
        job.sp.Nm,
        job.sp.dt,
        n_e=None,  # no electrons
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

    # Create a Gaussian laser profile
    laser_profile = GaussianLaser(
        a0=job.sp.a0,
        waist=job.sp.w0,
        tau=job.sp.ctau / c_light,
        z0=job.sp.z0,
        zf=job.sp.zfoc,
        theta_pol=0.0,
        lambda0=job.sp.lambda0,
        cep_phase=0.0,
        phi2_chirp=0.0,
        propagation_direction=1,
    )
    # Add it to the simulation
    add_laser_pulse(
        sim,
        laser_profile,
        gamma_boost=None,
        method="direct",
        z0_antenna=None,
        v_antenna=0.0,
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
            species={"electrons": plasma_elec},
            # select={"uz": [40.0, None]},
            comm=sim.comm,
            write_dir=write_dir,
        ),
    ]
    # TODO add electron tracking

    # Plot the Ex component of the laser
    # Get the fields in the half-plane theta=0 (Sum mode 0 and mode 1)
    gathered_grids = [sim.comm.gather_grid(sim.fld.interp[m]) for m in range(job.sp.Nm)]

    rgrid = gathered_grids[0].r
    zgrid = gathered_grids[0].z

    # construct the Er field for theta=0
    Er = gathered_grids[0].Er.T.real

    for m in range(1, job.sp.Nm):
        # There is a factor 2 here so as to comply with the convention in
        # Lifschitz et al., which is also the convention adopted in Warp Circ
        Er += 2 * gathered_grids[m].Er.T.real

    e0 = electric_field_amplitude_norm(lambda0=job.sp.lambda0)

    fig = pyplot.figure(figsize=(8, 8))
    sliceplots.Plot2D(
        fig=fig,
        arr2d=Er / e0,
        h_axis=zgrid * 1e6,
        v_axis=rgrid * 1e6,
        zlabel=r"$E_r/E_0$",
        xlabel=r"$z \;(\mu m)$",
        ylabel=r"$r \;(\mu m)$",
        extent=(
            zgrid[0] * 1e6,  # + 40
            zgrid[-1] * 1e6,  # - 20
            rgrid[0] * 1e6,
            rgrid[-1] * 1e6,  # - 15,
        ),
        cbar=True,
        vmin=-3,
        vmax=3,
        hslice_val=0.0,  # do a 1D slice through the middle of the simulation box
    )
    fig.savefig(job.fn("check_laser.png"))
    pyplot.close(fig)

    # set deterministic random seed
    np.random.seed(0)

    # Run the simulation
    sim.step(job.sp.N_step, show_progress=True)

    # redirect stdout back and close "stdout.txt"
    #sys.stdout = orig_stdout
    #f.close()


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


@ex
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post(are_rho_pngs)
def post_process_results(job: Job) -> None:
    """
    Loop through a whole simulation and, for *each ``fbpic`` iteration*:

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    h5_path: Union[bytes, str] = os.path.join(job.ws, "diags", "hdf5")
    rho_path: Union[bytes, str] = os.path.join(job.ws, "rhos")

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
            vmin=0.0,
            vmax=2.0,  # CHANGEME
            hslice_val=0,
        )

    # the field "rho" has (SI) units of charge/volume (Q/V), C/(m^3)
    # the initial density n_e has units of N/V, N = electron number
    # multiply by electron charge -q_e to get (N e) / V
    # so we get Q / N e, which is C/C, i.e. dimensionless
    # Note: one can also normalize by the critical density n_c


@ex
@Project.operation
@Project.pre(are_rho_pngs)
@Project.post.isfile("rho.mp4")
def generate_rho_movie(job: Job) -> None:
    """
    Generate a movie from all the .png files in {job_dir}/rhos/

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    command = ffmpeg_command(
        input_files=os.path.join(job.ws, "rhos", "rho*.png"),
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
