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
import subprocess
import glob
from typing import List, Optional, Union, Iterable, Callable, Tuple

import numpy as np
import pandas as pd
import sliceplots
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from matplotlib import pyplot
from openpmd_viewer import OpenPMDTimeSeries
from scipy.constants import physical_constants
from scipy.signal import hilbert
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
            default=36,
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


#####################
# UTILITY FUNCTIONS #
#####################


def are_files(file_names: Iterable[str]) -> Callable[[Job], bool]:
    """Check if given file names are in the ``job`` dir.
    Useful for pre- and post- operation conditions.

    :param file_names: iterable containing file names
    :return: anonymous function that does the check
    """
    return lambda job: all(job.isfile(fn) for fn in file_names)


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
    from fbpic.openpmd_diag import (
        FieldDiagnostic,
        ParticleDiagnostic,
        ParticleChargeDensityDiagnostic,
    )

    def ramp(z, *, center, sigma, p):
        """Gaussian-like function."""
        return np.exp(-(((z - center) / sigma) ** p))

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
    # orig_stdout = sys.stdout
    # f = open(job.fn("stdout.txt"), "w")
    # sys.stdout = f

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
        tau=job.sp.tau,
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
        # Since rho from `FieldDiagnostic` is 0 almost everywhere
        # (neutral plasma), it is useful to see the charge density
        # of individual particles
        ParticleChargeDensityDiagnostic(
            period=job.sp.diag_period,
            sim=sim,
            species={"electrons": plasma_elec},
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
    # sys.stdout = orig_stdout
    # f.close()


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
    tseries: OpenPMDTimeSeries,
    it: int,
    energy_min=1,
    energy_max=800,
    delta_energy=1,
    cutoff=35,  # CHANGEME
):
    """
    Compute the weighted particle energy histogram from ``tseries`` at step ``iteration``.

    :param tseries: whole simulation time series
    :param it: time step in the simulation
    :param energy_min: lower energy threshold (MeV)
    :param energy_max: upper energy threshold (MeV)
    :param delta_energy: size of each energy bin (MeV)
    :param cutoff: upper threshold for the histogram, in pC / MeV
    :return: histogram values and bin edges
    """
    nbins = (energy_max - energy_min) // delta_energy
    energy_bins = np.linspace(start=energy_min, stop=energy_max, num=nbins + 1)

    # FIXME get_particle now returns particle positions in meters instead of microns
    ux, uy, uz, w = tseries.get_particle(["ux", "uy", "uz", "w"], iteration=it)
    energy = mc2 * np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)

    # Explanation of weights:
    #     1. convert electron charge from C to pC (factor 1e12)
    #     2. multiply by weight w to get real number of electrons
    #     3. divide by energy bin size delta_energy to get charge / MeV
    hist, _ = np.histogram(
        energy, bins=energy_bins, weights=q_e * 1e12 / delta_energy * w
    )

    # cut off histogram
    np.clip(hist, a_min=None, a_max=cutoff, out=hist)

    return hist, energy_bins, nbins


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


def get_a0(
    tseries: OpenPMDTimeSeries,
    t: Optional[float] = None,
    it: Optional[int] = None,
    coord="x",
    m="all",
    slice_across=None,
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
    :param slice_across: the direction along which to slice the data eg., 'x', 'y' or 'z'
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
        slice_across=slice_across,
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


@ex
@Project.operation
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
    rho_path: Union[bytes, str] = os.path.join(job.ws, "rhos")

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=False)
    number_of_iterations: int = time_series.iterations.size

    # Do a mock histogram in order to get the number of bins
    _, _, nrbins = particle_energy_histogram(
        tseries=time_series,
        it=0,
    )
    all_hist = np.empty(shape=(number_of_iterations, nrbins), dtype=np.float64)
    hist_edges = np.empty(shape=(nrbins + 1,), dtype=np.float64)

    diags_file = open(job.fn("diags.txt"), "w")
    diags_file.write("iteration,time[fs],z₀[μm],a₀,w₀[μm],cτ[μm]\n")

    # loop through all the iterations in the job's time series
    for idx, it in enumerate(time_series.iterations):
        it_time = it * job.sp.dt

        z_0, a_0, w_0, c_tau = get_a0(time_series, it=it, lambda0=job.sp.lambda0)

        diags_file.write(
            f"{it:06d},{it_time * 1e15:.3e},{z_0 * 1e6:.3e},{a_0:.3e},{w_0 * 1e6:.3e},{c_tau * 1e6:.3e}\n"
        )

        # generate 1D energy histogram
        energy_hist, bin_edges, _ = particle_energy_histogram(
            tseries=time_series,
            it=it,
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
            path=rho_path,
            zlabel=r"$n/n_e$",
            vmin=0,
            vmax=2,  # CHANGEME
            hslice_val=0,
        )

    # the field "rho" has (SI) units of charge/volume (Q/V), C/(m^3)
    # the initial density n_e has units of N/V, N = electron number
    # multiply by electron charge -q_e to get (N e) / V
    # so we get Q / N e, which is C/C, i.e. dimensionless
    # Note: one can also normalize by the critical density n_c

    diags_file.close()

    np.savetxt(
        job.fn("all_hist.txt"),
        all_hist,
        header="One iteration per row, containing the energy histogram.",
    )
    np.savetxt(job.fn("hist_edges.txt"), hist_edges, header="Energy histogram bins.")


@ex
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
    hist2d = sliceplots.Plot2D(
        arr2d=all_hist.T,  # 2D data
        h_axis=z_0.values,  # x-axis
        v_axis=hist_edges[1:],  # y-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"E (MeV)",
        zlabel=r"dQ/dE (pC/MeV)",
        vslice_val=z_0.iloc[-1],  # can be changed to z_0.loc[iteration]
        extent=(z_0.iloc[0], z_0.iloc[-1], hist_edges[1], hist_edges[-1]),
    )
    hist2d.canvas.print_figure(job.fn("hist2d.png"))


@ex
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
        ylim=[0, 10],  # CHANGEME
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
