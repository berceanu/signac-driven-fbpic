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
from copy import copy
from typing import Union, Iterable, Callable, Tuple
import pathlib

import numpy as np
import pandas as pd
import sliceplots
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from matplotlib import pyplot, colors, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import colorcet as cc
from openpmd_viewer import addons
import unyt as u
from signac.contrib.job import Job

logger = logging.getLogger(__name__)
log_file_name = "fbpic-project.log"

# strip units
c_light = u.clight.to_value("m/s")
m_e = u.electron_mass.to_value("kg")
m_p = u.proton_mass.to_value("kg")
q_e = u.electron_charge.to_value("C")  # negative sign
q_p = u.proton_charge.to_value("C")  # positive sign
mc2 = (u.electron_mass * u.clight ** 2).to_value("MeV")
# 0.511 MeV


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

    time_series = addons.LpaDiagnostics(h5_path, check_all_files=True)
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
        zmin=job.sp.zmin,
        boundaries={"z": "open", "r": "reflective"},
        n_order=-1,
        use_cuda=True,
        verbose_level=2,
    )
    # 'r': 'open' can also be used, but is more computationally expensive

    # Add the plasma electrons
    plasma_elec = sim.add_new_species(
        q=q_e,
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


def particle_energy_histogram(
    tseries,
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

    ux, uy, uz, w = tseries.get_particle(["ux", "uy", "uz", "w"], iteration=it)
    energy = mc2 * np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)

    # Explanation of weights:
    #     1. convert electron charge from C to pC (factor 1e12)
    #     2. multiply by weight w to get real number of electrons
    #     3. divide by energy bin size delta_energy to get charge / MeV
    hist, _ = np.histogram(
        energy,
        bins=energy_bins,
        weights=u.elementary_charge.to_value("pC") / delta_energy * w,
    )

    # cut off histogram
    np.clip(hist, a_min=None, a_max=cutoff, out=hist)

    return hist, energy_bins, nbins


def laser_density_plot(
    tseries,
    iteration: int,
    rho_field_name="rho_electrons",
    laser_polarization="x",
    save_path=pathlib.Path.cwd(),
    n_c=1.7419595910637713e27,  # 1/m^3
    E0=4013376052599.5396,  # V/m
) -> None:
    """
    Plot on the same figure the laser pulse envelope and the electron density.
    """

    laser_cmap = copy(cc.m_fire)
    laser_cmap.set_under("black", alpha=0)

    rho, rho_info = tseries.get_field(
        field=rho_field_name,
        iteration=iteration,
    )
    envelope, env_info = tseries.get_laser_envelope(
        iteration=iteration, pol=laser_polarization
    )
    # get longitudinal field
    e_z_of_z, e_z_of_z_info = tseries.get_field(
        field="E",
        coord="z",
        iteration=iteration,
        slice_across="r",
    )

    fig, ax = pyplot.subplots(figsize=(10, 6))

    im_rho = ax.imshow(
        rho / (np.abs(q_e) * n_c),
        extent=rho_info.imshow_extent * 1e6,  # conversion to microns
        origin="lower",
        norm=colors.SymLogNorm(linthresh=1e-4, linscale=0.15, base=10),
        cmap=cm.get_cmap("cividis"),
    )
    im_envelope = ax.imshow(
        envelope / E0,
        extent=env_info.imshow_extent * 1e6,
        origin="lower",
        cmap=laser_cmap,
    )
    im_envelope.set_clim(vmin=1.0)

    # plot longitudinal field
    ax.plot(e_z_of_z_info.z * 1e6, e_z_of_z / E0 * 25 - 18, color="0.75")
    ax.axhline(-18, color="0.65", ls="-.")

    cbaxes_rho = inset_axes(
        ax,
        width="3%",  # width = 10% of parent_bbox width
        height="46%",  # height : 50%
        loc=2,
        bbox_to_anchor=(1.01, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbaxes_env = inset_axes(
        ax,
        width="3%",  # width = 5% of parent_bbox width
        height="46%",  # height : 50%
        loc=3,
        bbox_to_anchor=(1.01, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar_env = fig.colorbar(
        mappable=im_envelope,
        orientation="vertical",
        ticklocation="right",
        cax=cbaxes_env,
    )
    cbar_rho = fig.colorbar(
        mappable=im_rho, orientation="vertical", ticklocation="right", cax=cbaxes_rho
    )
    cbar_env.set_label(r"$eE_{x} / m c \omega_\mathrm{L}$")
    cbar_rho.set_label(r"$n_{e} / n_\mathrm{cr}$")
    # cbar_rho.set_ticks([1e-4,1e-2,1e0]) FIXME

    ax.set_ylabel(r"${} \;(\mu m)$".format(rho_info.axes[0]))
    ax.set_xlabel(r"${} \;(\mu m)$".format(rho_info.axes[1]))

    current_time = (tseries.current_t * u.second).to("picosecond")
    ax.set_title(f"t = {current_time:.2f}")

    filename = save_path / f"rho{iteration:06d}.png"

    fig.subplots_adjust(right=0.85)
    fig.savefig(filename)
    pyplot.close(fig)


def get_scalar_diagnostics(
    tseries,
    iteration: int,
    laser_polarization="x",
) -> Tuple[float, float, float, float]:
    """Compute z₀, a₀, w₀, cτ."""

    a0 = tseries.get_a0(iteration=iteration, pol=laser_polarization)
    w0 = tseries.get_laser_waist(iteration=iteration, pol=laser_polarization)
    ctau = tseries.get_ctau(iteration=iteration, pol=laser_polarization)

    current_time = tseries.current_t * u.second
    current_z = (u.clight * current_time).to_value("m")

    return current_z, a0, w0, ctau


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
    rho_path = pathlib.Path(job.ws) / "rhos"

    time_series = addons.LpaDiagnostics(h5_path, check_all_files=False)
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

        z_0, a_0, w_0, c_tau = get_scalar_diagnostics(tseries=time_series, iteration=it)

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
        laser_density_plot(
            tseries=time_series,
            iteration=it,
            rho_field_name="rho_electrons",
            save_path=rho_path,
            n_c=job.sp.n_c,
            E0=job.sp.E0,
        )

    # the field "rho" has (SI) units of charge/volume (Q/V), C/(m^3)
    # the initial density n_e has units of N/V, N = electron number
    # multiply by electron charge q_e to get (N e) / V
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
def plot_scalar_diags(job: Job) -> None:
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
    )
    fig.savefig(job.fn("a0.png"))
    pyplot.close(fig)

    fig, ax = pyplot.subplots(figsize=(10, 6))
    sliceplots.plot1d(
        ax=ax,
        v_axis=w_0,  # y-axis
        h_axis=z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$%s \;(\mu m)$" % "w_0",
    )
    fig.savefig(job.fn("w0.png"))
    pyplot.close(fig)

    fig, ax = pyplot.subplots(figsize=(10, 6))
    sliceplots.plot1d(
        ax=ax,
        v_axis=c_tau,  # y-axis
        h_axis=z_0,  # x-axis
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$c \tau \;(\mu m)$",
    )
    fig.savefig(job.fn("ctau.png"))
    pyplot.close(fig)


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
