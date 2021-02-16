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

import numpy as np
import sliceplots
from flow import FlowProject
from flow.environment import DefaultSlurmEnvironment
from matplotlib import pyplot
from openpmd_viewer.addons import LpaDiagnostics
import unyt as u
from util import du, ffmpeg_command, seconds_to_hms, shell_run, Timer
from simulation_diagnostics import (
    particle_energy_histogram,
    laser_density_plot,
    phase_space_plot,
)
from density_functions import plot_density_profile, make_gaussian_dens_func
from laser_profiles import make_flat_laser_profile, plot_laser_intensity
from render_lwfa_script import write_lwfa_script
from electron_spectrum import construct_electron_spectrum


logger = logging.getLogger(__name__)
log_file_name = "fbpic-project.log"


class OdinEnvironment(DefaultSlurmEnvironment):
    """Environment profile for the LGED cluster."""

    hostname_pattern = r".*\.ra5\.eli-np\.ro$"
    template = "odin.sh"
    mpi_cmd = "srun --mpi=pmi2"

    @classmethod
    def add_args(cls, parser):
        """Add arguments to parser.

        Parameters
        ----------
        parser : :class:`argparse.ArgumentParser`
            The argument parser where arguments will be added.

        """
        super().add_args(parser)
        parser.add_argument(
            "--partition",
            choices=("cpu", "gpu"),
            default="gpu",
            help="Specify the partition to submit to. (default=gpu)",
        )
        parser.add_argument(
            "-w",
            "--walltime",
            type=float,
            default=72,
            help="The wallclock time in hours. (default=72)",
        )


# TODO: add --mem-per-cpu argument


class Project(FlowProject):
    """
    Placeholder for ``FlowProject`` class.
    """


ex = Project.make_group(name="ex")

# TODO use job_util.py


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
    # FIXME what if list is empty?

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

    # FIXME replace LpaDiagnostics with a filesystem based approach which simply checks the files in /diags/hdf5
    time_series = LpaDiagnostics(h5_path)
    iterations: np.ndarray = time_series.iterations

    # estimate iteration array based on input parameters
    estimated_iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=int)

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
    iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=int)

    pngs = (f"{stem}{it:06d}.png" for it in iterations)

    return set(files) == set(pngs)


def are_rho_pngs(job):
    return are_pngs(job, "rho")


def are_phasespace_pngs(job):
    return are_pngs(job, "phasespace")


@ex
@Project.operation
@Project.post.true("macroparticle_count")
def add_macroparticle_count(job):
    count = job.sp.p_nt * job.sp.p_nr * job.sp.p_nz * job.sp.Nz * job.sp.Nr
    job.doc.setdefault("macroparticle_count", f"{count:.2e}")


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


# omp_num_threads=1 by default
# np=nranks * omp_num_threads by default
@ex.with_directives(
    dict(nranks=lambda job: job.sp.nranks, ngpu=lambda job: job.sp.nranks)
)
@Project.operation
@Project.pre.after(plot_laser)
@Project.post(fbpic_ran)
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
        n_order=job.sp.n_order,
        current_correction=job.sp.current_correction,
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
    job.doc.setdefault("runtime", str(seconds_to_hms(runtime)).split(".")[0])
    time_per_iteration = (runtime * u.second / job.sp.N_step).to(u.ms)
    job.doc.setdefault("time_per_iteration", f"{time_per_iteration:.0f}")

    # redirect stdout back and close "stdout.txt"
    sys.stdout = orig_stdout
    f.close()


@ex
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post(are_rho_pngs)
@Project.post(are_phasespace_pngs)
def save_pngs(job):
    """
    Loop through a whole simulation and, for *each ``fbpic`` iteration*:
    * save a snapshot of the plasma density field ``rho`` to {job_dir}/rhos/rho{it:06d}.png

    :param job: the job instance is a handle to the data of a unique statepoint

    Note: This operation takes about 30 mins on a single core.
    """
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    rho_path = pathlib.Path(job.ws) / "rhos"
    phasespace_path = pathlib.Path(job.ws) / "phasespaces"
    time_series = LpaDiagnostics(h5_path)

    for ts_it in time_series.iterations:
        laser_density_plot(
            iteration=ts_it,
            tseries=time_series,
            rho_field_name="rho_electrons",
            save_path=rho_path,
            n_c=job.sp.n_c,
            E0=job.sp.E0,
        )
        phase_space_plot(
            iteration=ts_it,
            tseries=time_series,
            uzmax=1.5e3,
            vmax=1.0e8,
            save_path=phasespace_path,
        )


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
@Project.post.isfile("final_histogram.png")
@Project.post.true("peak_charge")
@Project.post.true("peak_position")
def save_final_spectrum(job):
    """
    Save the histogram corresponding to the last iteration.
    Plot the electron spectrum corresponding to the last iteration.
    """
    es = construct_electron_spectrum(job)
    es.plot()
    es.savefig()

    job.doc["peak_position"] = int(es.hatch_window.peak_position)  # MeV
    job.doc["peak_charge"] = int(es.hatch_window.total_charge)  # pC


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
        species="electrons",
    )
    all_hist = np.empty(shape=(number_of_iterations, nrbins), dtype=np.float64)
    hist_edges = np.empty(shape=(nrbins + 1,), dtype=np.float64)

    # loop through all the iterations in the job's time series
    for idx, it in enumerate(time_series.iterations):
        # generate 1D energy histogram
        energy_hist, bin_edges, _ = particle_energy_histogram(
            tseries=time_series,
            iteration=it,
            species="electrons",
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
    iterations = np.arange(0, job.sp.N_step, job.sp.diag_period, dtype=int)
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


@ex
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post.true("disk_usage")
def store_disk_usage(job):
    usage = du(job.ws)
    job.doc.setdefault("disk_usage", usage)


# TODO (possibly) integrate `nvml.py` via `schedule` as background thread

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
