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
import pathlib
import sys
from multiprocessing import Pool
from functools import partial

import numpy as np
from flow import FlowProject, directives
from flow.environment import DefaultSlurmEnvironment
from openpmd_viewer.addons import LpaDiagnostics
import unyt as u
from prepic import Plasma
from util import ffmpeg_command, shell_run
from simulation_diagnostics import density_plot, centroid_plot
from density_functions import plot_density_profile, make_experimental_dens_func
from electron_bunch import (
    shade_bunch,
    write_bunch_openpmd,
    plot_bunch_energy_histogram,
    bunch_openpmd_to_dataframe,
    bunch_density,
    bunch_centroid_plot,
)


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

    time_series = LpaDiagnostics(h5_path, check_all_files=True)
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


def are_centroid_pngs(job):
    return are_pngs(job, "centroid")


@ex
@Project.operation
@Project.pre.isfile("exp_4deg.txt")
@Project.post.isfile("bunch/data_00000.h5")
def bunch_txt_to_opmd(job):
    write_bunch_openpmd(
        bunch_txt=job.fn("exp_4deg.txt"),
        bunch_charge=job.sp.bunch_charge,
        outdir=pathlib.Path(job.ws),
    )


@ex
@Project.operation
@Project.pre.after(bunch_txt_to_opmd)
@Project.post.isfile("bunch/energy_histogram.png")
def bunch_histogram(job):
    bunch_dir = pathlib.Path(job.ws) / "bunch"
    plot_bunch_energy_histogram(
        opmd_dir=bunch_dir,
        export_dir=bunch_dir,
    )


@ex
@Project.operation
@Project.pre.after(bunch_txt_to_opmd)
@Project.post.isfile("bunch/bunch_z_um_x_um.png")
def plot_initial_bunch(job):
    df = bunch_openpmd_to_dataframe(workdir=pathlib.Path(job.ws))
    shade_bunch(df, "z_um", "x_um", export_path=pathlib.Path(job.ws) / "bunch")


@ex
@Project.operation
@Project.pre.after(bunch_txt_to_opmd)
@Project.post.true("n_bunch")
@Project.post.true("λ_bunch")
def estimate_bunch_density(job):
    df = bunch_openpmd_to_dataframe(workdir=pathlib.Path(job.ws))
    n_bunch, sphere, _ = bunch_density(df)
    job.doc.setdefault(
        "n_bunch", float("{:.2e}".format(n_bunch.to_value(u.meter ** (-3))))
    )

    plasma = Plasma(n_pe=job.doc.n_bunch * u.meter ** (-3))
    job.doc.setdefault("λ_bunch", f"{plasma.λp:.1f}")


@ex
@Project.operation
@Project.pre.after(bunch_txt_to_opmd)
@Project.post.isfile("bunch/centroid000000.png")
def plot_bunch_centroid(job):
    bunch_centroid_plot(pathlib.Path(job.ws) / "bunch")


@ex
@Project.operation
@Project.pre.isfile("density_1_inlet_spacers.txt")
@Project.post.isfile("initial_density_profile.png")
def plot_initial_density_profile(job):
    """Plot the initial plasma density profile."""
    plot_density_profile(
        make_experimental_dens_func, job.fn("initial_density_profile.png"), job
    )


@ex.with_directives(directives=dict(ngpu=1))
@directives(ngpu=1)
@Project.operation
@Project.pre.after(bunch_txt_to_opmd)
@Project.post(fbpic_ran)
def run_fbpic(job):
    """
    This ``signac-flow`` operation runs a ``fbpic`` simulation.

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    from fbpic.main import Simulation
    from fbpic.openpmd_diag import (
        FieldDiagnostic,
        ParticleDiagnostic,
        ParticleChargeDensityDiagnostic,
    )
    from fbpic.lpa_utils.bunch import add_particle_bunch_openPMD

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
            "r": "reflective",
        },  # 'r': 'open' can also be used (more expensive)
        n_order=-1,
        use_cuda=True,
        verbose_level=2,
    )
    # Add the plasma electron and plasma ions
    plasma_elec = sim.add_new_species(
        q=u.electron_charge.to_value("C"),
        m=u.electron_mass.to_value("kg"),
        n=job.sp.n_e,
        dens_func=make_experimental_dens_func(job),
        p_zmin=job.sp.p_zmin,
        p_zmax=job.sp.p_zmax,
        p_rmax=job.sp.p_rmax,
        p_nz=job.sp.p_nz,
        p_nr=job.sp.p_nr,
        p_nt=job.sp.p_nt,
    )
    # The electron bunch
    # particles beam from txt file
    bunch = add_particle_bunch_openPMD(
        sim=sim,
        q=u.electron_charge.to_value("C"),
        m=u.electron_mass.to_value("kg"),
        ts_path=pathlib.Path(job.ws) / "bunch",
        species="bunch",
        iteration=0,
    )

    # Configure the moving window
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
            species={"electrons": plasma_elec, "bunch": bunch},
            comm=sim.comm,
            write_dir=write_dir,
        ),
        ParticleChargeDensityDiagnostic(
            period=job.sp.diag_period,
            sim=sim,
            species={"electrons": plasma_elec, "bunch": bunch},
            write_dir=write_dir,
        ),
    ]
    # set deterministic random seed
    np.random.seed(0)

    # Run the simulation
    sim.step(job.sp.N_step, show_progress=False)

    # redirect stdout back and close "stdout.txt"
    sys.stdout = orig_stdout
    f.close()


@ex.with_directives(directives=dict(np=3))
@directives(np=3)
@Project.operation
@Project.pre.after(run_fbpic)
@Project.post(are_rho_pngs)
@Project.post(are_centroid_pngs)
def save_pngs(job):
    """
    Loop through a whole simulation and, for *each ``fbpic`` iteration*:
    * save a snapshot of the plasma density field ``rho`` to {job_dir}/rhos/rho{it:06d}.png

    :param job: the job instance is a handle to the data of a unique statepoint
    """
    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    rho_path = pathlib.Path(job.ws) / "rhos"
    centroid_path = pathlib.Path(job.ws) / "centroids"
    time_series = LpaDiagnostics(h5_path, check_all_files=True)

    it_density_plot = partial(
        density_plot,
        tseries=time_series,
        rho_field_name="rho_bunch",
        save_path=rho_path,
        n_e=job.sp.n_e,
        n_bunch=job.doc.n_bunch,
    )
    it_centroid_plot = partial(
        centroid_plot, tseries=time_series, save_path=centroid_path
    )

    with Pool(3) as pool:
        pool.map(it_centroid_plot, time_series.iterations.tolist())
        pool.map(it_density_plot, time_series.iterations.tolist())


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
@Project.post.isfile("centroid.mp4")
def generate_centroid_movie(job):
    generate_movie(job, stem="centroid")


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
