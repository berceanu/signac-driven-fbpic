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

import numpy as np
from flow import FlowProject
from scipy.constants import c, physical_constants

from opmd_viewer import OpenPMDTimeSeries
from typing import List, Optional, Tuple
import postproc.plotz as plotz

logger = logging.getLogger(__name__)
# Usage: logger.info('message') or logger.warning('message')
logfname = "fbpic-minimal-project.log"


#####################
# UTILITY FUNCTIONS #
#####################

# https://stackoverflow.com/questions/3346430/what-is-the-most-efficient-way-to-get-first-and-last-line-of-a-text-file/18603065#18603065

def read_last_line(filename):
    with open(filename, "rb") as f:
        _ = f.readline()  # Read the first line.
        f.seek(-2, os.SEEK_END)  # Jump to the second last byte.
        while f.read(1) != b"\n":  # Until EOL is found...
            # ...jump back the read byte plus one more.
            f.seek(-2, os.SEEK_CUR)
        last = f.readline()  # Read last line.
    return last


def isemptyfile(filename):
    return lambda job: job.isfile(filename) and os.stat(job.fn(filename)).st_size == 0


def file_contains(filename, text):
    """Checks if `filename` contains `text`."""
    return (
        lambda job: job.isfile(filename) and text in open(job.fn(filename), "r").read()
    )


def arefiles(filenames):
    """Check if all `filenames` are in `job` folder."""
    return lambda job: all(job.isfile(fn) for fn in filenames)


def sh(*cmd, **kwargs):
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


class Project(FlowProject):
    pass


####################
# OPERATION LABELS #
####################


def data_to_ascii(fname):
    pre, _ = os.path.splitext(fname)
    outfname = pre + ".sed"

    shutil.copy(fname, outfname)

    # replace '\r' by '\n'
    sh(rf"sed -i 's/\x0d/\x0a/g' {outfname}", shell=True)

    # replace '\u2588'(█) by '-'
    sh(rf"sed -i 's/\xe2\x96\x88/-/g' {outfname}", shell=True)

    # remove '<ESC>[K'
    sh(
        rf"sed -i -e $(echo -e 's/\033\[K//g') {outfname}",
        shell=True,
        executable="/bin/bash",
    )
    return outfname


@Project.label
def progress(job):
    fn = "stdout.txt"
    if job.isfile(fn):
        outfn = data_to_ascii(job.fn(fn))
        # last_line = data.split("\033[K\r")[-1]
        # last_line = sh(f"tail -n 1 {job.fn(fn)}", shell=True)
        last_line = read_last_line(outfn).decode("UTF-8")
        if last_line.startswith("|"):
            percentage = last_line  # .split()[-4]
        else:  # already finished the matrix calculation
            percentage = "100.00"
    else:  # didn't yet start the matrix calculation
        percentage = "0.00"

    return percentage


###############################
# INITIALIZE & RUN SIMULATION #
###############################


@Project.operation
@Project.post(lambda job: job.document.ran_job is True)
def run_fbpic(job):
    from fbpic.lpa_utils.laser import add_laser
    from fbpic.main import Simulation
    from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic

    # The density profile
    def dens_func(z):
        """Returns relative density at position z and r"""
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

    # redirect stdout
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
    sim.set_moving_window(v=c)

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

    # redirect back
    sys.stdout = orig_stdout
    f.close()

    # convert stdout file from data to ascii
    # f = open(job.fn("stdout.txt"), "r")
    # contents = f.read()
    # contents.replace()

    # remove '<ESC>[K'
    # subprocess.call(["sed -i -e $(echo -e 's/\033\[K//g') stdout.txt"], shell=True)
    # replace '\r' by '\n'
    # subprocess.call(["sed -i 's/\x0d/\x0a/g' stdout.txt"], shell=True)
    # replace '\u2588'(█) by '-'
    # subprocess.call(["sed -i 's/\xe2\x96\x88/-/g' stdout.txt"], shell=True)

    job.document.ran_job = True


# @Project.operation
# @Project.pre.after(run_fbpic)
# @Project.post.isfile('')

############
# PLOTTING #
############

q_e = physical_constants["elementary charge"][0]
m_e = physical_constants["electron mass"][0]
mc2 = m_e * c ** 2 / (q_e * 1e6)


def particle_histogram(
    tseries: OpenPMDTimeSeries,
    iteration: int,
    energy_min=1.0,
    energy_max=300.0,
    nbins=100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the weighted particle energy histogram from ``tseries`` at step ``iteration``.

    :param tseries: whole simulation time series
    :param iteration: time step in the simulation
    :param energy_min: lower energy threshold
    :param energy_max: upper energy threshold
    :param nbins: number of bins
    :return: histogram values and bin edges
    """
    delta_energy = (energy_max - energy_min) / nbins
    energy_bins = np.linspace(start=energy_min, stop=energy_max, num=nbins + 1)

    ux, uy, uz, w = tseries.get_particle(["ux", "uy", "uz", "w"], iteration=iteration)
    energy = mc2 * np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)

    q_bins, edges = np.histogram(
        energy, bins=energy_bins, weights=q_e * 1e12 / delta_energy * w
    )

    return q_bins, edges


def field_snapshot(
    tseries: OpenPMDTimeSeries,
    iteration: int,
    field_name: str,
    norm_factor: float,
    coord: Optional[str] = None,
    m="all",
    theta=0.0,
    chop: Optional[List[float]] = None,
    **kwargs,
) -> None:
    """
    Plot the ``field_name`` field from ``tseries`` at step ``iteration``.

    :param tseries: whole simulation time series
    :param iteration: time step in the simulation
    :param field_name: which field to extract, eg. 'rho', 'E', 'B' or 'J'
    :param norm_factor: normalization factor for the extracted field
    :param coord: which component of the field to extract, eg. 'r', 't' or 'z'
    :param m: 'all' for extracting the sum of all the modes
    :param theta: the angle of the plane of observation, with respect to the x axis
    :param chop: adjusting extent of simulation box plot
    :param kwargs: extra plotting arguments, eg. labels, data limits etc.
    :return: saves field plot image to disk
    """
    if chop is None:
        chop = [0.0, 0.0, 0.0, 0.0]
    field, info = tseries.get_field(
        field=field_name, coord=coord, iteration=iteration, m=m, theta=theta
    )

    field *= norm_factor
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
        text="iteration {}".format(iteration),
        **kwargs,
    )

    plot.canvas.print_figure("{}{:06d}.png".format(field_name, iteration))


if __name__ == "__main__":
    logging.basicConfig(
        filename=logfname,
        format="%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info("==RUN STARTED==")
    Project().main()
    logger.info("==RUN FINISHED==")
