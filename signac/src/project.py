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
from scipy.constants import c  # m/s

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
@Project.post(lambda job: job.document.ran_job == True)
def run_fbpic(job):
    from fbpic.lpa_utils.laser import add_laser
    from fbpic.main import Simulation
    from fbpic.openpmd_diag import FieldDiagnostic, ParticleDiagnostic    
    
    # The density profile
    def dens_func(z, r):
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
