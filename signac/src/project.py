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
from typing import List, Optional, Tuple, Union

import numpy as np
import postproc.plotz as plotz
from flow import FlowProject
from opmd_viewer import OpenPMDTimeSeries
from scipy.constants import physical_constants
from scipy.signal import hilbert

logger = logging.getLogger(__name__)
# Usage: logger.info('message') or logger.warning('message')
logfname = "fbpic-minimal-project.log"

c_light = physical_constants[u"speed of light in vacuum"][0]
m_e = physical_constants[u"electron mass"][0]
q_e = physical_constants[u"elementary charge"][0]
mc2 = m_e * c_light ** 2 / (q_e * 1e6)


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
        ).communicate()[0].decode("utf-8")
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


############
# PLOTTING #
############


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


def get_a0(ts: OpenPMDTimeSeries,
           t: Optional[float] = None,
           it: Optional[int] = None,
           coord='x',
           m='all',
           lambda0=0.8e-6) -> Tuple[float, float, float, float]:
    """
    Compute z₀, a₀, w₀, cτ.

    :param ts: whole simulation time series
    :param t: time (in seconds) at which to obtain the data
    :param it: time step at which to obtain the data
    :param coord: which component of the field to extract
    :param m: 'all' for extracting the sum of all the modes
    :param lambda0: laser wavelength (meters)
    :return: z₀, a₀, w₀, cτ
    """
    # the direction along which to slice the data eg., 'x', 'y' or 'z'
    slicing_dir = 'y'
    # the angle of the plane of observation, with respect to the 'x' axis
    theta = 0.0

    # get E_x field in V/m
    electric_field_x, info_electric_field_x = ts.get_field(field='E', coord=coord,
                                                           t=t, iteration=it,
                                                           m=m, theta=theta, slicing_dir=slicing_dir)

    # normalized vector potential
    k0 = 2 * np.pi / lambda0
    electric_field_0 = m_e * c_light ** 2 * k0 / q_e

    a0 = electric_field_x / electric_field_0

    # get pulse envelope
    envelope = np.abs(hilbert(a0, axis=1))
    envelope_z = envelope[envelope.shape[0] // 2, :]

    a0_max = np.amax(envelope_z)

    # index of peak
    z_idx = np.argmax(envelope_z)
    # peak position
    z0 = info_electric_field_x.z[z_idx]

    # FWHM perpendicular size of beam, proportional to w0
    fwhm_a0_w0 = np.sum(np.greater_equal(envelope[:, z_idx], a0_max / 2)) * info_electric_field_x.dr

    # FWHM longitudinal size of the beam, proportional to ctau
    fwhm_a0_ctau = np.sum(np.greater_equal(envelope_z, a0_max / 2)) * info_electric_field_x.dz

    return z0, a0_max, fwhm_a0_w0, fwhm_a0_ctau


@Project.operation
@Project.pre.after(run_fbpic)
# @Project.post.isfile('rho.mp4')
@Project.post.isfile('diags.txt')
@Project.post.isfile("all_hist.txt")
@Project.post.isfile("all_bin_edges.txt")
def plot_rhos(job):
    base_dir = job.workspace()
    out_dir = "diags"
    h5_dir = "hdf5"
    h5_path: Union[bytes, str] = os.path.join(base_dir, out_dir, h5_dir)

    time_series: OpenPMDTimeSeries = OpenPMDTimeSeries(h5_path, check_all_files=False)
    number_of_iterations: int = time_series.iterations.size

    nbins = 349
    all_hist = np.empty(shape=(number_of_iterations, nbins), dtype=np.float64)
    all_bin_edges = np.empty(shape=(number_of_iterations, nbins + 1), dtype=np.float64)

    diags_file = open(job.fn("diags.txt"), "w")
    diags_file.write("# iteration, time[fs], z₀[μm], a₀, w₀[μm], cτ[μm]\n")

    # loop through all the iterations in the job's time series
    for idx, it in enumerate(time_series.iterations):
        time = it * job.sp.dt
        z_0, a_0, w_0, c_tau = get_a0(time_series, it=it)
        diags_file.write(f"{it:06d}, {time * 1e15:.3e}, {z_0 * 1e6:.3e}, {a_0:.3e}, {w_0 * 1e6:.3e}, {c_tau * 1e6:.3e}\n")

        energy_hist, bin_edges = particle_energy_histogram(
            tseries=time_series,
            it=it,
            energy_min=1.0,
            energy_max=350.0,
            nbins=nbins,
        )
        all_hist[idx, :] = energy_hist
        all_bin_edges[idx, :] = bin_edges

        # create folder "rhos"
        rho_path = os.path.join(base_dir, out_dir, "rhos")
        try:
            os.mkdir(rho_path)
        except OSError:
            logger.warning("Creation of the directory %s failed" % rho_path)
        else:
            logger.info("Successfully created the directory %s " % rho_path)

        # save "rho{it:06d}.png"
        field_snapshot(
            tseries=time_series,
            it=it,
            field_name="rho",
            normalization_factor=1.0 / (-q_e * job.sp.n_e),
            #chop=[40, -20, 15, -15],
            path=rho_path,
            zlabel=r"$n/n_e$",
            #vmin=0,
            #vmax=3,
        )

    diags_file.close()
    np.savetxt(job.fn("all_hist.txt"), all_hist,
               header="Each row contains an energy histogram, in order of increasing iteration number.")
    np.savetxt(job.fn("all_bin_edges.txt"), all_bin_edges,
               header="Each row contains an energy histogram bin edges, in order of increasing iteration number.")

    # run ffmpeg on all "rho{it:06d}.png" files and generate "rho.mp4"
    # use ``np.loadtxt`` to load "all_hist.txt" and "all_bin_edges.txt" and
    #   plot 2D energy-charge histogram
    # read "diags.txt" using ``pandas``, and
    #   plot a0 vs z0
    #   plot w0 vs z0
    #   plot ctau vs z0
    # plot electric field

# https://docs.signac.io/projects/core/en/latest/api.html#the-h5storemanager

# @Project.operation
# @directives(np=8) # only needed when submitting
# def generate_movie(job):
#     h5_files = glob(job.fn('diags/hdf5/*.h5'))
#     with multiprocessing.Pool(np=8) as pool:
#         pool.map(generate_png_files, h5files)
#     ffmpeg.input(job.fn(r'%016d.png'), framerate=24).output(job.fn('movie.mp4').run()


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
