# coding: utf-8
# python analysis.py

import numpy as np
from scipy.constants import physical_constants

import h5py
import os
from multiprocessing import Pool

# import psutil
import itertools

from opmd_viewer import OpenPMDTimeSeries

import matplotlib.colors as colors

import fbpic_script  # fb-pic parameter file, located in the same folder
import postpic as pp
import plotz

import matplotlib as mpl

mpl.rc("lines", linewidth=2)
mpl.rc("axes", labelsize=20, titlesize=20, linewidth=1.5)
# mpl.rc('font', size=18, family="serif", serif="Computer Modern Roman")
mpl.rc("legend", fontsize=12)
# mpl.rc('text', usetex=True)
mpl.rc("xtick", **{"labelsize": 18, "major.size": 5.5, "major.width": 1.5})
mpl.rc("ytick", **{"labelsize": 18, "major.size": 5.5, "major.width": 1.5})
mpl.rc("savefig", dpi=300, frameon=False, transparent=True)


base_dir = "./"
out_dir = "diags"
h5_dir = "hdf5"
h5_path = os.path.join(base_dir, out_dir, h5_dir)

timestep = 100  # 100
f_name = os.path.join(h5_path, "data{:08d}.h5".format(timestep))
f = h5py.File(f_name, "r")

bpath = f["/data/{}".format(timestep)]
# t = bpath.attrs["time"] * bpath.attrs["timeUnitSI"] * 1e15 # time in fs
dt = bpath.attrs["dt"] * bpath.attrs["timeUnitSI"] * 1e15  # time step in fs

q_e = physical_constants["elementary charge"][0]  # C, positive
c = physical_constants["speed of light in vacuum"][0]  # m/s
m_e = physical_constants["electron mass"][0]  #  Kg
mc2 = m_e * c ** 2 / (q_e * 1e6)  # electron rest energy in MeV: 0.511
tstep_to_pos = (c * 1e-9) * dt  # conversion factor in mu

n_e = fbpic_script.n_e  # initial electron density in electrons x m^{-3}
n_c = 1.75e27  # critical electron density in electrons x m^{-3}
fields = "/data/{}/fields".format(timestep)
handler = f[fields]
unit_factor = handler["rho"].attrs["unitSI"] / (-q_e * n_e)  # m^3/C
unit_factor_crit = handler["rho"].attrs["unitSI"] / (-q_e * n_c)  # m^3/C

ts_circ = OpenPMDTimeSeries(h5_path, check_all_files=False)


def field_snapshot(
    tseries,
    iteration,
    field,
    norm_factor,
    coord=None,
    m="all",
    theta=0.0,
    chop=[0.0, 0.0, 0.0, 0.0],
    **kwargs
):
    F, info = tseries.get_field(
        field=field, coord=coord, iteration=iteration, m=m, theta=theta
    )
    #
    F *= norm_factor
    plot = plotz.Plot2D(
        F,
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
        **kwargs
    )
    #
    # print(plot)
    plot.canvas.print_figure("{}{:06d}.png".format(field, iteration))


def particle_histogram(
    tseries,
    iteration,
    Emin=1.0,
    Emax=300.0,
    nbins=100,
    mc2=0.5109989462686102,
    q_e=1.6021766208e-19,
):
    dE = (Emax - Emin) / nbins
    E_bins = np.linspace(Emin, Emax, nbins + 1)
    ux, uy, uz, w = tseries.get_particle(["ux", "uy", "uz", "w"], iteration=iteration)
    E = mc2 * np.sqrt(1 + ux ** 2 + uy ** 2 + uz ** 2)  # Energy in MeV
    Q_bins, edges = np.histogram(
        E, E_bins, weights=q_e * 1e12 / dE * w
    )  # weights in pC/MeV
    return Q_bins.tolist(), edges.tolist()


def apply_func(iteration):
    field_snapshot(
        tseries=ts_circ,
        iteration=iteration,
        field="rho",
        norm_factor=unit_factor,
        chop=[40, -20, 15, -15],
        zlabel=r"$n/n_e$",
        vmin=0,
        vmax=3,
    )
    time_fs = iteration * dt
    z0, a0, w0, ctau = pp.get_a0(ts_circ, iteration=iteration)  # SI
    Q, edg = particle_histogram(
        tseries=ts_circ, iteration=iteration, Emin=1.0, Emax=350.0, nbins=349
    )
    return (iteration, time_fs, z0 * 1e6, a0, w0 * 1e6, ctau * 1e6, Q, edg)  # microns


if __name__ == "__main__":
    print(ts_circ.iterations[-1], tstep_to_pos)

    pool = Pool()
    diags = pool.map(apply_func, ts_circ.iterations.tolist())
    #
    extract = lambda i, lst_of_tuples: np.array([tple[i] for tple in lst_of_tuples])
    z0 = extract(2, diags)
    a0 = extract(3, diags)
    w0 = extract(4, diags)
    ctau = extract(5, diags)
    Q = extract(6, diags)
    # edg = extract(7, diags)

    plot = plotz.Plot1D(
        a0,
        z0,
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$%s$" % "a_0",
        xlim=[0, 900],
        ylim=[0, 10],
        figsize=(10, 6),
        color="red",
    )
    plot.canvas.print_figure("a0.png")  #
    #
    plot = plotz.Plot1D(
        w0,
        z0,
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"$%s \;(\mu m)$" % "w_0",
        figsize=(10, 6),
    )
    plot.canvas.print_figure("w0.png")  #
    #
    plot = plotz.Plot1D(
        ctau, z0, xlabel=r"$%s \;(\mu m)$" % "z", ylabel=r"$c \tau \;(\mu m)$"
    )
    plot.canvas.print_figure("ctau.png")  #

    z_min = ts_circ.iterations[0] * tstep_to_pos + 70  # mu
    z_max = (
        ts_circ.iterations[-1] * tstep_to_pos + 70
    )  # offset by 70 because left margin is at -70
    z_35100 = ts_circ.iterations[35100 // 100] * tstep_to_pos + 70
    #
    h_axis = np.linspace(z_min, z_max, ts_circ.iterations.size - 1)
    v_axis = np.linspace(1.0, 350.0, 349)
    #
    hist2d = plotz.Plot2D(
        Q.T,
        h_axis,
        v_axis,
        xlabel=r"$%s \;(\mu m)$" % "z",
        ylabel=r"E (MeV)",
        zlabel=r"dQ/dE (pC/MeV)",
        # hslice_val=0,
        vslice_val=z_35100,
        # hslice_opts={'color': 'firebrick', 'lw' : 0.5, 'ls':'-'},
        # vslice_opts={'color': 'blue', 'ls': '-'},
        extent=(z_min, z_max, 1.0, 350.0),
        vmin=0,
        vmax=10,
        # vmin=1e-3, vmax=1e2,
        # norm = colors.LogNorm(),
    )
    hist2d.canvas.print_figure("hist2d.png")
    #
    # Q_bins, edges = particle_histogram(tseries=ts_circ, iteration=35100, Emin=1., Emax=350., nbins=349,
    #                     mc2=0.5109989462686102, q_e=1.6021766208e-19)
    # #
    # left, right = edges[:-1], edges[1:]
    # #
    # X = np.array([left, right]).T.flatten()
    # Y = np.array([Q_bins, Q_bins]).T.flatten()
    # #
    # hist = plotz.Plot1D(Y, X, xlabel=r'E (MeV)', ylabel=r'dQ/dE (pC/MeV)',
    #                      figsize=(10, 6), xlim=[1., 350.], ylim=[0, 10],
    #                      text='iteration = {}'.format(35100))
    # hist.canvas.print_figure('histogram_35100.png')

    # field_snapshot(tseries=ts_circ, iteration=35100, field='rho', norm_factor=unit_factor_crit,
    # chop = [40, -20, 15, -15],
    # zlabel=r'$n_e/n_c$', vmin=0., vmax=0.1,
    # hslice_val=0,)# vslice_val=840.5)
    field_snapshot(
        tseries=ts_circ,
        iteration=35100,
        field="E",
        coord="x",
        norm_factor=1 / pp.E0(),
        chop=[40, -20, 15, -15],
        zlabel=r"$E_x / E_0$",
        vmin=-8,
        vmax=8,
        hslice_val=0.0,
    )


# 100 msec / image, 10 fps
# ffmpeg -framerate 10 -i rho%05d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p rho.mp4
# ffmpeg -framerate 10 -pattern_type glob -i 'rho*.png' -c:v libx264 -vf fps=25 -pix_fmt yuv420p rho.mp4
