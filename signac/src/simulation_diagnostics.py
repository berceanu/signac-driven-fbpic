"""Module containing various simulation diagnostic tools."""
import pathlib
from copy import copy
import numpy as np
from matplotlib import pyplot, colors, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import unyt as u
import colorcet as cc


def phase_space_hist(iteration, tseries, *, nbuz=200, nbz=200, uzmin=40.0, uzmax=None):
    """
    Compute the phase-space uz vs z 2D histogram,
    with given number of bins.
    """
    if uzmax is None:
        hist_range = None
    else:
        hist_range = [None, [uzmin, uzmax]]

    z, uz, w = tseries.get_particle(
        var_list=["z", "uz", "w"],
        species="electrons",
        iteration=iteration,
        select={"uz": [uzmin, uzmax]},
    )
    # convert from m to um
    z_um = z * 1e6

    H, zedges, uzedges = np.histogram2d(
        z_um,
        uz,
        bins=(nbz, nbuz),
        weights=w,  # convert to count of real electrons
        range=hist_range,
    )
    Z, UZ = np.meshgrid(zedges, uzedges)

    H = H.T  # Let each row list bins with common x range.

    return H, Z, UZ


def phase_space_plot(
    iteration,
    tseries,
    uzmax=None,
    vmax=None,
    save_path=pathlib.Path.cwd(),
):
    """
    Plot the longitudinal (uz v z) electron phase space.
    """

    H, Z, UZ = phase_space_hist(iteration, tseries, uzmax=uzmax)

    fig, ax = pyplot.subplots(figsize=(7, 5))

    current_time = (tseries.current_t * u.second).to(u.picosecond)
    ax.set_title(f"t = {current_time:.2f} (iteration {iteration:,g})")

    img = ax.pcolormesh(
        Z,
        UZ,
        H,
        cmap=cm.get_cmap("plasma"),
        norm=colors.SymLogNorm(linthresh=1, vmin=0.0, vmax=vmax, base=10),
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.02)
    cbar = ax.figure.colorbar(
        img,
        cax=cax,
    )
    cbar.set_label(r"number of electrons")

    ax.set_xlabel(r"$z$ ($\mathrm{\mu m}$)")
    ax.set_ylabel(r"$u_z$ ($m_e c$)")

    filename = pathlib.Path(save_path) / f"phasespace{iteration:06d}.png"
    fig.savefig(filename)
    pyplot.close(fig)


def laser_density_plot(
    iteration,
    tseries,
    rho_field_name="rho_electrons",
    laser_polarization="x",
    save_path=pathlib.Path.cwd(),
    n_c=1.7419595910637713e27,  # 1/m^3
    E0=4013376052599.5396,  # V/m
):
    """
    Plot on the same figure the laser pulse envelope and the electron density.
    """

    laser_cmap = copy(cc.m_kr)
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
    # the field "rho" has (SI) units of charge/volume (Q/V), C/(m^3)
    # the initial density n_e has units of N/V, N = electron number
    # multiply by electron charge q_e to get (N e) / V
    # so we get Q / N e, which is C/C, i.e. dimensionless
    # Note: one can also normalize by the critical density n_c

    fig, ax = pyplot.subplots(figsize=(10, 6))

    R = rho / (u.electron_charge.to_value("C") * n_c)
    # print(f"{R.min():.2f}, {R.max():.2f}")

    E = envelope / E0
    # print(f"{E.min():.2f}, {E.max():.2f}")

    im_rho = ax.imshow(
        R,
        extent=rho_info.imshow_extent * 1e6,  # conversion to microns
        origin="lower",
        norm=colors.SymLogNorm(
            vmin=-0.3, vmax=6.0, linthresh=1.0e-3, linscale=0.15, base=10
        ),
        cmap=cm.get_cmap("cividis"),
    )
    im_envelope = ax.imshow(
        E,
        extent=env_info.imshow_extent * 1e6,
        origin="lower",
        norm=colors.Normalize(vmin=0.0, vmax=5.0),
        cmap=laser_cmap,
    )
    im_envelope.set_clim(vmin=1.0)

    # plot longitudinal field
    ax.plot(e_z_of_z_info.z * 1e6, e_z_of_z / E0 * 15 - 15, color="0.75")
    ax.axhline(-15, color="0.65", ls="-.")

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

    ax.set_ylabel(r"${} \;(\mu m)$".format(rho_info.axes[0]))
    ax.set_xlabel(r"${} \;(\mu m)$".format(rho_info.axes[1]))

    current_time = (tseries.current_t * u.second).to(u.picosecond)
    ax.set_title(f"t = {current_time:.2f} (iteration {iteration:,g})")

    filename = save_path / f"rho{iteration:06d}.png"

    fig.subplots_adjust(right=0.85)
    fig.savefig(filename)
    pyplot.close(fig)


def particle_energy_histogram(
    tseries,
    iteration: int,
    energy_min=1,
    energy_max=500,
    delta_energy=1,
    cutoff=35,  # CHANGEME
    species=None,
):
    """
    Compute the weighted particle energy histogram from ``tseries`` at step ``iteration``.

    :param tseries: whole simulation time series
    :param iteration: time step in the simulation
    :param energy_min: lower energy threshold (MeV)
    :param energy_max: upper energy threshold (MeV)
    :param delta_energy: size of each energy bin (MeV)
    :param cutoff: upper threshold for the histogram, in pC / MeV
    :return: histogram values and bin edges
    """
    nbins = (energy_max - energy_min) // delta_energy
    energy_bins = np.linspace(start=energy_min, stop=energy_max, num=nbins + 1)

    ux, uy, uz, w = tseries.get_particle(
        ["ux", "uy", "uz", "w"], iteration=iteration, species=species
    )

    if (np.ndim(w) == 1) and (w.size == 1):
        fill_value = w.item(0)
        w = np.full_like(ux, fill_value)

    energy = (u.electron_mass * u.clight ** 2).to_value("MeV") * np.sqrt(
        1 + ux ** 2 + uy ** 2 + uz ** 2
    )

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


def main():
    import random
    from openpmd_viewer.addons import LpaDiagnostics
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    job = next(iter(proj))

    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path)

    it = random.choice(time_series.iterations.tolist())
    print(f"job {job.id}, iteration {it}")

    # compute 1D histogram
    energy_hist, bin_edges, _ = particle_energy_histogram(
        tseries=time_series,
        iteration=it,
        species="electrons",
        cutoff=np.inf,  # no cutoff
    )
    np.savez("final_histogram.npz", edges=bin_edges, counts=energy_hist)

    laser_density_plot(iteration=it, tseries=time_series)
    phase_space_plot(iteration=it, tseries=time_series)


if __name__ == "__main__":
    main()
