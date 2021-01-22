"""Module containing various simulation diagnostic tools."""
import pathlib
import numpy as np
from matplotlib import pyplot, colors, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import unyt as u
from scipy import integrate
from scipy import interpolate
from matplotlib.gridspec import GridSpec


def get_cubic_spline(x, y, smoothing_factor=1e-7):
    cs = interpolate.UnivariateSpline(x, y)
    cs.set_smoothing_factor(smoothing_factor)

    return cs


def centroid_plot(
    iteration,
    tseries,
    save_path=pathlib.Path.cwd(),
    smoothing_factor=1e-7,
    save_fig=True,
    fn_postfix=None,
):
    """
    Plot a line through the centroids of each z-slice in the particle positions phase space.
    """
    fig, ax = pyplot.subplots(figsize=(7, 5))
    z, x = tseries.get_particle(
        ["z", "x"],
        species="bunch",
        iteration=iteration,
        plot=True,
        use_field_mesh=False,
        nbins=2 ** 8 + 1,
    )

    img = ax.get_images()[0]
    z_min, z_max, x_min, x_max = img.get_extent()
    hist_data = img.get_array()

    r, c = np.shape(hist_data)
    z_coords = np.linspace(z_min, z_max, c)
    x_coords = np.linspace(x_min, x_max, r)
    z_m, x_m = np.meshgrid(z_coords, x_coords)

    centroid = np.ma.average(x_m, weights=hist_data, axis=0)
    x_avg = np.mean(centroid)

    if save_fig:
        ax.plot(z_coords, centroid)

        cs = get_cubic_spline(z_coords, centroid, smoothing_factor=smoothing_factor)
        ax.plot(z_coords, cs(z_coords), label="spline")

        ax.legend()
        ax.hlines(
            y=x_avg,
            xmin=z_coords[0],
            xmax=z_coords[-1],
            linestyle="dashed",
            color="0.75",
        )
        if fn_postfix is not None:
            filename = (
                pathlib.Path(save_path) / f"centroid{fn_postfix}.png"
            )
        else:
            filename = pathlib.Path(save_path) / f"centroid{iteration:06d}.png"

        fig.savefig(filename)

    pyplot.close(fig)

    return z_coords, centroid, x_avg


def compute_bending_energy(iteration, tseries, smoothing_factor=1e-7):
    x, y, _ = centroid_plot(iteration=iteration, tseries=tseries, save_fig=False)

    spline = get_cubic_spline(x, y, smoothing_factor=smoothing_factor)
    y2 = spline.derivative(2)(x)

    bending_energy = 1 / 2 * integrate.romb(y2 ** 2)

    return bending_energy  # m^-1


def plot_spline_derivatives(iteration, tseries, smoothing_factor=1e-7):
    x, y, _ = centroid_plot(
        iteration=iteration,
        tseries=tseries,
        smoothing_factor=smoothing_factor,
        save_fig=False,
    )

    spline = get_cubic_spline(x, y, smoothing_factor=smoothing_factor)

    bending_energy = compute_bending_energy(
        iteration=iteration, tseries=tseries, smoothing_factor=smoothing_factor
    )

    fig = pyplot.figure()
    G = GridSpec(4, 1, figure=fig)
    ax_top = fig.add_subplot(G[0, :])
    ax_middle_high = fig.add_subplot(G[1, :])
    ax_middle_low = fig.add_subplot(G[2, :])
    ax_bottom = fig.add_subplot(G[3, :])

    ax_top.plot(x, spline(x), color="C1")
    ax_middle_high.plot(x, spline.derivative(1)(x), color="C2")
    ax_middle_low.plot(x, spline.derivative(2)(x), color="C3")
    ax_bottom.plot(x, spline.derivative(2)(x) ** 2, color="C3")
    ax_bottom.fill_between(x, spline.derivative(2)(x) ** 2)

    fig.suptitle(
        r"$W = \frac{1}{2} \int{\left(\frac{\mathrm{d}^2 x}{\mathrm{d}z^2}\right)^2} \mathrm{d}z$ = %.3e m${}^{-1}$"
        % bending_energy
    )

    ax_top.set_ylabel(r"$x$ (m)")
    ax_middle_high.set_ylabel(r"$\frac{\mathrm{d}x}{\mathrm{d}z}$")
    ax_middle_low.set_ylabel(r"$\frac{\mathrm{d}^2 x}{\mathrm{d}z^2}$ (m${}^{-1}$)")
    ax_bottom.set_ylabel(
        r"$\left(\frac{\mathrm{d}^2 x}{\mathrm{d}z^2}\right)^2$ (m${}^{-2}$)"
    )
    ax_bottom.set_xlabel(r"$z$ (m)")

    for ax in ax_top, ax_middle_high, ax_middle_low:
        ax.set_xticklabels([])

    for ax in ax_top, ax_middle_high, ax_middle_low, ax_bottom:
        ax.grid()

    fig.savefig(f"derivatives{iteration:06d}.png", bbox_inches="tight")
    pyplot.close(fig)


def density_plot(
    iteration,
    tseries,
    rho_field_name="rho_electrons",
    save_path=pathlib.Path.cwd(),
    n_e=1e21,  # 1/m^3
    n_bunch=3.6e20,  # 1/m^3
):
    """
    Plot the electron density.
    """

    rho, rho_info = tseries.get_field(
        field=rho_field_name,
        iteration=iteration,
    )
    # the field "rho" has (SI) units of charge/volume (Q/V), C/(m^3)
    # the initial density n_e has units of N/V, N = electron number
    # multiply by electron charge q_e to get (N e) / V
    # so we get Q / N e, which is C/C, i.e. dimensionless

    fig, ax = pyplot.subplots(figsize=(10, 3))

    im_rho = ax.imshow(
        rho / (u.electron_charge.to_value("C") * n_bunch),
        extent=rho_info.imshow_extent * 1e6,  # conversion to microns
        origin="lower",
        norm=colors.Normalize(vmin=-1.0, vmax=5.0),
        # norm=colors.SymLogNorm(linthresh=1e-4, linscale=0.15, base=10),
        # norm=colors.LogNorm(vmin=1e-6, vmax=10),
        cmap=cm.get_cmap("cividis"),
    )
    cbaxes_rho = inset_axes(
        ax,
        width="3%",  # width = 10% of parent_bbox width
        height="100%",  # height : 50%
        loc=2,
        bbox_to_anchor=(1.01, 0.0, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )
    cbar_rho = fig.colorbar(
        mappable=im_rho, orientation="vertical", ticklocation="right", cax=cbaxes_rho
    )
    cbar_rho.set_label(r"$n / n_\mathrm{bunch}$")

    ax.set_ylabel(r"${} \;(\mu m)$".format(rho_info.axes[0]))
    ax.set_xlabel(r"${} \;(\mu m)$".format(rho_info.axes[1]))

    current_time = (tseries.current_t * u.second).to("picosecond")
    ax.set_title(f"t = {current_time:.2f}")

    filename = save_path / f"rho{iteration:06d}.png"

    fig.subplots_adjust(right=0.85)
    fig.savefig(filename, dpi=192)
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
    from openpmd_viewer.addons import LpaDiagnostics
    import signac

    proj = signac.get_project(search=False)
    job = proj.open_job(id="2ecf4ec87c9f1388ab56b4bee9428859")

    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path, check_all_files=True)

    it = 20196
    print(f"job {job.id}, iteration {it}")

    _, _, x_avg = centroid_plot(
        iteration=it,
        tseries=time_series,
        smoothing_factor=1e-8,
    )
    plot_spline_derivatives(iteration=it, tseries=time_series, smoothing_factor=1e-8)
    bending_energy = compute_bending_energy(
        iteration=it, tseries=time_series, smoothing_factor=1e-8
    )

    print(f"<x> = {x_avg:.3e} m")
    print(f"W = {bending_energy:.3e} / m")


if __name__ == "__main__":
    main()
