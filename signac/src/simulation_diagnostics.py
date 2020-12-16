"""Module containing various simulation diagnostic tools."""
import pathlib
import numpy as np
from matplotlib import pyplot, colors, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import unyt as u


def centroid_plot(
    iteration,
    tseries,
    save_path=pathlib.Path.cwd(),
):
    """
    Plot a line through the centroids of each z-slice in the particle positions phase space.
    """
    fig, ax = pyplot.subplots(figsize=(7, 5))
    z, x = tseries.get_particle(
        ["z", "x"], species="bunch", iteration=iteration, plot=True
    )

    img = ax.get_images()[0]
    z_min, z_max, x_min, x_max = img.get_extent()
    hist_data = img.get_array()

    r, c = np.shape(hist_data)
    z_coords = np.linspace(z_min, z_max, c)
    x_coords = np.linspace(x_min, x_max, r)
    z_m, x_m = np.meshgrid(z_coords, x_coords)

    centroid = np.ma.average(x_m, weights=hist_data, axis=0)

    ax.plot(z_coords, centroid)

    filename = pathlib.Path(save_path) / f"centroid{iteration:06d}.png"
    fig.savefig(filename)
    pyplot.close(fig)


def density_plot(
    iteration,
    tseries,
    rho_field_name="rho_electrons",
    save_path=pathlib.Path.cwd(),
    n_e=1e21,  # 1/m^3
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
        rho / (u.elementary_charge.to_value("C") * n_e),
        extent=rho_info.imshow_extent * 1e6,  # conversion to microns
        origin="lower",
        norm=colors.SymLogNorm(linthresh=1e-4, linscale=0.15, base=10),
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
    cbar_rho.set_label(r"$n / n_\mathrm{e}$")

    ax.set_ylabel(r"${} \;(\mu m)$".format(rho_info.axes[0]))
    ax.set_xlabel(r"${} \;(\mu m)$".format(rho_info.axes[1]))

    current_time = (tseries.current_t * u.second).to("picosecond")
    ax.set_title(f"t = {current_time:.2f}")

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

    ux, uy, uz, w = tseries.get_particle(["ux", "uy", "uz", "w"], iteration=iteration)
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
    ids = [job.id for job in proj]
    job = proj.open_job(id=random.choice(ids))

    h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
    time_series = LpaDiagnostics(h5_path, check_all_files=True)

    it = random.choice(time_series.iterations.tolist())
    print(f"job {job.id}, iteration {it}")

    # _, _, _ = particle_energy_histogram(tseries=time_series, iteration=it)

    centroid_plot(iteration=it, tseries=time_series)
    density_plot(iteration=it, tseries=time_series)


if __name__ == "__main__":
    main()
