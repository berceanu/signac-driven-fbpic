import pathlib
import numpy as np
import proplot as pplt
from scipy.constants import golden
import unyt as u
from openpmd_viewer.addons import LpaDiagnostics
from simulation_diagnostics import (
    centroid_plot,
    compute_bending_energy,
)
from util import ffmpeg_command, shell_run
import signac


def plot_vs_density(x, y, ylabel=None, fn="out.png"):
    fig, ax = pplt.subplots(aspect=(golden * 3, 3))

    ax.plot(x, y, "*--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ ($\mathrm{cm^{-3}}$)")
    ax.set_ylabel(ylabel)

    ax.grid(which="both")

    fig.savefig(fn, transparent=False)


def main():
    proj = signac.get_project(search=False)
    N = len(proj)

    job_densities = np.empty(N) / u.cm ** 3
    job_centroid_positions = np.empty(N) * u.micrometer
    job_bending_energies = np.empty(N) / u.micrometer

    for count, (_, jobs) in enumerate(proj.groupby(key="n_e")):
        job = next(jobs)  # assuming single job per group

        ne = job.sp.n_e / u.meter ** 3
        job_densities[count] = ne.to(1 / u.cm ** 3)

        # get final iteration
        h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
        time_series = LpaDiagnostics(h5_path, check_all_files=True)
        it = time_series.iterations[-1]

        W = (
            compute_bending_energy(
                iteration=it, tseries=time_series, smoothing_factor=1e-8
            )
            / u.meter
        )
        job_bending_energies[count] = W.to(u.micrometer ** (-1))

        x_avg = (
            centroid_plot(
                iteration=it,
                tseries=time_series,
                smoothing_factor=1e-8,
                fn_postfix=f"{count:06d}",
                vmax=5e5,
                plot_range=[[None, None], [-600e-6, 400e-6]],
                cmap="cividis",
                annotation=f"ne = {job_densities[count]:.3e}, W = {job_bending_energies[count]:.3e}",
            )[2]
            * u.meter
        )
        job_centroid_positions[count] = x_avg.to(u.micrometer)

    plot_vs_density(
        job_densities,
        job_centroid_positions,
        ylabel=r"$\langle x \rangle$ ($\mathrm{\mu m}$)",
        fn=f"average_centroids_no_cut.png",
    )
    plot_vs_density(
        job_densities,
        job_bending_energies,
        ylabel=r"$W$ ($\mathrm{\mu m^{-1}}$)",
        fn=f"bending_energies_no_cut.png",
    )

    command = ffmpeg_command(
        input_files=pathlib.Path.cwd() / "centroid*.png", output_file="centroid.mp4"
    )
    shell_run(command, shell=True)


if __name__ == "__main__":
    main()
