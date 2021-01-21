import pathlib
import numpy as np
from matplotlib import pyplot
from scipy.constants import golden
import unyt as u
from openpmd_viewer.addons import LpaDiagnostics
from simulation_diagnostics import (
    plot_spline_derivatives,
    centroid_plot,
    compute_bending_energy,
)
import signac


def plot_vs_density(x, y, ylabel="", fn="out.png", up_to=None):
    if up_to is not None:
        x = x[:up_to].copy()
        y = y[:up_to].copy()

    fig, ax = pyplot.subplots(figsize=(golden * 4, 4))

    ax.plot(x, y, "*--", linewidth=1)

    ax.set_xscale("log")
    ax.set_xlabel(r"$n_e$ (cm${}^{-3}$)")
    ax.set_ylabel(ylabel)

    ax.grid(which="both")

    fig.savefig(fn, bbox_inches="tight")
    pyplot.close(fig)


def main():
    proj = signac.get_project(search=False)
    N = len(proj)

    job_densities = np.empty(N) / u.cm ** 3
    job_centroid_positions = np.empty(N) * u.micrometer
    job_bending_energies = np.empty(N) / u.micrometer

    counter = 0

    for _, jobs in proj.groupby(key="n_e"):
        job = next(jobs)  # assuming single job per group

        ne = job.sp.n_e / u.meter ** 3
        job_densities[counter] = ne.to(1 / u.cm ** 3)

        # get final iteration
        h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"
        time_series = LpaDiagnostics(h5_path, check_all_files=True)
        it = time_series.iterations[-1]

        x_avg = centroid_plot(iteration=it, tseries=time_series)[2] * u.meter
        job_centroid_positions[counter] = x_avg.to(u.micrometer)

        # plot_spline_derivatives(iteration=it, tseries=time_series)

        W = compute_bending_energy(iteration=it, tseries=time_series) / u.meter
        job_bending_energies[counter] = W.to(u.micrometer ** (-1))

        print(job.id)
        print(it)

        counter += 1


    for UP_TO in 19, None:
        plot_vs_density(
            job_densities,
            job_centroid_positions,
            up_to=UP_TO,
            ylabel=r"$\langle x \rangle$ ($\mu$m)",
            fn=f"average_centroids_{UP_TO}.png",
        )
        plot_vs_density(
            job_densities,
            job_bending_energies,
            up_to=UP_TO,
            ylabel=r"$W$ ($\mu$m${}^{-1}$)",
            fn=f"bending_energies_{UP_TO}.png",
        )


if __name__ == "__main__":
    main()
