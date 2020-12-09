import pathlib
from openpmd_viewer import addons
from project import particle_energy_histogram
from matplotlib import pyplot
import unyt as u
import signac


def main():
    proj = signac.get_project(root="../", search=False)

    fig, ax = pyplot.subplots(figsize=(10, 6))

    ax.set_xlabel("E (MeV)")
    ax.set_ylim(0, 35)
    ax.set_xlim(1, 500)
    ax.set_ylabel("dQ/dE (pC/MeV)")

    for zf, jobs in proj.groupby(key="zfoc"):
        job = next(jobs)  # assuming single job per group

        # get path to job's hdf5 files
        h5_path = pathlib.Path(job.ws) / "diags" / "hdf5"

        # get last iteration from time series
        time_series = addons.LpaDiagnostics(h5_path, check_all_files=True)
        last_iteration = time_series.iterations[-1]

        # compute 1D histogram
        energy_hist, bin_edges, nrbins = particle_energy_histogram(
            tseries=time_series,
            it=last_iteration,
        )
        zfoc = (job.sp.zfoc * u.meter).to("micrometer")

        ax.step(bin_edges[1:], energy_hist, label=f"$z_f$ = {zfoc:.0f}")

    ax.legend()
    fig.savefig("zoo.png")


if __name__ == "__main__":
    main()
