import numpy as np
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
import unyt as u
import signac
import electron_spectrum as es


def main():
    proj = signac.get_project(search=False)

    spectra = es.multiple_jobs_single_iteration(
        jobs=proj.find_jobs(),
        key="zfoc_from_nozzle_center",
        label=lambda job, key: f"{key} = {job.sp[key] * 1.0e+6:.0f}",
    )

    peak_position_charge = list()
    for _, jobs in proj.groupbydoc(key="x"):
        job = next(jobs)  # assuming single job per group
        peak_position_charge.append(
            (
                float(f"{x.to_value():.0f}"),
                job.doc.peak_position,
                job.doc.peak_charge,
            )
        )
        # TODO replace with code from electron_spectrum.py

    # job.doc.peak_charge: 1496
    # job.doc.peak_position: 212

    x, peak_position, peak_charge = zip(*peak_position_charge)

    fig, ax1 = pyplot.subplots()
    ax2 = ax1.twinx()

    ax1.hlines(
        y=250,
        xmin=x[0],
        xmax=x[-1],
        linewidth=0.75,
        linestyle="dashed",
        color="0.75",
    )
    ax1.plot(x, peak_position, "o--", color="C1")
    ax2.plot(x, peak_charge, "o--", color="C0")

    ax1.set_xlabel(r"$x$ ($\mathrm{\mu m}$)")
    ax1.set_ylabel("E (MeV)", color="C1")
    ax2.set_ylabel("Q (pC)", color="C0")

    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)

    ax2.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.spines["left"].set_color("C1")
    ax2.spines["right"].set_color("C0")

    ax1.tick_params(axis="y", colors="C1")
    ax2.tick_params(axis="y", colors="C0")

    ax1.xaxis.set_major_locator(MultipleLocator(200))
    ax1.tick_params(labelbottom=True, labeltop=True)

    ax1.grid(which="major", axis="x", linewidth=0.75, linestyle="dashed", color="0.75")

    fig.savefig("positions_charges.png")


if __name__ == "__main__":
    main()
