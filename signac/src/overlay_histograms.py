from collections import defaultdict
import numpy as np
from matplotlib import pyplot
from matplotlib.ticker import MultipleLocator
from cycler import cycler
from scipy.constants import golden
import unyt as u
import signac

line_colors = ["C1", "C2", "C3"]
line_styles = ["-", "--", ":", "-."]
cyl = cycler(color=line_colors) * cycler(linestyle=line_styles)
loop_cy_iter = cyl()
STYLE = defaultdict(lambda: next(loop_cy_iter))


def main():
    proj = signac.get_project(search=False)

    peak_position_charge = list()

    fig, ax = pyplot.subplots(figsize=(golden * 8, 8))

    ax.set_xlabel("E (MeV)")
    ax.set_ylabel("dQ/dE (pC/MeV)")

    for zf, jobs in proj.groupby(key="zfoc"):
        job = next(jobs)  # assuming single job per group

        zfoc = (job.sp.zfoc * u.meter).to(u.micrometer)
        label = f"$z_f$ = {zfoc:.0f}"

        peak_position_charge.append(
            (
                float(f"{zfoc.to_value():.0f}"),
                job.doc.peak_position,
                job.doc.peak_charge,
            )
        )

        npzfile = np.load(job.fn("final_histogram.npz"))
        energy = npzfile["edges"][1:]
        charge = npzfile["counts"]

        mask = (energy > 100) & (energy < 200)  # MeV
        energy = energy[mask]
        charge = np.clip(charge, 0, 60)[mask]

        ax.step(
            energy,
            charge,
            label=label,
            color=STYLE[label]["color"],
            linestyle=STYLE[label]["linestyle"],
            linewidth=0.5,
        )

    ax.legend(frameon=False)
    ax.axvline(x=160, linewidth=0.75, linestyle="dashed", color="0.75")

    fig.savefig("histograms.png", dpi=192)

    zfoc, peak_position, peak_charge = zip(*peak_position_charge)

    fig, ax1 = pyplot.subplots(figsize=(golden * 8, 8))
    ax2 = ax1.twinx()

    ax1.hlines(
        y=160,
        xmin=zfoc[0],
        xmax=zfoc[-1],
        linewidth=0.75,
        linestyle="dashed",
        color="0.75",
    )
    ax1.plot(zfoc, peak_position, "o--", color="C1")
    ax2.plot(zfoc, peak_charge, "o--", color="C0")

    ax1.set_xlabel(r"$z_f$ ($\mathrm{\mu m}$)")
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

    ax1.tick_params(labelbottom=True, labeltop=True)
    ax1.xaxis.set_major_locator(MultipleLocator(50))

    ax1.grid(which="major", axis="x", linewidth=0.75, linestyle="dashed", color="0.75")

    fig.savefig("positions_charges.png")


if __name__ == "__main__":
    main()
