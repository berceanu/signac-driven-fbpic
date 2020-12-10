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
    ax.axvline(x=160, color="0.75", linestyle="dotted")

    fig.savefig("histograms.png", dpi=192, transparent=False)

    zfoc, peak_position, peak_charge = zip(*peak_position_charge)

    fig, ax1 = pyplot.subplots(figsize=(golden * 8, 8))
    ax2 = ax1.twinx()

    ax1.axhline(y=160, color="C1", linestyle="dotted")
    ax1.plot(zfoc, peak_position, "o--", color="C1")
    ax2.plot(zfoc, peak_charge, "o--", color="C0")

    ax1.set_xlabel(r"$z_f$ ($\mathrm{\mu m}$)")
    ax1.set_ylabel("E (MeV)", color="C1")
    ax2.set_ylabel("Q (pC)", color="C0")

    ax1.xaxis.set_major_locator(MultipleLocator(50))

    ax1.grid(which="major", axis="x", linewidth=0.75, linestyle="dashed", color="0.75")

    fig.savefig("positions_charges.png")


if __name__ == "__main__":
    main()
