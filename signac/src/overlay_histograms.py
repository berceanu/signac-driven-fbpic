from collections import defaultdict
import numpy as np
from matplotlib import pyplot
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

    fig, ax = pyplot.subplots(figsize=(golden * 8, 8))

    ax.set_xlabel("E (MeV)")
    ax.set_ylabel("dQ/dE (pC/MeV)")

    for zf, jobs in proj.groupby(key="zfoc"):
        job = next(jobs)  # assuming single job per group

        zfoc = (job.sp.zfoc * u.meter).to(u.micrometer)
        label = f"$z_f$ = {zfoc:.0f}"

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
    ax.legend()
    fig.savefig("histograms.png", dpi=192, transparent=False)


if __name__ == "__main__":
    main()
