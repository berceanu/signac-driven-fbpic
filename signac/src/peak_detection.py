"""Module containing peak-detection algorithm and related functionality."""

from collections import defaultdict
import numpy as np
from matplotlib import pyplot
from cycler import cycler

line_colors = ["C1", "C2", "C3", "C4", "C5", "C6"]
line_styles = ["-", "--", ":", "-.", (0, (1, 10)), (0, (5, 10))]
cyl = cycler(color=line_colors) + cycler(linestyle=line_styles)
loop_cy_iter = cyl()
STYLE = defaultdict(lambda: next(loop_cy_iter))


class Peak:
    def __init__(self, startidx):
        self.born = self.left = self.right = startidx
        self.died = None

    def get_persistence(self, seq):
        return float("inf") if self.died is None else seq[self.born] - seq[self.died]


def get_persistent_homology(seq):
    peaks = []
    # Maps indices to peaks
    idxtopeak = [None for s in seq]
    # Sequence indices sorted by values
    indices = range(len(seq))
    indices = sorted(indices, key=lambda i: seq[i], reverse=True)

    # Process each sample in descending order
    for idx in indices:
        lftdone = idx > 0 and idxtopeak[idx - 1] is not None
        rgtdone = idx < len(seq) - 1 and idxtopeak[idx + 1] is not None
        il = idxtopeak[idx - 1] if lftdone else None
        ir = idxtopeak[idx + 1] if rgtdone else None

        # New peak born
        if not lftdone and not rgtdone:
            peaks.append(Peak(idx))
            idxtopeak[idx] = len(peaks) - 1

        # Directly merge to next peak left
        if lftdone and not rgtdone:
            peaks[il].right += 1
            idxtopeak[idx] = il

        # Directly merge to next peak right
        if not lftdone and rgtdone:
            peaks[ir].left -= 1
            idxtopeak[idx] = ir

        # Merge left and right peaks
        if lftdone and rgtdone:
            # Left was born earlier: merge right to left
            if seq[peaks[il].born] > seq[peaks[ir].born]:
                peaks[ir].died = idx
                peaks[il].right = peaks[ir].right
                idxtopeak[peaks[il].right] = idxtopeak[idx] = il
            else:
                peaks[il].died = idx
                peaks[ir].left = peaks[il].left
                idxtopeak[peaks[ir].left] = idxtopeak[idx] = ir

    # This is optional convenience
    return sorted(peaks, key=lambda p: p.get_persistence(seq), reverse=True)


def integrated_charge(spectrum_file, from_energy, to_energy):
    """Compute the integrated charge between the two energy limits."""

    npzfile = np.load(spectrum_file)
    energy = npzfile["edges"]
    charge = npzfile["counts"]

    delta_energy = np.diff(energy)
    energy = energy[1:]

    mask = (energy >= from_energy) & (energy <= to_energy)  # MeV

    Q = np.sum(delta_energy[mask] * charge[mask])  # integrated charge

    return Q


def peak_position(spectrum_file, from_energy, to_energy):
    """Find position of max charge in given interval."""

    npzfile = np.load(spectrum_file)
    energy = npzfile["edges"][1:]
    charge = npzfile["counts"]

    mask = (energy >= from_energy) & (energy <= to_energy)  # MeV
    energy = energy[mask]
    charge = charge[mask]

    idx_max = np.argmax(charge)

    return energy[idx_max]


def plot_electron_energy_spectrum(spectrum_file, fig_file) -> None:
    """Plot the electron spectrum from file."""

    npzfile = np.load(spectrum_file)
    energy = npzfile["edges"]
    charge = npzfile["counts"]

    delta_energy = np.diff(energy)
    energy = energy[1:]

    mask = (energy > 0) & (energy < 350)  # MeV
    energy = energy[mask]
    charge = np.clip(charge, 0, 60)[mask]

    h = get_persistent_homology(charge)

    # plot it
    fig, ax = pyplot.subplots(figsize=(10, 6))

    ax.step(
        energy,
        charge,
    )
    ax.set_xlabel("E (MeV)")
    ax.set_ylabel("dQ/dE (pC/MeV)")

    for peak_number, peak in enumerate(
        h[:6]
    ):  # go through first peaks, in order of importance
        peak_index = peak.born
        energy_position = energy[peak_index]
        charge_value = charge[peak_index]

        persistence = peak.get_persistence(charge)
        ymin = charge_value - persistence
        if np.isinf(persistence):
            ymin = 0

        Q = np.sum(
            delta_energy[peak.left : peak.right] * charge[peak.left : peak.right]
        )  # integrated charge
        ax.annotate(
            text=f"#{peak_number}, {Q:.0f} pC, {energy_position:.1f} MeV",
            xy=(energy_position + 5, charge_value + 0.02),
            xycoords="data",
            color=STYLE[str(peak_index)]["color"],
        )
        ax.axvline(
            x=energy_position,
            linestyle=STYLE[str(peak_index)]["linestyle"],
            color=STYLE[str(peak_index)]["color"],
            linewidth=2,
        )
        ax.fill_between(
            energy,
            charge,
            ymin,
            where=(energy > energy[peak.left]) & (energy <= energy[peak.right]),
            color=STYLE[str(peak_index)]["color"],
            alpha=0.9,
        )
    fig.savefig(fig_file)
    pyplot.close(fig)


def main():
    import random
    import signac

    random.seed(42)

    proj = signac.get_project(search=False)
    ids = [job.id for job in proj]
    job = proj.open_job(id=random.choice(ids))

    plot_electron_energy_spectrum(job.fn("final_histogram.npz"), "final_histogram.png")

    Q = integrated_charge(job.fn("final_histogram.npz"), from_energy=100, to_energy=200)
    pos = peak_position(job.fn("final_histogram.npz"), from_energy=100, to_energy=200)

    print(f"{Q} pc between 100 and 200 MeV. peak at {pos:.1f} MeV")


if __name__ == "__main__":
    main()
