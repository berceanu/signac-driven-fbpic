from cycler import cycler
import numpy as np
import sliceplots
from matplotlib import pyplot
from collections import defaultdict

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
        lftdone = (idx > 0 and idxtopeak[idx - 1] is not None)
        rgtdone = (idx < len(seq) - 1 and idxtopeak[idx + 1] is not None)
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




if __name__ == "__main__":
    npzfile = np.load('histogram.npz')

    edges = npzfile['edges']
    counts = npzfile['counts']

    energy = np.array([edges[:-1], edges[1:]]).T.flatten()
    charge = np.array([counts, counts]).T.flatten()

    mask = (energy > 90) & (energy < 710)  # MeV
    energy = energy[mask]
    charge = np.clip(charge, 0, 1)[mask]

    h = get_persistent_homology(charge)

    # plot it
    fig, ax = pyplot.subplots(figsize=(10, 6))
    sliceplots.plot1d(
        ax=ax,
        v_axis=charge,
        h_axis=energy,
        xlabel=r"E (MeV)",
        ylabel=r"dQ/dE (pC/MeV)",
        ylim=[0, 1.1]
    )
    for peak_number, peak in enumerate(h[:6]):  # go through first peaks, in order of importance
        peak_index = peak.born
        energy_position = energy[peak_index]
        charge_value = charge[peak_index]
        persistence = peak.get_persistence(charge)
        ymin = charge_value - persistence
        if persistence == float("inf"):
            ymin = 0
        ax.annotate(s=f"{peak_number}", xy=(energy_position + 5, charge_value + 0.02), xycoords="data",
                    color=STYLE[str(peak_index)]["color"], size=14)
        ax.axvline(x=energy_position, linestyle=STYLE[str(peak_index)]["linestyle"],
                color=STYLE[str(peak_index)]["color"], linewidth=2)
        ax.fill_between(energy, charge, ymin, where=(energy > energy[peak.left]) & (energy <= energy[peak.right]),
                        color=STYLE[str(peak_index)]["color"], alpha=0.9)

    fig.savefig('peak_detection.png')
