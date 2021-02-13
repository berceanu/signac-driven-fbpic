"""
Reads GPU usage info from CSV file and plots it.
Saves the figure(s) as PNG file(s).
Usage: python nvml_reader.py filename.csv
"""
import pandas as pd
import pathlib
from matplotlib import pyplot
from dataclasses import dataclass, field
import pynvml
import numpy as np
from typing import List
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def split_in_half(some_list):
    list_length = len(some_list)
    if not list_length % 2 == 0:
        raise ValueError("List should contain even number of elements.")
    half = list_length // 2
    return some_list[:half], some_list[half:]


def length_of_longest_string(list_of_strings):
    return len(max(list_of_strings, key=len))


def min_len_unique_uuid(uuids):
    """Determine the minimum string length required for a UUID to be unique."""
    uuid_length = length_of_longest_string(uuids)
    tmp = set()
    for i in range(uuid_length):
        tmp.clear()
        for _id in uuids:
            if _id[:i] in tmp:
                break
            else:
                tmp.add(_id[:i])
        else:
            break
    return i


@dataclass
class GpuDevice:
    """Stores GPU properties, per device."""

    index: int  # from 0 to N - 1, with N the number of GPUs
    uuid: str  # eg, "GPU-1dfe3b5c-79a0-0422-f0d0-b22c6ded0af0"
    uuid_short: str = field(init=False)  #  eg, "GPU-1d"
    total_memory: int  # MiB (Mebibyte)
    power_limit: int  # Watt

class GpuListError(Exception):
    """Custom Error class for GpuList."""


@dataclass
class GpuList:
    gpus: List[GpuDevice]
    num_gpus: int = field(init=False, repr=False)
    uuids: List[str] = field(init=False, repr=False)
    short_uuids: List[str] = field(init=False, repr=False)
    indexes: List[int] = field(init=False, repr=False)

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, idx):
        return self.gpus[idx]

    def __len__(self):
        return len(self.gpus)

    def __post_init__(self):
        self.num_gpus = len(self)
        self.uuids = [gpu.uuid for gpu in self]
        self.indexes = [gpu.index for gpu in self]
        self.set_short_uuids()

    def set_short_uuids(self):
        min_len = min_len_unique_uuid(self.uuids)
        # mutating!
        short_ids = list()
        for gpu in self:
            gpu.uuid_short = gpu.uuid[:min_len]
            short_ids.append(gpu.uuid_short)
        self.short_uuids = short_ids

    def split(self):
        try:
            first_half, second_half = split_in_half(self.gpus)
        except ValueError:
            raise GpuListError("Can't split in half and odd number of GPUs.")
        return GpuList(first_half), GpuList(second_half)


class GpuPanelError(Exception):
    """Custom Error class for GpuPanel."""

@dataclass
class GpuPanel:
    ax: Axes = field(repr=False)
    gpus: GpuList
    xdata: np.ndarray  # TODO add repr=False
    bar_pos: np.ndarray = field(init=False, repr=False)
    num_gpus: int = field(init=False, repr=False)
    indexes: List[int] = field(init=False, repr=False)

    def __post_init__(self):
        if not self.gpus:
            raise GpuPanelError("A plot panel must have at least 1 GPU.")
        for attr in "num_gpus", "indexes":
            setattr(self, attr, getattr(self.gpus, attr)) 
        n = self.num_gpus
        self.bar_pos = n - (np.arange(n) + 0.5)

    def draw_horizontal_bars(self):
        Xx = np.full(shape=self.num_gpus, fill_value=self.num_gpus, dtype=int)
        # plot n bars of equal widths at vertical positions bar_pos
        self.ax.barh(y=self.bar_pos, width=Xx, height=0.75, color=".95", zorder=-20)
        self.ax.set_xlim(0, self.num_gpus)
        self.ax.set_ylim(0, self.num_gpus)
        self.ax.set_axis_off()
        return self.ax

    def print_gpu_labels(self):
        for bar_pos, bar_label in zip(self.bar_pos, self.gpus.short_uuids):
            # GPU label
            self.ax.text(
                x=-0.1, y=bar_pos, s=bar_label, horizontalalignment="right", fontsize=16
            )
            # black vertical line on the left edge
            self.ax.axvline(
                x=0,
                ymin=(bar_pos - 0.4) / self.num_gpus,
                ymax=(bar_pos + 0.4) / self.num_gpus,
                color="black",
                linewidth=3,
            )
        return self.ax

    def print_major_tick_labels(self, labels=None):
        right_edge = self.ax.get_xlim()[1]
        if labels is None:
            labels = {"0.5": 0.25, "1.0": 0.5, "1.5": 0.75}
        for label, x_fraction in labels.items():
            # major tick label
            self.ax.text(
                x=x_fraction * right_edge,
                y=0,
                s=label,
                verticalalignment="top",
                horizontalalignment="center",
                fontsize=10,
            )
            for bar_pos in self.bar_pos:
                # gray vertical line at major tick
                self.ax.axvline(
                    x=x_fraction * right_edge,
                    ymin=(bar_pos - 0.375) / self.num_gpus,
                    ymax=(bar_pos + 0.375) / self.num_gpus,
                    color="0.5",
                    linewidth=0.5,
                    zorder=-15,
                )
        return self.ax

    def prepare(self):
        self.draw_horizontal_bars()
        self.print_gpu_labels()
        self.print_major_tick_labels()

    def plot_data(self, y_data, color="black", linewidth=2, zorder=2):
        self.ax.plot(
            self.xdata, y_data, color=color, linewidth=linewidth, zorder=zorder
        )
        return self.ax

class GpuFigureError(Exception):
    """Custom Error class for GpuFigure."""


@dataclass
class GpuFigure:
    gpus: GpuList = field(repr=False)
    X: np.ndarray = field(repr=False)
    Y: np.ndarray = field(repr=False)
    xdata: np.ndarray = field(init=False)  # TODO add , repr=False
    fig: Figure = field(init=False, repr=False)
    panel_left: GpuPanel = field(init=False)
    panel_right: GpuPanel = field(init=False)
    num_gpus: int = field(init=False, repr=False)
    indexes: List[int] = field(init=False, repr=False)

    def __post_init__(self):
        self.create_figure()
        for attr in "indexes", "num_gpus":
            setattr(self, attr, getattr(self.panel_left, attr) + getattr(self.panel_right, attr))
        if self.num_gpus != self.gpus.num_gpus:
            raise GpuFigureError("Number of GPUs in figure is different from the sum of the number of GPUs in each panel.")

    def __iter__(self):
        return iter((self.panel_left, self.panel_right))

    def create_figure(self, figsize=(20, 8)):
        self.fig = pyplot.figure(figsize=figsize)

        nrows, ncols = 1, 2
        axs, subplots = dict(), dict(left=1, right=2)
        for pos, index in subplots.items():
            axs[pos] = pyplot.subplot(nrows, ncols, index, aspect=1)

        left_gpus, right_gpus = self.gpus.split()  # FIXME
        assert left_gpus.num_gpus == right_gpus.num_gpus, "Different number of GPUs in the two panels."
        self.xdata = self.X * left_gpus.num_gpus / self.X.max()  # rescaling x data

        self.panel_left = GpuPanel(axs["left"], left_gpus, self.xdata)
        self.panel_right = GpuPanel(axs["right"], right_gpus, self.xdata)

    def prepare(self):
        for panel in self:
            panel.prepare()

    def plot_lines(self):
        for panel in self:
            for count, p_gpu_idx in enumerate(panel.indexes):
                # plot main line
                panel.plot_data(y_data=count + 0.5 + 2 * self.Y[p_gpu_idx] / panel.num_gpus)
                for f_gpu_idx in self.indexes:
                    if p_gpu_idx != f_gpu_idx:
                        # plot other lines
                        panel.plot_data(
                            y_data=count + 0.5 + 2 * self.Y[f_gpu_idx] / panel.num_gpus,
                            color="0.5",
                            linewidth=0.5,
                            zorder=-10,
                        )

    def save(self, fname="gpu_figure.png", dpi=192):
        self.fig.savefig(fname, dpi=dpi)
        pyplot.close(self.fig)


def get_all_gpus():
    """Populates the list of available GPUs on the machine."""
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()

    gpus = list()
    for gpu in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        uuid = pynvml.nvmlDeviceGetUUID(handle).decode("UTF-8")
        mem_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 ** 2
        enforced_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
        device = GpuDevice(
            index=gpu,
            uuid=uuid,
            total_memory=int(mem_total),
            power_limit=int(enforced_power_limit),
        )
        gpus.append(device)
    pynvml.nvmlShutdown()
    return GpuList(gpus)


def main():
    """Main entry point."""
    gpus = get_all_gpus()

    # ----------------------------------------------------------------------------------

    p = pathlib.Path.cwd() / "nvml_20210209-230247.csv"
    df = pd.read_csv(p)

    uuid_min_len = len(gpus[0].uuid_short)
    df["gpu_uuid"] = df["gpu_uuid"].astype("string")
    df["short_gpu_uuid"] = df["gpu_uuid"].str[:uuid_min_len]

    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df.set_index("time_stamp", inplace=True)

    # select by date / time
    df = df.loc["2021-02-10 11:28":"2021-02-10 20:24"]

    grouped = df.groupby(["short_gpu_uuid"])
    print(grouped[["used_power_W", "used_gpu_memory_MiB"]].agg(["max", "mean", "std"]))

    # ----------------------------------------------------------------------------------

    num_data_points = 20
    X = np.linspace(start=0, stop=2, num=num_data_points, endpoint=True)
    Y = np.random.uniform(low=-0.75, high=0.5, size=(gpus.num_gpus, num_data_points))

    # ----------------------------------------------------------------------------------

    f = GpuFigure(gpus, X, Y)
    f.prepare()
    f.plot_lines()
    f.save()

# nvml.py started @ 16:54 on 13 Feb 2021

#               Start                 End
# ------------------- -------------------
# 2021-02-10T11:28:01 2021-02-10T20:23:42
# 2021-02-10T11:28:01 2021-02-10T20:23:42

# check also sacct --format="Start, End" -j 702
#               Start                 End
# ------------------- -------------------
# 2021-02-10T21:07:04             Unknown
# 2021-02-10T21:07:04             Unknown

# Format output is, YYYY-MM-DDTHH:MM:SS

# Workflow: select a duration: Start to End
# Plot all 16 GPUs during that time window on a single figure
# Separate figures for used power and used memory
# Generic rougier-style plot which is fed the data

# normalize used power and memory by their MAX values

if __name__ == "__main__":
    main()
