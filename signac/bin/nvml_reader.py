"""
Reads GPU usage info from CSV file and plots it.
Saves the figure(s) as PNG file(s).
Usage: python nvml_reader.py filename.csv
"""
import pathlib
from dataclasses import dataclass, field
from typing import List, ClassVar, Tuple

import numpy as np
import pandas as pd
import pynvml
from matplotlib import pyplot
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def normalize_to_interval(a, b, data):
    """Given the `data` array, normalize its values in the [a, b] interval."""
    d = np.atleast_1d(data.copy())
    norm_data = (b - a) * (d - d.min()) / (d.max() - d.min()) + a
    return norm_data


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


class GpuListError(Exception):
    """Custom Error class for GpuList."""


class GpuPanelError(Exception):
    """Custom Error class for GpuPanel."""


class GpuFigureError(Exception):
    """Custom Error class for GpuFigure."""


@dataclass
class GpuDevice:
    """Stores GPU properties, per device."""

    index: int  # from 0 to N - 1, with N the number of GPUs
    uuid: str  # eg, "GPU-1dfe3b5c-79a0-0422-f0d0-b22c6ded0af0"
    uuid_short: str = field(init=False)  #  eg, "GPU-1d"
    total_memory: int  # MiB (Mebibyte)
    power_limit: int  # Watt


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


@dataclass
class Tick:
    __slots__ = ["position", "label"]
    position: float
    label: str


@dataclass
class HorizontalBar:
    y_position: float
    width: float
    xdata: np.ndarray = field(repr=False)  # TODO add repr=False
    ydata: np.ndarray = field(repr=False)  # TODO add repr=False
    ax: Axes = field(repr=False)
    label: str = field(repr=False)
    ticks: List[Tick] = field(repr=False)
    index: int  # from 0 to N - 1, with N the number of GPUs
    height: ClassVar[float] = 0.75
    background_color: ClassVar[str] = "0.95"
    hide_tick_labels: bool = True
    ydata_normalized: np.ndarray = field(init=False, repr=False)
    y_extent: Tuple[float] = field(init=False, repr=False)

    def __post_init__(self):
        self.y_extent = (
            self.y_position - self.height / 2,
            self.y_position + self.height / 2,
        )
        self.ydata_normalized = self.normalize_y()

    def __eq__(self, other):
        if not isinstance(other, HorizontalBar):
            return NotImplemented
        return self.index == other.index

    def __str__(self):
        return f"{self.index} {self.label}: {self.y_extent[0]}<{self.y_position}>{self.y_extent[1]}"

    def normalize_y(self, y_data=None):
        if y_data is None:
            y_data = self.ydata

        a, b = self.y_extent
        return normalize_to_interval(a, b, y_data)

    def draw_solid_background(self):
        self.ax.barh(
            y=self.y_position,
            width=self.width,
            height=self.height,
            color=self.background_color,
            zorder=-20,
        )

    def add_label(self, left_offset=-0.1):
        self.ax.text(
            x=left_offset,
            y=self.y_position,
            s=self.label,
            horizontalalignment="right",
            fontsize=16,
        )

    def add_left_spine(self, color="black", linewidth=3):
        self.ax.axvline(
            x=0,
            ymin=(self.y_position - 0.4) / 8,  # FIXME
            ymax=(self.y_position + 0.4) / 8,
            color=color,
            linewidth=linewidth,
        )

    def add_vertical_tick_line(self, x_pos, color="0.5", linewidth=0.5):
        self.ax.axvline(
            x=x_pos,
            ymin=(self.y_position - 0.375) / 8,  # FIXME
            ymax=(self.y_position + 0.375) / 8,
            color=color,
            linewidth=linewidth,
            zorder=-15,
        )

    def label_tick(self, x_pos, tick_label):
        self.ax.text(
            x=x_pos,
            y=0,
            s=tick_label,
            verticalalignment="top",
            horizontalalignment="center",
            fontsize=10,
        )

    def add_tick_lines(self):
        for tick in self.ticks:
            self.add_vertical_tick_line(tick.position * 8)  # FIXME

    def add_tick_labels(self):
        for tick in self.ticks:
            self.label_tick(tick.position * 8, tick.label)  # FIXME

    def add_ticks(self):
        self.add_tick_lines()
        if not self.hide_tick_labels:
            self.add_tick_labels()

    def prepare(self):
        self.draw_solid_background()
        self.add_label()
        self.add_left_spine()
        self.add_ticks()

    def plot_data(self, y_data=None, color="black", linewidth=2, zorder=2):
        if y_data is None:
            y_data = self.ydata_normalized

        self.ax.plot(
            self.xdata, y_data, color=color, linewidth=linewidth, zorder=zorder
        )


@dataclass
class GpuPanel:
    ax: Axes = field(repr=False)
    gpus: GpuList
    xdata: np.ndarray = field(repr=False)  # TODO add repr=False
    ydata: np.ndarray = field(repr=False)  # TODO add repr=False
    num_gpus: int = field(init=False, repr=False)
    indexes: List[int] = field(init=False, repr=False)
    short_uuids: List[str] = field(init=False, repr=False)
    bars: List[HorizontalBar] = field(init=False)  # TODO add repr=False

    def __post_init__(self):
        if not self.gpus:
            raise GpuPanelError("A plot panel must have at least 1 GPU.")
        for attr in "num_gpus", "indexes", "short_uuids":
            setattr(self, attr, getattr(self.gpus, attr))
        nrows, ncols = self.ydata.shape
        if nrows != self.num_gpus:
            raise GpuPanelError(
                "Number of rows in Y data different from number of GPUs in Panel."
            )
        self.create_horizontal_bars()

    def create_horizontal_bars(self):
        """
        We split the y-axis of self.ax into n_bars intervals of height 0.75
        the height of the y-axis is 8.0
        """
        n_bars = 8  # FIXME
        bar_center = n_bars - (np.arange(n_bars) + 0.5)
        bar_width = np.full(shape=n_bars, fill_value=n_bars, dtype=float)
        bar_ticks = [
            Tick(pos, label)
            for pos, label in zip((0.25, 0.5, 0.75), ("0.5", "1.0", "1.5"))
        ]  # FIXME

        bars = list()
        for idx, y_pos, width, s_uuid, y_row in zip(
            self.indexes, bar_center, bar_width, self.short_uuids, self.ydata
        ):
            hb = HorizontalBar(
                y_position=y_pos,
                width=width,
                xdata=self.xdata,
                ydata=y_row,
                ax=self.ax,
                label=s_uuid,
                ticks=bar_ticks,
                index=idx,
            )
            bars.append(hb)
        self.bars = bars

    def set_axis_limits(self):
        self.ax.set_xlim(0, 8)
        self.ax.set_ylim(0, 8)  # FIXME

    def hide_axis(self):
        self.ax.set_axis_off()

    def draw_horizontal_bars(self):
        """Plot n bars of equal widths displaced vertically."""
        for bar in self.bars:
            bar.prepare()

    def add_tick_labels(self):
        """Only add labels to bottom bar."""
        self.bars[-1].hide_tick_labels = False
        self.bars[-1].add_tick_labels()

    def prepare(self):
        self.set_axis_limits()
        self.hide_axis()
        self.draw_horizontal_bars()
        self.add_tick_labels()

    def plot_lines(self):
        for bar in self.bars:
            bar.plot_data()


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
    bars: List[HorizontalBar] = field(init=False)  # TODO add repr=False

    def __post_init__(self):
        self.create_panels()
        for attr in "indexes", "num_gpus", "bars":
            setattr(
                self,
                attr,
                getattr(self.panel_left, attr) + getattr(self.panel_right, attr),
            )
        if self.num_gpus != self.gpus.num_gpus:
            raise GpuFigureError(
                "Number of GPUs in figure is different from the sum of the number of GPUs in each panel."
            )
        nrows, ncols = self.Y.shape
        if nrows != self.num_gpus:
            raise GpuFigureError(
                "Number of rows in Y data different from total number of GPUs."
            )

    def __iter__(self):
        return iter((self.panel_left, self.panel_right))

    def create_panels(self, figsize=(20, 8)):
        self.fig = pyplot.figure(figsize=figsize)

        nrows, ncols = 1, 2
        axs, subplots = dict(), dict(left=1, right=2)
        for pos, index in subplots.items():
            axs[pos] = pyplot.subplot(nrows, ncols, index, aspect=1)

        gpus_left, gpus_right = self.gpus.split()  # FIXME
        assert (
            gpus_left.num_gpus == gpus_right.num_gpus
        ), "Different number of GPUs in the two panels."
        self.xdata = self.X * gpus_left.num_gpus / self.X.max()  # rescaling x data

        ydata_left, ydata_right = np.vsplit(self.Y, 2)

        self.panel_left = GpuPanel(axs["left"], gpus_left, self.xdata, ydata_left)
        self.panel_right = GpuPanel(axs["right"], gpus_right, self.xdata, ydata_right)

    def prepare(self):
        for panel in self:
            panel.prepare()

    def plot_lines(self):
        for panel in self:
            panel.plot_lines()

    def plot_other_lines(self):
        for main_bar in self.bars:
            for secondary_bar in self.bars:
                if main_bar != secondary_bar:
                    main_bar.plot_data(
                        y_data=main_bar.normalize_y(secondary_bar.ydata),
                        color="0.5",
                        linewidth=0.5,
                        zorder=-10,
                    )

    def render(self):
        self.prepare()
        self.plot_lines()
        self.plot_other_lines()

    def save(self, fname="gpu_figure.png", dpi=192):
        self.fig.savefig(fname, dpi=dpi)
        pyplot.close(self.fig)


def main():
    """Main entry point."""
    gpus = GpuList(get_all_gpus()[:4])
    # TODO implement slicing

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
    f.render()
    f.save()

    # i uuid   c   low    high
    # 0 GPU-c5 7.5 7.3125 7.625
    # 1 GPU-61 6.5 6.3125 6.625
    # 2 GPU-89 5.5 5.3125 5.625
    # 3 GPU-cd 4.5 4.3125 4.625
    # 4 GPU-c7 3.5 3.3125 3.625
    # 5 GPU-60 2.5 2.3125 2.625
    # 6 GPU-3c 1.5 1.3125 1.625
    # 7 GPU-fd 0.5 0.3125 0.625

    # TODO: add extra padding ^^
    # https://ivergara.github.io/ABC-and-dataclasses.html
    for bar in f.panel_right.bars:
        print(bar)

    # 8 GPU-ce: 7.125<7.5>7.875
    # 9 GPU-db: 6.125<6.5>6.875
    # 10 GPU-d0: 5.125<5.5>5.875
    # 11 GPU-91: 4.125<4.5>4.875
    # 12 GPU-28: 3.125<3.5>3.875
    # 13 GPU-1e: 2.125<2.5>2.875
    # 14 GPU-1d: 1.125<1.5>1.875
    # 15 GPU-04: 0.125<0.5>0.875

# nvml_20210215-193040.csv

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
