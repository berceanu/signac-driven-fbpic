"""
Reads GPU usage info from CSV file and plots it.
Saves the figure(s) as PNG file(s).
Usage: python nvml_reader.py filename.csv
"""
import collections.abc
import pathlib
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import pandas as pd
import pynvml
from horizontal_bars_figure import FigureHorizontalBars, Tick


def get_gpus_in_csv(path_to_csv, gpus_on_machine):
    df = pd.read_csv(path_to_csv)
    csv_uuids = tuple(df["gpu_uuid"].unique())

    gpus_in_csv = list()

    for gpu in gpus_on_machine:
        if gpu.uuid in csv_uuids:
            gpus_in_csv.append(gpu)

    return GpuList(tuple(gpus_in_csv))


def min_len_unique_uuid(uuids):
    """Determine the minimum string length required for a UUID to be unique."""

    def length_of_longest_string(list_of_strings):
        return len(max(list_of_strings, key=len))

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


def get_gpus_on_machine():
    """Populates the list of available GPUs on the machine."""
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()

    gpus_on_machine = list()
    for gpu in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        uuid = pynvml.nvmlDeviceGetUUID(handle).decode("UTF-8")
        mem_total = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024 ** 2
        enforced_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000
        device = GpuDevice(
            uuid=uuid,
            total_memory=int(mem_total),
            power_limit=int(enforced_power_limit),
        )
        gpus_on_machine.append(device)
    pynvml.nvmlShutdown()
    return GpuList(tuple(gpus_on_machine))


@dataclass
class GpuDevice:
    """Stores GPU properties, per device."""

    uuid: str  # eg, "GPU-1dfe3b5c-79a0-0422-f0d0-b22c6ded0af0"
    total_memory: int  # MiB (Mebibyte)
    power_limit: int  # Watt


@dataclass
class GpuList(collections.abc.Sequence):
    gpus: Tuple[GpuDevice]
    short_uuid_len: int = field(init=False)

    def __post_init__(self):
        self.short_uuid_len = min_len_unique_uuid(list(gpu.uuid for gpu in self))

    def __getitem__(self, key):
        return self.gpus.__getitem__(key)

    def __len__(self):
        return self.gpus.__len__()


def main():
    """Main entry point."""
    path_to_csv = pathlib.Path.cwd() / "nvml_20210228-165008.csv"

    gpus_on_machine = get_gpus_on_machine()
    gpus_in_csv = get_gpus_in_csv(path_to_csv, gpus_on_machine)

    # ----------------------------------------------------------------------------------


    df = pd.read_csv(path_to_csv)

    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df.set_index("time_stamp", inplace=True)

    df["gpu_uuid"] = df["gpu_uuid"].astype("string")
    df["hw_slowdown"] = df["hw_slowdown"].astype("category")
    df["sw_power_cap"] = df["sw_power_cap"].astype("category")


    grouped = df.groupby(["gpu_uuid"])
    print(grouped[["used_power_W", "used_gpu_memory_MiB"]].agg(["max", "mean", "std"]))

    # df = df.resample('3T').mean()


    X = np.linspace(start=0, stop=2 * np.pi, num=81, endpoint=True)
    Y = np.zeros(shape=(len(gpus_in_csv), 81), dtype=np.float64)

    ylabels = list()
    for row, gpu in enumerate(gpus_in_csv):
        mask = df["gpu_uuid"]==gpu.uuid
        series = df.loc[mask, "used_gpu_memory_MiB"].resample('5T').mean() / gpu.total_memory
        size = len(series)
        print(size)
        Y[row, :size] = series
        ylabels.append(gpu.uuid[:gpus_in_csv.short_uuid_len])

    print(Y.max()*100, Y.min())
    Y_labels = tuple(ylabels)
    X_ticks = tuple(
        [
            Tick(0.0, "$0$"),
            Tick(0.25, "$\\frac{\\pi}{2}$"),
            Tick(0.5, "$\\pi$"),
            Tick(0.75, "$\\frac{3\\pi}{2}$"),
            Tick(1.0, "$2\\pi$"),
        ]
    )

    f = FigureHorizontalBars(
        X=X,
        Y=Y,
        x_ticks=X_ticks,
        y_labels=Y_labels,
    )
    f.render()
    f.save(fname="nvml_20210228-165008.png")






if __name__ == "__main__":
    main()
