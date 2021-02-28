"""
Reads GPU usage info from CSV file and plots it.
Saves the figure(s) as PNG file(s).
Usage: python nvml_reader.py filename.csv
"""
import collections.abc
import pathlib
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import pynvml


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


@dataclass
class GpuDevice:
    """Stores GPU properties, per device."""

    index: int  # from 0 to N - 1, with N the number of GPUs
    uuid: str  # eg, "GPU-1dfe3b5c-79a0-0422-f0d0-b22c6ded0af0"
    uuid_short: str = field(init=False)  #  eg, "GPU-1d"
    total_memory: int  # MiB (Mebibyte)
    power_limit: int  # Watt


@dataclass
class GpuList(collections.abc.Sequence):
    gpus: List[GpuDevice]
    num_gpus: int = field(init=False, repr=False)
    uuids: List[str] = field(init=False, repr=False)
    short_uuids: List[str] = field(init=False, repr=False)
    indexes: List[int] = field(init=False, repr=False)

    def __post_init__(self):
        self.num_gpus = len(self)
        self.uuids = [gpu.uuid for gpu in self]
        self.indexes = [gpu.index for gpu in self]
        self.set_short_uuids()

    def __getitem__(self, key):
        return self.gpus.__getitem__(key)

    def __len__(self):
        return self.gpus.__len__()

    def set_short_uuids(self):
        min_len = min_len_unique_uuid(self.uuids)
        # mutating!
        short_ids = list()
        for gpu in self:
            gpu.uuid_short = gpu.uuid[:min_len]
            short_ids.append(gpu.uuid_short)
        self.short_uuids = short_ids


def main():
    """Main entry point."""
    gpus = get_all_gpus()

    # ----------------------------------------------------------------------------------

    p = pathlib.Path.cwd() / "nvml_20210228-165008.csv"
    df = pd.read_csv(p)

    uuid_min_len = len(gpus[0].uuid_short)
    df["gpu_uuid"] = df["gpu_uuid"].astype("string")
    df["short_gpu_uuid"] = df["gpu_uuid"].str[:uuid_min_len]

    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df.set_index("time_stamp", inplace=True)

    # select by date / time
    # df = df.loc["2021-02-10 11:28":"2021-02-10 20:24"]

    grouped = df.groupby(["short_gpu_uuid"])
    print(grouped[["used_power_W", "used_gpu_memory_MiB"]].agg(["max", "mean", "std"]))


# "nvml_20210228-165008.csv" -> "nvml_20210228-165008.png"

# The GpuDevice has an associated GpuData object holding a dataframe.

# To build the figure, the following steps are needed:
# 1. Build a list of GPU devices on the system,
# with total memory and power limit for each
# uuid_short: str = field(init=False)  #  eg, "GPU-1d"
# total_memory: int  # MiB (Mebibyte)
# power_limit: int  # Watt

# 1. get a list of unique GPUs from the .csv file
# 2. for each such GPU, build a GpuData object
# holding *normalized* values of power and memory usage

# Separate figures for used power and used memory
# normalize used power and memory by their MAX values

if __name__ == "__main__":
    main()
