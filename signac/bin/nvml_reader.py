"""
Reads GPU usage info from CSV file and plots it.
Saves the figure(s) as PNG file(s).
Usage: python nvml_reader.py filename.csv
"""
import sys
import pandas as pd
import pathlib
from matplotlib import pyplot
from dataclasses import dataclass
import pynvml

# TODO find smallest non-overlapping set of short UUID
# see how signac-dashboard does it for the job subtitle


@dataclass
class InventoryItem:
    """Class for keeping track of an item in inventory."""

    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand


@dataclass
class GpuDevice:
    """Stores GPU properties, per device."""

    index: int  # from 0 to N - 1, with N the number of GPUs
    uuid: str  # eg, "GPU-1dfe3b5c-79a0-0422-f0d0-b22c6ded0af0"
    total_memory: int  # MiB
    power_limit: int  # Watt


def get_properties_of_all_gpus():
    """Populates the GpuDevice class for each GPU on the system."""
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()

    gpus = list()
    for gpu in range(gpu_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        uuid = pynvml.nvmlDeviceGetUUID(handle).decode("UTF-8")
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_total = mem_info.total / 1024**2
        enforced_power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        device = GpuDevice(
            index=gpu,
            uuid=uuid,
            total_memory=int(mem_total),
            power_limit=int(enforced_power_limit),
        )
        gpus.append(device)
    return gpus


def plot_vs_time(grouped, *, ax, col=None, ylabel=None):
    """All plotting is done with time on the x axis,
    this is the base function for that."""
    for uuid, pid in grouped.groups.keys():
        grouped[col].get_group((uuid, pid)).plot(ax=ax, label=f"{uuid}, PID={pid}")

    ax.grid()

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Timestamp")

    return ax


def plot_gpu_usage(grouped, *, ax):
    return plot_vs_time(grouped, col="GPU_Util_%", ylabel="Utilization (%)", ax=ax)


def plot_used_power(grouped, *, ax):
    return plot_vs_time(grouped, col="used_power_W", ylabel="Power (W)", ax=ax)


def plot_used_memory(grouped, *, ax):
    return plot_vs_time(
        grouped, col="used_gpu_memory_MiB", ylabel="MEM usage (MiB)", ax=ax
    )


def main():
    """Main entry point."""
    gpus = get_properties_of_all_gpus()
    print(gpus)

    # csv_fname = str(sys.argv[1])
    p = pathlib.Path.cwd() / "nvml_20210209-230247.csv"
    csv_timestamp = p.stem.split("_")[1]

    df = pd.read_csv(p)

    df["gpu_uuid"] = df["gpu_uuid"].astype("string")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])
    df.set_index("time_stamp", inplace=True)

    # select by date / time
    df = df.loc['2021-02-10 11:28' : '2021-02-10 20:24']

    print(df.describe(), "\n")  # TODO describe after aggregation on GPU/PID

    grouped = df.groupby(["gpu_uuid", "pid"])

    # TODO separate per-GPU plots
    fig, axes = pyplot.subplots(figsize=(12, 8), nrows=3, sharex=True)

    for i, (foo, ax) in enumerate(
        zip((plot_gpu_usage, plot_used_power, plot_used_memory), axes.flatten())
    ):
        foo(grouped, ax=ax)
        if i == 0:
            ax.legend()

    fig.savefig(f"nvml_{csv_timestamp}.png", dpi=192)
    pyplot.close(fig)


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
