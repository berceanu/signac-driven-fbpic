import pandas as pd
import pathlib
from matplotlib import pyplot

# 'gpu_uuid', 'pid'


def plot_vs_time(df, *, col=None, ylabel=None, ax=None):
    if ax is None:
        ax = pyplot.gca()

    ax.grid()

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time")

    for uuid in df["gpu_uuid"].cat.categories:
        ax.plot("time_stamp", col, data=df, label=uuid)

    ax.legend()

    return ax


def plot_gpu_usage(df):
    ax = plot_vs_time(df, col="GPU_Util_%", ylabel="GPU usage (%)")
    return ax


def plot_used_power(df):
    ax = plot_vs_time(df, col="used_power_W", ylabel="Used power (W)")
    return ax


def plot_used_memory(df):
    ax = plot_vs_time(df, col="used_gpu_memory_MiB", ylabel="GPU MEM usage (MiB)")


def main():
    p = pathlib.Path.cwd() / "nvml.csv"
    df = pd.read_csv(p)
    df["gpu_uuid"] = df["gpu_uuid"].astype("|S40")
    df["gpu_uuid"] = df["gpu_uuid"].astype("category")
    df["time_stamp"] = pd.to_datetime(df["time_stamp"])

    fig, ax = pyplot.subplots()

    # plot_gpu_usage(df)
    # plot_used_power(df)
    plot_used_memory(df)

    fig.savefig("nvml.png")
    pyplot.close(fig)


if __name__ == "__main__":
    main()
