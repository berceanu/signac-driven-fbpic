"""
Periodically polls all the GPUs and writes usage info
for all running Python processes to disk in CSV format.
Usage: python src/nvml.py > /dev/null 2>&1 &
"""
import pynvml
import pathlib
import datetime
import time
import schedule

# Accomodate more than one `fbpic` run per GPU
# Accomodate PIConGPU usecase => track all 16 GPUs; don't search for Python processes only

def job():
    print("Executing..")

schedule.every(10).seconds.do(job)


def is_python_process(pid):
    """Checks process identified by PID is a python process."""
    name = pynvml.nvmlSystemGetProcessName(pid)
    p = pathlib.Path(name.decode("UTF-8"))
    return p.stem == "python"


def is_python_running(number_of_gpus):
    """Return True if there's *at least 1* running python process on *any* of the GPUs."""
    for gpu in range(number_of_gpus):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in processes:
            if is_python_process(p.pid):
                return True
    return False


# TODO adapt for PIConGPU / general non-python processes
# TODO add other metrics like temperature, throttling state and/of clock freq


def main():
    """Main entry point."""
    pynvml.nvmlInit()
    gpu_count = pynvml.nvmlDeviceGetCount()

    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_file = pathlib.Path.cwd() / f"nvml_{now}.csv"

    with out_file.open("w") as f:
        f.write("time_stamp,gpu_uuid,pid,used_gpu_memory_MiB,used_power_W,GPU_Util_%\n")

    while is_python_running(gpu_count):
        time.sleep(10)
        for gpu in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for p in processes:
                if is_python_process(p.pid):
                    mem_MiB = p.usedGpuMemory / 1024 / 1024

                    time_stamp = datetime.datetime.now()
                    uuid = pynvml.nvmlDeviceGetUUID(handle).decode("UTF-8")
                    pow_draw_watt = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    gpu_util_percentage = pynvml.nvmlDeviceGetUtilizationRates(
                        handle
                    ).gpu

                    with out_file.open("a") as f:
                        f.write(
                            f"{time_stamp},{uuid},{p.pid},{mem_MiB:g},{pow_draw_watt},{gpu_util_percentage}\n"
                        )

    pynvml.nvmlShutdown()

    while True:
        schedule.run_pending()

if __name__ == "__main__":
    main()
