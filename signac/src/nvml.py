import pynvml
import pathlib

def main():
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()

    version = pynvml.nvmlSystemGetDriverVersion().decode('UTF-8')
    print(f"Driver Version: {version}")

    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)

        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        uuid = pynvml.nvmlDeviceGetUUID(handle).decode('UTF-8')

        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if procs:
            print(f"{uuid}: {info.used / 1024 / 1024}")
            for p in procs:
                name = pynvml.nvmlSystemGetProcessName(p.pid)
                p_path = pathlib.Path(name.decode('UTF-8'))
                mem_MiB = p.usedGpuMemory / 1024 / 1024
                print(p_path.stem, mem_MiB)
                if p_path.stem == "python":
                    print(f"{uuid}: {mem_MiB} MiB")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
