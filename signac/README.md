# Usage

## Initialize the project's workspace

### from scratch

To initialize the `signac` project from scratch:

```console
$ ./init.sh
```

This will create the folder structure corresponding to `src/init.py`, and
**delete** any existing simulation results (see `init.sh` script for details).

### existing data

If, instead of starting from scratch, one wants to add new simulations to an
existing project, `src/init.py` should be used instead of `init.sh`:

```console
$ python3 src/init.py
```

## Run the project operations

### parallel execution

To run the `fbpic` simulations defined in `src/init.py`, in parallel, using `N`
GPUs and all CPU cores available on the machine:

```console
$ screen -S fbpic
$ ./project.sh N
```

It is convenient to run the project under a `screen` session, as the `fbpic`
simulations might take a few hours to complete.

### serial execution

To run the simulations on a single GPU, in a serial manner, do

```console
$ screen -S fbpic
$ [time] python3 src/project.py run
```

The optional `time` command will give the total runtime once the project
operations are all completed.

## Submit jobs to SLURM

In order to execute the project through the SLURM workload manager, 

```console
$ python src/project.py submit [--pretend]
```

Where available, the command [`nvtop`](https://github.com/Syllo/nvtop) can be
used to check the usage of the machine's GPUs. Otherwise, use `nvidia-smi`.


# Notes

- all commands should be ran from the directory that contains this README file
