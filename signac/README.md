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
$ python src/init.py
```

## Run the project operations

### parallel execution

To submit a `SLURM` job for `N` parallel `fbpic` simulations:

```console
$ ./project.sh N
```

### post-processing

```console
$ watch -c gpustat -cp --color
```

```console
$ python src/dashboard.py run
```

```console
# copy outputs to runs/ folder
$ python src/copy_with_hash.py

# create bunch_centroid/ 
$ python src/centroid.py

$ python src/paper_figures.py
```

# Notes

- all commands should be ran from the directory that contains this README file
