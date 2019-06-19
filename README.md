# `fbpic` + `signac` = 💓

Integration of the [`fbpic`](https://fbpic.github.io) particle-in-cell code with the [`signac`](https://signac.io) data management framework.

## Requirements

- NVIDIA Driver Version >= 410.104
- `ffmpeg`

## Installation

Download and install the [Anaconda](https://www.anaconda.com) (or Miniconda) Python 3
Distribution for your OS.

*Note*: in case you have an older version of the `signac-driven-fbpic` `conda` environment
already, remove it with `conda remove --name signac-driven-fbpic --all`.

```console
$ conda env create -f environment.yml
$ conda activate signac-driven-fbpic

# test `fbpic`
$ python3 minimal_fbpic_script.py
```

Installation instructions for developers can be found in [`devINSTALL.md`](https://github.com/berceanu/signac-driven-fbpic/blob/master/devINSTALL.md).

## Usage

See [`signac/README.md`](https://github.com/berceanu/signac-driven-fbpic/blob/master/signac/README.md).

## Todo

- [ ] convert to `.rst` and add screen captures -- see [`README.rst`](https://github.com/hansec/fortran-language-server/blob/master/README.rst)
