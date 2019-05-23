# `fbpic` + `signac` = 💓

Integration of the [`fbpic`](https://fbpic.github.io) particle-in-cell code with the [`signac`](https://signac.io) data management framework.

## Installation

Download and install the [Anaconda](https://www.anaconda.com) Python 3
Distribution for your OS.

```console
conda env create -f environment.yml
conda activate signac-driven-fbpic
conda list # check installed packages

python minimal_fbpic_script_injection.py # test fbpic
```

## TODO

- [X] create minimal CPU example
- [ ] unit testing
- [ ] integrate 2D plotz library