# `fbpic` + `signac` = 💓

Integration of the [`fbpic`](https://fbpic.github.io) particle-in-cell code with the [`signac`](https://signac.io) data management framework.

```console
conda env create -f environment.yml
conda activate signac-driven-fbpic
conda list # check installed packages
```

## TODO

- [X] create minimal CPU example
- [X] install `autopep8` and `pylint` in `signac-driven-fbpic` environment
- [X] install `matplotlib`
- [X] install `black` python formatter
- [X] make the `defaults` channel top priority over `conda-forge`
- [ ] update the `environment.yml` file
- [ ] unit testing
- [ ] integrate 2D plotz library
- [ ] sync/upload vscode from home