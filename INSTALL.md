# Create fbpic conda environment with dependencies

```console
conda config --add channels conda-forge ✔️
conda create -n signac-driven-fbpic numba==0.42 scipy h5py mkl cudatoolkit=8.0 pyculib ✔️
conda install -n signac-driven-fbpic -c conda-forge mpi4py signac signac-flow signac-dashboard ✔️
pip install fbpic
conda activate signac-driven-fbpic ✔️
conda env export > environment.yml ✔️
```

See [Managing conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more info on using conda environments.
For matching NVIDIA drivers to particular CUDA versions, see [Anaconda's GPU software requirements](https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/#software-requirements).

## Clone your fork from Github, and update it

```console
git clone git@github.com:berceanu/fbpic.git ✔️
cd fbpic ️✔
git checkout dev ✔
git checkout master ✔
```

```console
git remote add upstream https://github.com/fbpic/fbpic.git ✅
git checkout master ✔
git pull --ff-only upstream master ✔
git push origin master ✔
git checkout dev ✔
git pull --ff-only upstream dev ✔
git push origin dev ✔
```

### Install `fbpic`

```console
python setup.py install
python setup.py test # optional
```

**Outcome**: installed `fbpic` in `~/anaconda3/envs/signac-driven-fbpic/lib/python3.6/site-packages/fbpic-0.12.0-py3.6.egg/`.

**Usage**: `python fbpic_script.py`

### Python files

```console
# fbpic scripts
/calder/no_injection/calder_no_injection.py
/calder/high_density/fbpic_script.py
/calder/injection/fbpic_script.py ⭐️

# jupyter notebooks
/cetal/calder_no_injection.py
/cetal/calder_high_density.py

# analysis
/calder/high_density/analysis.py
/calder/injection/analysis.py ⭐️

# crap
/calder/experiment_2012/calder_experiment.py
/cetal/calder_experiment.py
```

**Previous runs** from CETAL server are stored on `ra5_berceanu/runs/fbpic`.