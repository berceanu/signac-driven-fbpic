# Create fbpic conda environment with dependencies

Download and install the [Anaconda](https://www.anaconda.com) Python 3
Distribution for your OS.


```console
conda create -n signac-driven-fbpic -c defaults numba scipy h5py mkl cudatoolkit=10.0 matplotlib pandas
conda install -n signac-driven-fbpic -c conda-forge mpi4py signac signac-flow signac-dashboard unyt
# conda install -n signac-driven-fbpic -c rlehe openpmd_viewer

conda activate signac-driven-fbpic
pip install cupy-cuda100 openPMD-viewer fbpic

conda env export > environment.yml
```

See [Managing conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for more info on using conda environments.
For matching NVIDIA drivers to particular CUDA versions, see [Anaconda's GPU software requirements](https://docs.anaconda.com/anaconda/user-guide/tasks/gpu-packages/#software-requirements).
Which GPUs are supported by the [latest NVIDIA drivers](https://www.nvidia.com/object/unix.html) and how to install the latest driver on [Ubuntu 18.04](https://askubuntu.com/questions/1054954/how-to-install-nvidia-driver-in-ubuntu-18-04).

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
python setup.py develop
python setup.py test # optional
```

**Outcome**: installed `fbpic` in `~/anaconda3/envs/signac-driven-fbpic/lib/python3.6/site-packages/fbpic-0.12.0-py3.6.egg/`.

**Usage**: `python3 fbpic_script.py`

### Python files

```console
# fbpic scripts
calder/no_injection/calder_no_injection.py
calder/high_density/fbpic_script.py
calder/injection/fbpic_script.py ⭐️

# jupyter notebooks
cetal/calder_no_injection.py
cetal/calder_high_density.py

# analysis
calder/high_density/analysis.py
calder/injection/analysis.py ⭐️

# crap
calder/experiment_2012/calder_experiment.py
cetal/calder_experiment.py
```

**Previous runs** from CETAL server are stored on `ra5_berceanu/runs/fbpic`.
