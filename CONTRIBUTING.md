## Guideline for Code Contributions

* Use the [OneFlow](https://www.endoflineblog.com/oneflow-a-git-branching-model-and-workflow) model of development:
  - Both new features and bug fixes should be developed in branches based on `master`.
  - Hotfixes (critical bugs that need to be released *fast*) should be developed in a branch based on the latest tagged release.


## Installation

### NVIDIA Drivers

For matching NVIDIA drivers to particular CUDA versions, see [CUDA Toolkit and Compatible Driver Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#major-components__table-cuda-toolkit-driver-versions).
Which GPUs are supported by the [latest NVIDIA drivers](https://www.nvidia.com/object/unix.html) and how to install the latest driver on [Ubuntu 18.04](https://www.linuxbabe.com/ubuntu/install-nvidia-driver-ubuntu-18-04).


```bash
$ sudo ubuntu-drivers devices
$ sudo ubuntu-drivers autoinstall
$ sudo apt install cuda-drivers-fabricmanager-455
$ sudo systemctl unmask nvidia-fabricmanager
$ sudo systemctl enable nvidia-fabricmanager
```

### Python distro

```bash
$ wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh
$ bash Mambaforge-Linux-x86_64.sh -b -p $HOME/software/pkg/miniforge3pic
$ mkdir -p $HOME/MyModules/miniforge3pic
$ cd $HOME/MyModules/miniforge3pic
$ wget https://raw.githubusercontent.com/CHPC-UofU/anaconda-modules/master/miniconda3/latest.lua
# change myanapath to software/pkg/miniforge3pic inside latest.lua
```

```bash
# ~/custom.sh
#!/bin/bash

module use $HOME/MyModules
module load miniforge3pic/latest
```

### Dependencies 

```bash
$ git clone git@github.com:berceanu/fbpic.git
$ cd fbpic/
$ git checkout dev
$ git remote add upstream https://github.com/fbpic/fbpic.git
$ git checkout dev
$ git pull --ff-only upstream dev
$ git push origin dev
```

```bash
$ source ~/custom.sh
$ mamba install -c conda-forge six wheel fastrlock python-dateutil unyt numba scipy matplotlib pandas h5py openpmd-api mkl cupy pyopencl ocl-icd-system signac signac-flow signac-dashboard pylint black rope pynvml schedule
```

```bash
$ cd Development/fbpic/
$ git checkout dev
$ python -m pip install .
```
