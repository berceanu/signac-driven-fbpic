Download and install the [Anaconda](https://www.anaconda.com) Python 3
Distribution for your OS.

conda create -n signac-driven-fbpic numba==0.42 scipy h5py mkl cudatoolkit=9.0 pyculib matplotlib pylint
conda install -n signac-driven-fbpic -c conda-forge mpi4py signac signac-flow signac-dashboard black

conda activate signac-driven-fbpic
pip install fbpic

python minimal_fbpic_script_injection.py

conda env export > environment.yml