# create fbpic conda environment with dependencies

```bash
conda config --add channels conda-forge ✔️
conda create -n signac-driven-fbpic numba==0.42 scipy h5py mkl cudatoolkit=8.0 pyculib ✔️
conda install -n signac-driven-fbpic -c conda-forge mpi4py signac signac-flow signac-dashboard ✔️
conda activate signac-driven-fbpic ✔️
```

[Managing conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

# clone your fork from Github, and update it 

```
$ git clone git@github.com:berceanu/fbpic.git
$ cd fbpic
$ git checkout dev
$ git checkout master
```

```
$ git remote add upstream https://github.com/fbpic/fbpic.git # only first time
$ git pull --ff-only upstream master
$ git push origin master
$ git checkout dev
$ git pull --ff-only upstream dev
$ git push origin dev
```

```
$ python setup.py install
$ python setup.py test
```

# installed fbpic `0.12.0` in
`/home/berceanu/anaconda3/envs/fbpic/lib/python3.6/site-packages/fbpic-0.12.0-py3.6.egg/`

## Usage
`$ python fbpic_script.py`


## CETAL fbpic runs

| filename              | status    |
|-----------------------|-----------|
|post_cetal.py          |  NOT FOUND|
|post_injection.py      |  NOT FOUND|
|cetal.py               |  NOT FOUND|
|calder_injection.py    |  NOT FOUND|
|calder_high_density.py |  NOT FOUND|


## ~/Development/fbpic/docs/source/example_input ##

/calder/no_injection/calder_no_injection.py
/calder/high_density/fbpic_script.py

JUPYTER NOTEBOOKS
/cetal/calder_no_injection.py
/cetal/calder_high_density.py

ANALYSIS
/calder/high_density/analysis.py

CRAP
/calder/experiment_2012/calder_experiment.py
/cetal/calder_experiment.py

# symbolic link to `.h5` files
/cetal/diags -> /Date1/andrei/runs/fbpic/cetal/diags

#################################
### We chose as example case: ###
#################################

BASE_PATH = /home/berceanu/runs/fbpic/docs/source/example_input/calder/injection

$BASE_PATH/fbpic_script.py
$BASE_PATH/analysis.py
  