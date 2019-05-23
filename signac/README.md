# Usage

```console
conda activate signac-driven-fbpic

export FBPIC_DISABLE_THREADING=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export OMP_NUM_THREADS=1

python3 src/init.py
python3 src/project.py run --parallel
python3 src/project.py status --pretty --full --stack
python3 src/dashboard.py run --host 0.0.0.0 --port 7777 
```
