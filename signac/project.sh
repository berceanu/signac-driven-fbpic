#!/usr/bin/env bash

# Submit SLURM job for running the project operations
# Usage: ./project.sh [N]
# The number of bundles N can be added via `--bundle=$1` and should be chosen as follows:
# - without MPI: ngpu = 1 always, so N should be set to min(num of runnable signac GPU jobs, num of GPUs on machine)
#                N signac jobs will run in parallel, using 1 GPU each
#                max N is the number of GPUs on the machine, eg 16
# - with MPI: eg. if nranks = ngpu = 4, N = 2 would submit a single job reserving 2 * 4 = 8 GPUs
#             2 signac jobs will run in parallel, using 4 GPUs each
#             max N is the number of GPUs on the machine / nranks, eg 16 / 4 = 4

python src/project.py submit -o preprocessing --parallel
python src/project.py submit -o fbpic --parallel
python src/project.py submit -o postprocessing --parallel
