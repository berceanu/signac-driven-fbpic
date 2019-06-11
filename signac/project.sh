#!/usr/bin/env bash

# Usage: ./project.sh N

eval "$(conda shell.bash hook)"
conda activate signac-driven-fbpic

# parallel on N GPUs
python3 src/project.py submit --bundle=$1 --parallel --test | /bin/bash
# complete remaining CPU operations, using all cores
python3 src/project.py run --parallel