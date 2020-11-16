#!/usr/bin/env bash

# Run the project operations, in parallel, using at most N GPUs.
# Usage: ./project.sh N

# parallel on N GPUs
python src/project.py submit -o ex --bundle=$1 --parallel --test | /bin/bash

