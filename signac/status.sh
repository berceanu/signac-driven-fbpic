#!/usr/bin/env bash

# Check job completion status while jobs are running.
# Usage: ./status.sh

eval "$(conda shell.bash hook)"
conda activate signac-driven-fbpic

python3 src/project.py status --pretty --full --stack