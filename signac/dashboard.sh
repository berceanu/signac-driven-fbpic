#!/usr/bin/env bash

# Launch the web interface for result visualization.
# Usage: ./dashboard.sh

eval "$(conda shell.bash hook)"
conda activate signac-driven-fbpic

python3 src/dashboard.py run --port 7777