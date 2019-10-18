#!/usr/bin/env bash

# Run the project operations, in parallel, using at most N GPUs and all CPU cores.
# Usage: ./project.sh N

# parallel on N GPUs
python src/project.py submit --bundle=$1 --parallel --test | /bin/bash

# complete remaining CPU operations, using all cores
python src/project.py run --parallel

