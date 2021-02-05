#!/usr/bin/env bash

# Run the project operations, in parallel, using at most N GPUs.
# Usage: ./project.sh N

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# parallel on N GPUs
python src/project.py submit -o ex --bundle=$1 --parallel

( sleep 120 ; python src/nvml.py > /dev/null 2>&1 & ) &
