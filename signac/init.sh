#!/usr/bin/env bash

# Initialize the project folders. If folders already present, it deletes them!
# Usage: ./init.sh

rm -rf .bundles/
rm -f fbpic-project.log
rm -f signac.rc
rm -f signac_statepoints.json
rm -f .signac_sp_cache.json.gz
rm -rf workspace/

eval "$(conda shell.bash hook)"
conda activate signac-driven-fbpic

python3 src/init.py
