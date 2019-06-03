#!/usr/bin/env bash

rm -rf .bundles/
rm -f fbpic-minimal-project.log
rm -f signac.rc
rm -f signac_statepoints.json
rm -rf workspace/

eval "$(conda shell.bash hook)"
conda activate signac-driven-fbpic

python3 src/init.py

