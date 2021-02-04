#!/usr/bin/env bash

# Initialize the project folders. If folders already present, it deletes them!
# Usage: ./init.sh

rm -rf .bundles/
rm -f fbpic-project.log
rm -f slurm*
rm -f signac.rc
rm -f .signac_sp_cache.json.gz
rm -f signac_project_document.json
rm -f signac_statepoints.json
rm -rf /scratch/berceanu/runs/signac-driven-fbpic/workspace_lwfa/
rm -rf src/__pycache__

python src/init.py
