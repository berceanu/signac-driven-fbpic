#!/usr/bin/env bash

# Initialize the project folders. If folders already present, it deletes them!
# Usage: ./init.sh

rm -rf .bundles/
rm -f fbpic-project.log
rm -f signac.rc
rm -f .signac_sp_cache.json.gz
rm -f signac_project_document.json
# rm -rf /scratch/berceanu/runs/signac-driven-fbpic/workspace/ FIXME
rm -rf workspace/

python src/init.py
