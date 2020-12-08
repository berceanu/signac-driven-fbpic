#!/usr/bin/env bash

rsync -avP --exclude "*.h5" thorviaheimdall:/scratch/berceanu/runs/signac-driven-fbpic/workspace/ workspace
