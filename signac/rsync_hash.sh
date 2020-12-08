#!/usr/bin/env bash

rsync -avP --exclude "*.h5" thorviaheimdall:/data/storage/berceanu/Development/signac-driven-fbpic/signac/runs/ runs
