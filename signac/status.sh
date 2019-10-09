#!/usr/bin/env bash

# Check job completion status while jobs are running.
# Usage: ./status.sh

python src/project.py status --pretty --full --stack
