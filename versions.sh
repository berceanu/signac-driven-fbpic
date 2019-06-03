#!/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate signac-driven-fbpic

./versions.py
