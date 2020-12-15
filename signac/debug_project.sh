#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=15
python src/project.py run -o ex --show-traceback
