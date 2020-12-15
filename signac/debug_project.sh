#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=15
python src/project.py exec save_pngs 548df05f5de26d318d9481bdfae35fb4 --show-traceback
