#!/bin/bash

#SBATCH --partition=gpu	      	  

#SBATCH --ntasks=2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --gpus=2

#SBATCH --gpus-per-task=1

srun --gres=gpu:1 bash -c 'CUDA_VISIBLE_DEVICES=$SLURM_PROCID env' | grep CUDA_VISIBLE

# We want 2 processes (ntasks) to stay on the same
# node (tasks-per-node = ntasks).
# Each process can use 4 cores for multithreading
# (cpus-per-task), for a total of 8 cores.
# We want 2 GPUs (gpu:2), each process should use
# 1 GPU (gpus-per-task)
#  If your program is a parallel MPI program,
#  srun takes care of creating all the MPI processes.
#  If not, srun will run your program as many times
#  as specified by the --ntasks option.
