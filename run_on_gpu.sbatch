#!/bin/bash
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive

module load cuda

## strong scaling for static scheduling
./ray_tracer_gpu.exe --ray=1000000000 --block=10 --thread=1000
