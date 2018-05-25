#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive

module load cuda

## Optimized comfigurations experiment against 2^20 * 100 total tasks

## GPU: throughput 51200
./ray_tracer_gpu.exe --ray=104857600 --thread=32 --block=1600
