#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclusive

module load cuda

./ray_tracer_gpu.exe --thread=100 --block=100 --ray=10000
./ray_tracer_gpu.exe --thread=100 --block=1000 --ray=100000
./ray_tracer_gpu.exe --thread=100 --block=1000 --ray=1000000
./ray_tracer_gpu.exe --thread=100 --block=1000 --ray=10000000
./ray_tracer_gpu.exe --thread=100 --block=1000 --ray=100000000
./ray_tracer_gpu.exe --thread=100 --block=1000 --ray=1000000000
