#!/bin/bash
#SBATCH --time=00:05:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --exclusive

module load gcc/6.1

## strong scaling for static scheduling
./ray_tracer_cpu.exe -r 1000000000
