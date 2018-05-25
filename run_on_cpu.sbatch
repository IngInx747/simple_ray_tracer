#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=sandyb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

#module load gcc/6.1

## strong scaling for static scheduling
./ray_tracer_cpu.exe --thread=1 --ray=10000
./ray_tracer_cpu.exe --thread=1 --ray=100000
./ray_tracer_cpu.exe --thread=1 --ray=1000000
./ray_tracer_cpu.exe --thread=1 --ray=10000000
./ray_tracer_cpu.exe --thread=1 --ray=100000000
./ray_tracer_cpu.exe --thread=1 --ray=1000000000