# GPU Ray Tracing with CUDA

A simple ray tracer program implemented with CPU version and GPU version on CUDA

## Usage

Build:

> make

> make ray_tracer_cpu

> make ray_tracer_gpu

Before building project, check architecture flags to fit one's platform.

Run:

> ./ray_tracer_cpu.exe --ray=100000000 --thread=1

> ./ray_tracer_gpu.exe --ray=100000000 --block=1000 --thread==1000

## Plot

> python3 plot2D.py -f output.gpu.out -n 1000

## Misc

I don't have Nvidia GPU on my device. All debugging and performance tests were done on Midway cluster, The University of Chicago.
