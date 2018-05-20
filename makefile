
SRC = src
INCLUDE = include
OBJ = obj

## Compiler
CPUCOMPILER = gcc
GPUCOMPILER = nvcc
GPULINKER = nvlink

## Flags
HEADFLAG = -I"$(INCLUDE)/"
OMPFLAG = -fopenmp
XPTFLAG = -Xcompiler
ARCHFLAG = -arch=sm_30

##
CC = $(CPUCOMPILER) $(HEADFLAG) $(OMPFLAG)
NV = $(GPUCOMPILER) $(HEADFLAG)
NL = $(GPULINKER) $(ARCHFLAG)

all: ray_tracer_cpu ray_tracer_gpu

ray_tracer_cpu: ray_tracer_cpu.o vec.o
	$(CC) $(OBJ)/*.o -o $@.exe -lm

ray_tracer_gpu: ray_tracer_gpu.co vec_gpu.co
	$(NL) $(OBJ)/*.co -o $@.exe

ray_tracer_cpu.o: $(SRC)/ray_tracer_cpu.c
	$(CC) -c $(SRC)/ray_tracer_cpu.c -o $(OBJ)/$@ -lm

ray_tracer_gpu.co: $(SRC)/ray_tracer_gpu.cu
	$(NV) -dc $(SRC)/ray_tracer_gpu.cu -o $(OBJ)/$@

vec.o: $(SRC)/vec.c $(INCLUDE)/vec.h
	$(CC) -c $< -o $(OBJ)/$@

vec_gpu.co: $(SRC)/vec_gpu.cu $(INCLUDE)/vec_gpu.h
	$(NV) -dc $< -o $(OBJ)/$@

clean:
	rm *.exe $(OBJ)/*.o $(OBJ)/*.co
