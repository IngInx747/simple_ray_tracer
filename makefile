
SRC = src
INCLUDE = include
OBJ = obj

## Compiler
CPUCOMPILER = gcc
GPUCOMPILER = nvcc

## Flags
HEADFLAG = -I"$(INCLUDE)/"
OMPFLAG = -fopenmp
XPTFLAG = -Xcompiler
ARCHFLAG = -arch=sm_30

##
CC = $(CPUCOMPILER) $(HEADFLAG) $(OMPFLAG)
NV = $(GPUCOMPILER) $(HEADFLAG) $(ARCHFLAG)

all: ray_tracer_cpu ray_tracer_gpu

ray_tracer_cpu: ray_tracer_cpu.o vec_cpu.o
	$(CC) $(OBJ)/*_cpu.o -o $@.exe -lm

ray_tracer_gpu: ray_tracer_gpu.o vec_gpu.o
	$(NV) $(OBJ)/*_gpu.o -o $@.exe

ray_tracer_cpu.o: $(SRC)/ray_tracer_cpu.c
	$(CC) -c $(SRC)/ray_tracer_cpu.c -o $(OBJ)/$@ -lm

ray_tracer_gpu.o: $(SRC)/ray_tracer_gpu.cu
	$(NV) -dc $< -o $(OBJ)/$@

vec_cpu.o: $(SRC)/vec.c $(INCLUDE)/vec.h
	$(CC) -c $< -o $(OBJ)/$@

vec_gpu.o: $(SRC)/vec_gpu.cu $(INCLUDE)/vec_gpu.h
	$(NV) -dc $< -o $(OBJ)/$@

clean:
	rm *.exe $(OBJ)/*.o
