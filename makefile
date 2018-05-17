
SRC = src
INCLUDE = include
OBJ = obj

COMPILER = gcc
HEADFLAG = -I"$(INCLUDE)/"
OMPFLAG = -fopenmp
CC = $(COMPILER) $(HEADFLAG) $(OMPFLAG)

all: ray_tracer_serial

ray_tracer_serial: $(SRC)/ray_tracer_serial.c vec.o
	$(CC) $(SRC)/ray_tracer_serial.c $(OBJ)/*.o -o $@.exe -lm

vec.o: $(SRC)/vec.c $(INCLUDE)/vec.h
	$(CC) -c $(SRC)/vec.c -o $(OBJ)/$@

clean:
	rm *.exe $(OBJ)/*.o
