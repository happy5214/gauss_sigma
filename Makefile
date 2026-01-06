CC = gcc
FLAGS = -ftrapv -O3 -g
LIBS = -lm -lgmp -fopenmp

objs = gauss_sigma.o

.PHONY: all clean

all: gauss_sigma

gauss_sigma: $(objs)
	$(CC) -o $@ $^ $(LIBS)

%.o: %.c
	$(CC) -c -o $@ $< $(FLAGS)

clean: 
	rm -f gauss_sigma *.o
