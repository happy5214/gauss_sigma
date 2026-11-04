CC = g++
FLAGS = -ftrapv -O3 -g -fpermissive -std=c++20
LIBS = -lm -fopenmp

objs = gauss_sigma.o modular.o primes.o arg_parser.o

.PHONY: all clean

all: gauss_sigma

gauss_sigma: $(objs)
	$(CC) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(FLAGS)

clean: 
	rm -f gauss_sigma *.o
