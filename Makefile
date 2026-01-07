CC = g++
FLAGS = -ftrapv -O3 -g -fpermissive
LIBS = -lm -fopenmp

objs = gauss_sigma.o

.PHONY: all clean

all: gauss_sigma

gauss_sigma: $(objs)
	$(CC) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(FLAGS)

clean: 
	rm -f gauss_sigma *.o
