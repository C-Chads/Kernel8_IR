CC= gcc
CFLAGS= -O3 -fopenmp -Wall

all:
	$(CC) $(CFLAGS) *.c -o k8.out 

clean:
	rm -f *.exe *.out *.o
