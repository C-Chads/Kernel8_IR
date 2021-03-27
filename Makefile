CC= gcc
CFLAGS= -O3 -fopenmp -Wno-unused-function

all:
	$(CC) $(CFLAGS) *.c -o k8.out 

clean:
	rm -f *.exe *.out *.o
