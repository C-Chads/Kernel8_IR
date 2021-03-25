CC= gcc
CFLAGS= -O3

all:
	$(CC) $(CFLAGS) kernel8.c -o k8.out -fopenmp -lm -g

clean:
	rm -f *.exe *.out *.o
