CC= gcc
CFLAGS= -O3 -fopenmp -Werror

all:
	$(CC) $(CFLAGS) kernel8.c -o k8.out 

clean:
	rm -f *.exe *.out *.o
