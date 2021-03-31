CC= gcc
CFLAGS= -O3 -lm -fopenmp -Wno-unused-function -std=gnu11 -finline-limit=64000 -fno-math-errno

all: main intmath floatmath

main:
	$(CC) $(CFLAGS) kernel8.c other.c -o k8.out 

intmath:
	$(CC) $(CFLAGS) intmath.c -o int.out

floatmath:
	$(CC) $(CFLAGS) floatmath.c -o float.out

clean:
	rm -f *.exe *.out *.o
