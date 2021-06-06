CC= gcc
#CC= clang
CFLAGS= -lm -O3 -fopenmp -Wno-unused-function -Wno-absolute-value -fgnu89-inline -std=gnu99 -fno-math-errno

all: main intmath floatmath

main:
	$(CC) kernel8.c $(CFLAGS) other.c -o k8.out 

intmath:
	$(CC)  intmath.c $(CFLAGS) -o int.out

floatmath:
	$(CC)  floatmath.c $(CFLAGS) -o float.out

clean:
	rm -f *.exe *.out *.o
