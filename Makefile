CC= gcc
#CC= clang
CFLAGS= -O3 -lm -fopenmp -Wno-unused-function -Wno-absolute-value -fgnu89-inline -std=gnu99 -fno-math-errno

all: main intmath floatmath

main:
	$(CC) $(CFLAGS) kernel8.c other.c -o k8.out 

intmath:
	$(CC) $(CFLAGS) intmath.c -o int.out

floatmath:
	$(CC) $(CFLAGS) floatmath.c -o float.out

clean:
	rm -f *.exe *.out *.o
