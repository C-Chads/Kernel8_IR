Kernel8 Intermediate Representation Language


High performance functional programming for C and intermediate representation for higher level code.




### What is a kernel?

Under Kernel8-IR, a kernel is a state transformation function.

A function is a kernel if...

* it has no global state of any kind which is relevant to the operation of the program.

* its output is exactly the same every single time that it is invoked.

* it could be replaced with a (potentially very large) lookup table.

A very simple kernel would be "and127"

(kernelpb syntax)
```c
unsigned char and127(unsigned char *c){
	*c = *c & 127;
}
```

or with pass-by-value semantics...

(kernelb syntax)
```c
unsigned char and127(unsigned char c){
	c &= 127;
	return c;
}
```

and127 is a function that takes in one byte of state, modifies it, and returns it.

it can very easily be represented with a lookup table of size 256.

0 returns 0, 1 returns 1,... 127 returns 127, 128 returns 0, 129 returns 1 ... 255 returns 127.

Therefore, and127 is a valid kernel.

It even has the special property of being defined for every possible input- a "Complete kernel"

All such functions which operate on 1 byte of state could be replaced by 256 byte lookup tables.

There are 256^2 such mappings, or 65536 different possible 1-byte kernels.

Kernels are classified by the minimum power-of-two number of bytes needed to encapsulate their input and output.

kernelb1 operates on 1 byte of state
kernelb2 operates on 2 bytes of state
kernelb3 operates on 4 bytes of state
kernelb4 operates on 8 bytes of statae...

however, most kernels (the most flexible and best for compiler optimization) are actually "kernelpb"

which means they use pass-by-pointer semantics.

### What makes kernels so special? Aren't they just functions?

Kernels have special properties

* Erroneous inputs into a kernel can be determined quite trivially and propagated through callers.

it should be possible to write a static analyzer that does this for programs written from small kernels

* It is extremely easy to inline and optimize code written with kernels

* It is extremely trivial to parallelize kernel operations
