# Kernel8- the rather odd state machine experimentation notation

Written by Gek/DMHSW, March of 2021.


This is a rather odd little way of writing state machines which is itself a sort of 
state machine ISA I thought of.

Here's the gist- "kernels" are finite state-to-state transformation machines.

The primary advantage of kernel code is that solutions written in it often match neatly with real CPU processor
instructions, and if an instruction does not exist, it should be trivial to make one that does in an FPGA.

Kernels take a series of bytes and output a series of bytes of the same length.

The simplest way to explain what kernels are is to give you an example.

This is a 1-byte(state1 or State8) copy kernel (kernelb1 or Kernel8)
which performs an "and" on its input with 127 (0x7f) and returns the result.
```c
state1 and127(state1 c){
	c.state[0] &= 127;
	return c;
}
```

This is a 4-byte (kernelb3 or Kernel32) copy kernel which
returns the input 'c' only if it is prime, otherwise it returns zero.
```c
state3 is_prime(state3 c){
	int32_t val = from_state3(c);
	if(val == 2 || val == 3) return c;
	if(val == 1) return to_state3(0);
	if(val%2 == 0) return to_state3(0);
	for(int32_t i = 3; i < val/2; i+=2){
		if(val%i == 0) return to_state3(0);
	}
	return c;
}
```

copy kernels have the prototype:
```c
stateX kernelname(stateX)
```
Whereas in-place kernels have the prototype:
```c
void kernelname(stateX*)
```
which is more suitable for large states.

The notation is as such (Pay attention to capitalization)

kernelbX is a kernel with 2^(X-1) bytes.

KernelX is a kernel with X bits.

stateX is the amount of state that kernelbX operates on, 2^(X-1) bytes.

StateX is the number of BITS of state that KernelX operates on
```
kernelb1 is Kernel8 which operates on state1 (State8), 1 byte.
kernelb2 is Kernel16 which operates on state2 (State16), 2 bytes.
kernelb3 is Kernel32 which operates on state3 (State32), 4 bytes.
kernelb4 is Kernel64 which operates on state4 (State64), 8 bytes.
```

The code uses kernelbX and stateX notation only, for convenience.

the capitalized notations are not used in-code because they are potentially too confusing,
but I find thinking about problems with the capitalized notations easier.

If you want to speak about it clearly, "32 bit kernel" "32 bit state" "4 byte kernel" "4 byte state"
should work. 




### Inter-state operations

* A KernelX for StateX greater than 8 (state1, kernelb1) can subdivided into sets of any KernelY where 8 <= Y <= X.

Reduction is done either by indexing (mystate7.state2s[0]) or by arbitrary offsets (state_reduce7(mystate7, byteoffset))

You can also use state_highX and state_lowX.



Any KernelY as described above can be "upgraded" or "raised" to KernelX by
"multiplexing", and there are a multitude of schemas to do this.

The code to multiplex kernels is thoroughly demonstrated in the example program, kernel8.c

In many cases, you want to work with an element from an array, and its index into that array.
In those cases, you use KERNEL_MULTIPLEX_INDEXED_POINTER (Non-pointer variants are not written.)

This places the index (as a uint8_t, uint16_t or uint32_t, whatever will fit) into the upper half of the state.

It is not possible with the current implementation to, for instance, 
interpret an array as individual bytes but use a 32 bit index.

The solution in such a case is to access 4 bytes of data at once, multiply the index given by eight, and add a 
"subindex"
```c
//Print individual bytes, with a 32 bit index.
state4 k_printer8ind32(state4 c){
	uint32_t ind = from_state3(k_high4(c))<<2; //We recieved four bytes of data!
	state3 dataseg = k_low4(c);
	uint8_t bytes[4];
	for(size_t i = 0; i < 4; i++) //Subindex
		bytes[i] = from_state1(state_get1_3(&dataseg, i));
	for(uint32_t i = 0; i < 4; i++) //Subindex
		printf("BYTEPRINTER32! %u, %u\n", ind + i, bytes[i]);
}
```

Example of multiplex indexed is provided in kernel8.c

You may want to decide *where* in the array your multiplexed result will go, by providing an index.
This is done with

KERNEL_MULTIPLEX_POINTER_INDEXED_EMPLACE
or alternatively
KERNEL_MULTIPLEX_INDEXED_EMPLACE

Note that these functions cannot be multithreaded or parallelized.

In kernel8.c an example of both is provided.

It is also a common case in computer science to have to perform an nlogn operation

```c
for(int i = 0; i < size - 1; i++)
for(int j = i+1; j < size; j++)
	dosomething(array[i], array[j])
```
I.E. the [0] element is operated on in combination with all the other elements,
then the [1] element is operated on in combination with elements 2 and beyond..

This functionality is provided by KERNEL_MULTIPLEX_NLOGN or KERNEL_MULTIPLEX_NLOGN_POINTER

Parallel variants where the i'th element is read only are provided by KERNEL_MULTIPLEX_NLOGNRO_PARALLEL_POINTER
and
KERNEL_MULTIPLEX_NLOGNRO_PARALLEL

NOTE: If you have OpenMP or OpenACC support on your compiler, you can get
multithreading when running multiplexed kernels.

Additionally, it is possible to make any KernelX work on a KernelY by duplication or mixing of states

Mixing is the general form of taking two StateX's and making them the upper and lower halves of State2X

Duplication is the specific case of mixing where both halves are identical.

The code for this is as such:
```c
statemix3(a,b); //takes two state3's and returns a state4.
statedup3(a); //semantically equivalent to statemix3(a,a)
```



### Type conversion (Type Punning)

You can convert any power-of-two byte object in memory into a kernel (Up to 1 gigabyte)

```c
//most compilers will perform this in-place.
state3 mem_to_state3(void* p)
state4 mem_to_state4(void* p)

//In case it doesn't...
void mem_to_statep3(void* p, state3* a)
void mem_to_statep4(void* p, state4* a)
```

You can convert from signed and unsigned integer types (Up to uint64_t and int64_t if your system supports it)

```c
state3 to_state3(uint32_t)
state3 signed_to_state3(int32_t)

uint32_t from_state3(state3)
int32_t signed_from_state3(state3)
```

You can convert to and from floats and doubles (Doubles are assumed to be 8 bytes)
```c
state3 float_to_state3(float a)
float float_from_state3(state3 q)

state4 double_to_state4(double a)
double double_from_state4(state4 q)
```

You can convert arbitrary byte-offsets into a state variable into another state using memcpy.

```c
memcpy(mystate17.state + 47, mystate10.state, sizeof(mystate10));
```

If you wish to treat an arbitrary StateX as a power-of-two array of StateY,
I've got great news for you!

```c
state9 q = state20.state9s[7];
```

stateX is a union of every possible state less than it (This is why compile times are really large.)


This breaks down very trivially into memcpys, which themselves break down very nicely into assembly.


Please note that the limit of 1 Gigabyte for stateX is a hard limit.
