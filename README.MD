# Kernel8- the rather odd state machine experimentation notation

Written by Gek/DMHSW, March of 2021.


This is a rather odd little way of writing state machines which is itself a sort of 
state machine ISA I thought of.

Here's the gist- "kernels" are finite state-to-state transformation machines.

The primary advantage of kernel code is that solutions written in it often match neatly with real CPU processor
instructions, and if an instruction does not exist, it should be trivial to make one that does in an FPGA.

Kernels take a series of bytes and output a series of bytes of the same length.

The simplest way to explain what kernels are is to give you an example.

This is a 1-byte(state1 or State8) kernel (kernelb1 or Kernel8)
which performs an "and" on its input with 127 (0x7f) and returns the result.
```c
state1 and127(state1 c){
	c.state[0] &= 127;
	return c;
}
```

This is a 4-byte (kernelb3 or Kernel32) kernel which
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

* A KernelX for StateX greater than 8 can subdivided into sets of any KernelY where 8 <= Y <= X.

Any KernelY as described above can be "upgraded" or "raised" to KernelX by
"multiplexing", and there are a multitude of schemas to do this.

The code to multiplex a kernel is as such
```c
//Multiplex the and127 1-byte kernel to state3.
//the resulting function is state3 and127_mt3(state3)
//This will be multithreaded if you have OpenMP or OpenACC
KERNEL_MULTIPLEX(and127, 1, 3);
//Ditto, but with SIMD, only on OpenMP
KERNEL_MULTIPLEX_SIMD(and127, 1, 3);

//Same as above but with pass-by-pointer semantics rather than pass-by-value.
//the resulting function is void and127_mtp3(state3*)
KERNEL_MULTIPLEX_POINTER(and127, 1, 3)

//Ditto, but with SIMD, only on OpenMP
KERNEL_MULTIPLEX_POINTER_SIMD(and127, 1, 3)

//Nonparallel variants
//These are extremely useful for filling arrays with data, printing to standard out, etcetera.
//produces and127_nopara_mt3
KERNEL_MULTIPLEX_NOPARA(and127, 1, 3)
//produces and127_nopara_mtp3
KERNEL_MULTIPLEX_POINTER_NOPARA(and127, 1, 3)


```

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

Example of multiplex indexed.

```c
//High state3 is the index, Low state3 is the data at that index.
//As a reminder, state3 is 4 bytes, state4 is 8 bytes.
state4 k_fillerind(state4 c){ //Real kernel using the "MultiplexIndexed" syntax.
	state4 ret;
	uint32_t index = from_state3(k_high4(c));
	ret = statemix3(state3_zero(), to_state3(index));
	return ret;
}
//Declare multiplexing functions for k_fillerind for state20 and state30.
//macro args are function name, state to interpret as,
//size larger state (MUST be the second argument +1...)
//and finally, the array state size.
KERNEL_MULTIPLEX_INDEXED_POINTER(k_fillerind, 3, 4, 20);
KERNEL_MULTIPLEX_INDEXED_POINTER(k_fillerind, 3, 4, 30);
```

You may want to decide *where* in the array your multiplexed result will go, by providing an index.
This is done with

KERNEL_MULTIPLEX_POINTER_INDEXED_EMPLACE(func, nn, nnn, nm)
or alternatively
KERNEL_MULTIPLEX_INDEXED_EMPLACE(func, nn, nnn, nm)

Note that these functions cannot be multithreaded or parallelized.

In kernel8.c an example of both is provided.

```c
state4 k_modsort(state4 c){ //Sort by modulo.
	//uint32_t index = from_state3(k_high4(c));
	uint32_t val = from_state3(k_low4(c));
	return statemix3(to_state3(val), to_state3(val));
}

//Emplacing modsort
KERNEL_MULTIPLEX_INDEXED_EMPLACE(k_modsort, 3, 4, 20);
KERNEL_MULTIPLEX_POINTER_INDEXED_EMPLACE(k_modsort, 3, 4, 20);

//Later...
s20 = k_modsort_mtpi20(s20);
//or
k_modsort_mtpie20(&s20);
```

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

If you wish to treat an arbitrary StateX as a power-of-two array of StateY, you can generate array insertion
and retrieval functions like this:

```c
//Functions to operate on state20 as if it was an array of state3's.

//Retrieve element at index
state3 state_get3_20 (state20* input, long long index)
//Retrieve element at index, use pointers
void state_get3_20 (state20* input, state3* ret, long long index)

//Insert at index
void state_insert3_20 (state20* targ, state3 val, long long index)
//insert by pointer.
void state_insertp3_20 (state20* targ, state3* val, long long index)

SUBSTATE_ARRAY(3, 20);
```

This breaks down very trivially into memcpys, which themselves break down very nicely into assembly.

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

You can convert arbitrary byte-offsets into a state variable into another state.
(This is effectively a shorthand for memcpy)
```c
state3 state_get_byteoffset3_12 (state12* input, size_t index)
state7 state_get_byteoffset7_30 (state30* input, size_t index)
void state_getp_byteoffset8_30 (state30* input, state8* ret, size_t index)
```

You can insert a state at an arbitrary offset into another state.
(This is effectively a shorthand for memcpy)
```c
void state_insertp_byteoffset3_7(state7* targ, state3* val, size_t index)
void state_insert_byteoffset9_28 (state28* targ, state9 val, size_t index)
```
