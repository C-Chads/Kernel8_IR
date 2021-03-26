//DMHSW's "Kernel 8"
//A notation for state machine computing which reflects underlying hardware.
//Unknown if it will actually be useful for any computer nerds out there,
//but it's kinda cool

/*
********
What is a kernel?
********
A finite state transformation.
It maps from a finite domain of states into a finite range of states.

A kernel takes in a power-of-2 number of bytes and outputs some result of the same size.


Every single kernel can be replaced by a lookup table given the user has enough storage space to accomodate it.

The fundamental and most important part of a kernel is that it MUST ***ALWAYS*** operate ***ONLY*** on
the state it is passed- there are no "global variables"


Kernels are named by the following convention
kernelN is a kernel which operates on N *BITS*. not bytes, BITS.
kernelbN is a kernel which operates on 2^(n-1) *BYTES*. 
kernelpb is kernelbN, but with pass-by-pointer syntax. This is more expensive
for small states (<64 bytes) but absolutely essential with large states.

kernelb1 operates on a single byte. it is equivalent to "Kernel8"

kernelb10 operates on 512 bytes. it is equivalent to "Kernel4096"



******
What is a state?
******
Bits on your computers, in groupings of bytes.

state1 is a  1 byte state.
state2 is a  2 byte state.
state3 is a  4 byte state.
state4 is an 8 byte state.
state5 is a 16 byte state...

A kernel operates ONLY Using the state it is passed, and nothing else.

It will never return a different result if given the same state.

if it violates this rule, it is not a valid kernel, it is an ordinary function.

You can write a function which matches the syntax of a kernel (this might be useful)

but it is *not* a kernel even if the type system says it is.

*******
Known special properties of kernels
*******
1) Kernels are extremely trivial to optimize at compile time. 
2) Multithreading the operation of kernels is trivial since they never use external state.
3) Every state machine can be represented as a kernel. A single-threaded program can be encapsulated in one.
4) You can propagate error conditions and edge cases through a kernel tree.
5) Code written as kernel code is very often a direct mapping to real hardware instructions.
	This means that getting SIMD acceleration with portable code is extremely easy- just write the code
	that does what the simd instruction does in kernel code, and it will probably compile into the simd instruction.
	States are aligned to memory boundaries matching their size (Until 16, which is the largest useful on my machine)
6) Kernels are *already* serialized
	Big note:
	The endianness of your computer may affect serialization. use k_endian_cond_swap
	if you are trying to transfer state from one computer to another and you are taking advantage of
	a hardware floating point unit, you may find that your results vary with that as well.
7) C library code written as a kernel is incredibly portable (No external variables!)
8) it should be possible to write a static analyzer for your kernel code which can
	detect all possible errors by propagating edge cases. (This is something I am still thinking about)

	You might say "this is true for any finite state machine!" and you would be correct.
	However, even for a 128 bit arbitrary state machine, it would take centuries to
	brute force all 2^128 possible cases.

	With kernel code, there is one huge trick

	Since a kernel can be replaced with a lookup table, we can propagate domain restrictions
	into range restrictions. We can define contiguous "regions" of states to be possible while others impossible,
	and some erroneous while others not.	
9)  Functional programming paradigms- kernels have no side effects.
10) Kernel code can be created in ASIC/FPGA hardware, making this a form of HDL.

*/

#ifndef KERNELN_H
#define KERNELN_H

#if defined(_OPENMP)
#define PRAGMA_PARALLEL _Pragma("omp parallel for")
#define PRAGMA_SIMD _Pragma("omp simd")
#else
#define PRAGMA_PARALLEL _Pragma("acc loop")
#define PRAGMA_SIMD /*a comment*/
#endif

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
//Kernel8 = KernelB
//Kernel16 = KernelB2
#ifndef KERNEL_NO_ALIGN
#include <stdalign.h>
#define KERNEL_ALIGN(n) alignas(n)
#else
#define KERNEL_ALIGN(n) /*a comment*/
#endif

//define a 2^(n-1) byte state, and kernel type,
//as well as common operations.
#define KERNELB_NO_OP(n, alignment)\
typedef struct{\
  KERNEL_ALIGN(alignment) uint8_t state[(size_t)1<<(n-1)];\
} state##n;\
typedef state##n (* kernelb##n )( state##n);\
typedef void (* kernelpb##n )( state##n*);\
static inline state##n state##n##_zero() {state##n a = {0}; return a;}\
static inline state##n mem_to_state##n(void* p){state##n a; memcpy(a.state, p, 1<<(n-1)); return a;}\
static inline void mem_to_statep##n(void* p, state##n *a){memcpy(a->state, p, 1<<(n-1));}

#define KERNELB(n, alignment)\
KERNELB_NO_OP(n, alignment)\
/*perform the operation between the two halves and return it*/\
static state##n k_and##n (state##n a){\
	state##n ret;\
	for(long long i = 0; i < (1<<(n-1))/2; i++)\
		ret.state[i] = a.state[i] & a.state[i + (1<<(n-1))/2];\
	return ret;\
}\
static state##n k_or##n (state##n a){\
	state##n ret;\
	for(long long i = 0; i < (1<<(n-1))/2; i++)\
		ret.state[i] = a.state[i] | a.state[i + (1<<(n-1))/2];\
	return ret;\
}\
static state##n k_xor##n (state##n a){\
	state##n ret;\
	for(long long i = 0; i < (1<<(n-1))/2; i++)\
		ret.state[i] = a.state[i] ^ a.state[i + (1<<(n-1))/2];\
	return ret;\
}\
static state##n k_swap##n (state##n a){\
	state##n ret;\
	for(long long i = 0; i < (1<<(n-1)); i++){\
		ret.state[i] = a.state[(1<<(n-1))-1-i];\
	}\
	return ret;\
}\
static state##n k_endian_cond_swap##n (state##n a){\
	static const int i = 1;\
	if(*((char*)&i))\
		return k_swap##n(a);\
	return a;\
}

//Define the conversion function fron nn to nm.
//Takes two nn's and returns an nm
//the by-pointer versions are also defined.
#define KERNELCONV(nn, nm)\
static inline state##nm statemix##nn(state##nn a, state##nn b){\
	state##nm ret = state##nm##_zero();\
	memcpy(ret.state   , a.state, 1<<(nn-1));\
	memcpy(ret.state+(1<<(nn-1)), b.state, 1<<(nn-1));\
	return ret;\
}\
static inline void statemixp##nn(state##nn *a, state##nn *b, state##nm *ret){\
	memcpy(ret->state,	 			a->state, 1<<(nn-1));\
	memcpy(ret->state+(1<<(nn-1)), 	b->state, 1<<(nn-1));\
}\
/*Duplicate */\
static inline state##nm statedup##nn(state##nn a){\
	return statemix##nn(a,a);\
}\
/*Retrieve the highest precision bits*/\
static inline state##nn k_high##nm(state##nm a){\
	state##nn ret;\
	memcpy(ret.state, a.state, 1<<(nn-1));\
	return ret;\
}\
static inline state##nn k_highp##nm(state##nm *a, state##nn *ret){\
	memcpy(ret->state, a->state, 1<<(nn-1));\
}\
/*Retrieve the lowest precision bits*/\
static inline state##nn k_low##nm(state##nm a){\
	state##nn ret;\
	memcpy(ret.state, a.state + ((1<<(nn-1))), 1<<(nn-1));\
	return ret;\
}\
static inline state##nn k_lowp##nm(state##nm *a, state##nn *ret){\
	memcpy(ret->state, a->state + ((1<<(nn-1))), 1<<(nn-1));\
}


#define KERNEL_MULTIPLEX_CALL(iscopy, func) KERNEL_MULTIPLEX_CALL_##iscopy(func)
#define KERNEL_MULTIPLEX_CALL_1(func) current = func(current);
#define KERNEL_MULTIPLEX_CALL_0(func) func(&current);

//Multiplex a low level kernel to a higher level.
//The last argument specifies what type of kernel it is-
//if it is a type1 or type 2 kernel
#define KERNEL_MULTIPLEX(func, nn, nm, iscopy)\
static state##nm func##_mt##nm(state##nm a){\
	state##nm ret;\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	state##nn current;												\
		memcpy(current.state, a.state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		KERNEL_MULTIPLEX_CALL(iscopy, func);\
		memcpy(ret.state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
	return ret;\
}

#define KERNEL_MULTIPLEX_SIMD(func, nn, nm, iscopy)\
static state##nm func##_simd_mt##nm(state##nm a){\
	state##nm ret;\
	PRAGMA_SIMD\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	state##nn current;												\
		memcpy(current.state, a.state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		KERNEL_MULTIPLEX_CALL(iscopy, func);\
		memcpy(ret.state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
	return ret;\
}

#define KERNEL_MULTIPLEX_NOPARA(func, nn, nm, iscopy)\
static state##nm func##_nopara_mt##nm(state##nm a){\
	state##nm ret;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	state##nn current;												\
		memcpy(current.state, a.state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		KERNEL_MULTIPLEX_CALL(iscopy, func);\
		memcpy(ret.state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
	return ret;\
}
//Multiplex a low level kernel to a higher level, by POINTER
//Useful for applying a kernel to an extremely large state which is perhaps hundreds of megabytes.
#define KERNEL_MULTIPLEX_POINTER(func, nn, nm, iscopy)\
static void func##_mtp##nm(state##nm *a){\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	state##nn current;\
		memcpy(current.state, a->state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		KERNEL_MULTIPLEX_CALL(iscopy, func);\
		memcpy(a->state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
}

#define KERNEL_MULTIPLEX_POINTER_SIMD(func, nn, nm, iscopy)\
static void func##_simd_mtp##nm(state##nm *a){\
	PRAGMA_SIMD\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	state##nn current;\
		memcpy(current.state, a->state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		KERNEL_MULTIPLEX_CALL(iscopy, func);\
		memcpy(a->state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
}

#define KERNEL_MULTIPLEX_POINTER_NOPARA(func, nn, nm, iscopy)\
static void func##_nopara_mtp##nm(state##nm *a){\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	state##nn current;\
		memcpy(current.state, a->state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		KERNEL_MULTIPLEX_CALL(iscopy, func);\
		memcpy(a->state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
}


#define KERNEL_MULTIPLEX_ICALL(iscopy, func) KERNEL_MULTIPLEX_ICALL_##iscopy(func)
#define KERNEL_MULTIPLEX_ICALL_1(func) current_indexed = func(current_indexed);
#define KERNEL_MULTIPLEX_ICALL_0(func) func(&current_indexed);

//Multiplex a low level kernel to a higher level, with index in the upper half.
//Your kernel must operate on statennn but the input array will be treated as statenn's
#define KERNEL_MULTIPLEX_INDEXED_POINTER(func, nn, nnn, nm, iscopy)\
static void func##_mtpi##nm(state##nm *a){\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		memcpy(current.state, a->state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		if(nn == 1)/*Single byte indices.*/\
			index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed = statemix##nn(index,current);\
		KERNEL_MULTIPLEX_ICALL(iscopy, func);\
		/*Run the function on the indexed thing and return the low */\
		current = k_low##nnn(current_indexed);\
		memcpy(a->state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
}


#define KERNEL_MULTIPLEX_INDEXED(func, nn, nnn, nm, iscopy)\
static state##nm func##_mti##nm(state##nm a){\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		memcpy(current.state, a.state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		if(nn == 1)/*Single byte indices.*/\
			index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed = statemix##nn(index,current);\
		/*Run the function on the indexed thing and return the low */\
		KERNEL_MULTIPLEX_ICALL(iscopy, func);\
		/*Run the function on the indexed thing and return the low */\
		current = k_low##nnn(current_indexed);\
		memcpy(a.state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
	return a;\
}

#define KERNEL_MULTIPLEX_INDEXED_NOPARA_POINTER(func, nn, nnn, nm, iscopy)\
static void func##_nopara_mtpi##nm(state##nm *a){\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		memcpy(current.state, a->state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		if(nn == 1)/*Single byte indices.*/\
			index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed = statemix##nn(index,current);\
		/*Run the function on the indexed thing and return the low */\
		KERNEL_MULTIPLEX_ICALL(iscopy, func);\
		/*Run the function on the indexed thing and return the low */\
		current = k_low##nnn(current_indexed);\
		memcpy(a->state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
}


#define KERNEL_MULTIPLEX_INDEXED_NOPARA(func, nn, nnn, nm, iscopy)\
static state##nm func##_mti##nm(state##nm a){\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		memcpy(current.state, a.state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		if(nn == 1)/*Single byte indices.*/\
			index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed = statemix##nn(index,current);\
		/*Run the function on the indexed thing and return the low */\
		KERNEL_MULTIPLEX_ICALL(iscopy, func);\
		/*Run the function on the indexed thing and return the low */\
		current = k_low##nnn(current_indexed);\
		memcpy(a.state + i*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
	}\
	return a;\
}

//Same as above, but the return value's upper part is used to decide where in the result the
//return value's lower half will be placed.
#define KERNEL_MULTIPLEX_INDEXED_EMPLACE(func, nn, nnn, nm, iscopy)\
static state##nm func##_mtie##nm(state##nm a){\
	state##nm ret = {0};\
	static const size_t emplacemask = (1<<(nm-1)) / (1<<(nn-1)) - 1;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		memcpy(current.state, a.state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		if(nn == 1)/*Single byte indices.*/\
			index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed = statemix##nn(index,current);\
		/*Run the function on the indexed thing and return the low */\
		KERNEL_MULTIPLEX_ICALL(iscopy, func);\
		index = k_high##nnn(current_indexed);\
		current = k_low##nnn(current_indexed);\
		if(nn == 1){/*Single byte indices.*/\
			memcpy(&ind8, index.state, 1);\
			ind8 &= emplacemask;\
			memcpy(ret.state + ind8*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}else if (nn == 2){/*Two byte indices*/\
			memcpy(&ind16, index.state, 2);\
			ind16 &= emplacemask;\
			memcpy(ret.state + ind16*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}else if (nn == 3){/*Three byte indices*/\
			memcpy(&ind32, index.state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret.state + ind32*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}else{	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(&ind32, index.state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret.state + ind32*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}\
	}\
	return ret;\
}


#define KERNEL_MULTIPLEX_POINTER_INDEXED_EMPLACE(func, nn, nnn, nm, iscopy)\
static void func##_mtpie##nm(state##nm* a){\
	state##nm* ret = malloc(sizeof(state##nm));\
	if(!ret) return;\
	memcpy(ret, a, sizeof(state##nm));\
	static const size_t emplacemask = (1<<(nm-1)) / (1<<(nn-1)) - 1;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		memcpy(current.state, a->state + i*(1<<(nn-1)), (1<<(nn-1)) );\
		if(nn == 1)/*Single byte indices.*/\
			index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed = statemix##nn(index,current);\
		/*Run the function on the indexed thing and return the low */\
		/*Run the function on the indexed thing and return the low */\
		KERNEL_MULTIPLEX_ICALL(iscopy, func);\
		index = k_high##nnn(current_indexed);\
		current = k_low##nnn(current_indexed);\
		if(nn == 1){/*Single byte indices.*/\
			memcpy(&ind8, index.state, 1);\
			ind8 &= emplacemask;\
			memcpy(ret->state + ind8*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}else if (nn == 2){/*Two byte indices*/\
			memcpy(&ind16, index.state, 2);\
			ind16 &= emplacemask;\
			memcpy(ret->state + ind16*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}else if (nn == 3){/*Three byte indices*/\
			memcpy(&ind32, index.state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret->state + ind32*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}else{	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(&ind32, index.state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret->state + ind32*(1<<(nn-1)), current.state, (1<<(nn-1)) );\
		}\
	}\
	memcpy(a, ret, sizeof(state##nm));\
	free(ret);\
}

//The shared state function.
//func must take in nnn of state.
//the very first statenn within the statenm is considered "shared"
//every single nn thereafter is looped over.
//The shared state and the i'th statenn is merged into a single statennn
//which is passed to your function.
//nnn must be nn + 1
//The shared state is presumed to be very large, so this is all done with pointers and heap memory.
//All that said, you *can* pass a copy-kernel.
#define KERNEL_SHARED_CALL(iscopy, func) KERNEL_SHARED_CALL_##iscopy(func)
#define KERNEL_SHARED_CALL_1(func) *passed = func(*passed);
#define KERNEL_SHARED_CALL_0(func) func(passed);
#define KERNEL_SHARED_STATE_POINTER(func, nn, nnn, nm, iscopy)\
void func##_sharedp##nn##_##nm(state##nm *a){\
	state##nnn *passed = malloc(sizeof(state##nnn));\
	if(!passed) return;\
	memcpy(passed->state, a->state, sizeof(state##nn));\
	/*i = 1 because the 0'th element is shared.*/\
	for(size_t i = 1; i < (1<<(nm-1)) / (1<<(nn-1)); i++){\
		memcpy(passed->state + sizeof(state##nn), a->state + i * sizeof(state##nn), sizeof(state##nn));\
		KERNEL_SHARED_CALL(iscopy, func)\
		memcpy(a->state + i * sizeof(state##nn), passed->state + sizeof(state##nn), sizeof(state##nn));\
	}\
	/*Copy the shared state back.*/\
	memcpy(a->state, passed->state, sizeof(state##nn));\
	free(passed);\
}\



//Fetch a lower state out of a higher state by index.
//these are not kernels.
#define SUBSTATE_ARRAY(nn, nm)\
static inline state##nn state_get##nn##_##nm (state##nm* input, size_t index){\
	state##nn ret;\
	memcpy(ret.state, input->state + index*(1<<(nn-1)), (1<<(nn-1)));\
	return ret;\
}\
static inline state##nn state_get_byteoffset##nn##_##nm (state##nm* input, size_t index){\
	state##nn ret;\
	memcpy(ret.state, input->state + index, (1<<(nn-1)));\
	return ret;\
}\
static inline void state_getp##nn##_##nm (state##nm* input, state##nn* ret, size_t index){\
	memcpy(ret->state, input->state + index*(1<<(nn-1)), (1<<(nn-1)));\
}\
static inline void state_getp_byteoffset##nn##_##nm (state##nm* input, state##nn* ret, size_t index){\
	memcpy(ret->state, input->state + index, (1<<(nn-1)));\
}\
static inline void state_insert##nn##_##nm (state##nm* targ, state##nn val, size_t index){\
	memcpy(targ->state + index*(1<<(nn-1)), val.state, (1<<(nn-1)) );\
}\
static inline void state_insert_byteoffset##nn##_##nm (state##nm* targ, state##nn val, size_t index){\
	memcpy(targ->state + index, val.state, (1<<(nn-1)) );\
}\
static inline void state_insertp##nn##_##nm (state##nm* targ, state##nn* val, size_t index){\
	memcpy(targ->state + index*(1<<(nn-1)), val->state, (1<<(nn-1)) );\
}\
static inline void state_insertp_byteoffset##nn##_##nm (state##nm* targ, state##nn* val, size_t index){\
	memcpy(targ->state + index, val->state, (1<<(nn-1)) );\
}


//There is no relevant op for 1.
KERNELB_NO_OP(1,1);
//helper function.
static inline state1 to_state1(uint8_t a){
	state1 q;
	memcpy(q.state, &a,1);
	return q;
}
static inline state1 signed_to_state1(int8_t a){
	state1 q;
	memcpy(q.state, &a,1);
	return q;
}
static inline uint8_t from_state1(state1 q){
	return q.state[0];
}
static inline int8_t signed_from_state1(state1 q){
	int8_t ret; 
	memcpy(&ret, q.state, 1);
	return ret;
}
//state2. Contains 2^(2-1) bytes, or 2 bytes.
KERNELB(2,2);
static inline state2 to_state2(uint16_t a){
	state2 q;
	memcpy(q.state, &a,2);
	return q;
}
static inline uint16_t from_state2(state2 q){
	uint16_t a;
	memcpy(&a, q.state, 2);
	return a;
}


static inline state2 signed_to_state2(int16_t a){
	state2 q;
	memcpy(q.state, &a, 2);
	return q;
}
static inline int16_t signed_from_state2(state2 q){
	int16_t a;
	memcpy(&a, q.state, 2);
	return a;
}
static state2 k_add2(state2 a){
	uint8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 + arg2;
	memcpy(a.state, &ret, 1);
	return a;
}
static state2 k_sadd2(state2 a){
	int8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 + arg2;
	memcpy(a.state, &ret, 1);
	return a;
}

static state2 k_sub2(state2 a){
	uint8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 - arg2;
	memcpy(a.state, &ret, 1);
	return a;
}
static state2 k_ssub2(state2 a){
	int8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 - arg2;
	memcpy(a.state, &ret, 1);
	return a;
}

static state2 k_mult2(state2 a){
	uint8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 * arg2;
	memcpy(a.state, &ret, 1);
	return a;
}
static state2 k_smult2(state2 a){
	int8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 * arg2;
	memcpy(a.state, &ret, 1);
	return a;
}

static state2 k_div2(state2 a){
	uint8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 / arg2;
	memcpy(a.state, &ret, 1);
	return a;
}
static state2 k_sdiv2(state2 a){
	int8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 / arg2;
	memcpy(a.state, &ret, 1);
	return a;
}

static state2 k_mod2(state2 a){
	uint8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 % arg2;
	memcpy(a.state, &ret, 1);
	return a;
}
static state2 k_smod2(state2 a){
	int8_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 1);
	memcpy(&arg2, a.state+1, 1);
	ret = arg1 % arg2;
	memcpy(a.state, &ret, 1);
	return a;
}


//Conversion function to up from 1 byte to 2 bytes.
KERNELCONV(1,2);
//Array functions.
SUBSTATE_ARRAY(1,2);



//state3. contains 4 bytes- so, most of your typical types go here.
KERNELB(3,4);
static inline state3 to_state3(uint32_t a){
	state3 q;
	memcpy(q.state, &a, 4);
	return q;
}
static inline uint32_t from_state3(state3 q){
	uint32_t a;
	memcpy(&a, q.state, 4);
	return a;
}


static inline state3 signed_to_state3(int32_t a){
	state3 q;
	memcpy(q.state, &a, 4);
	return q;
}
static inline int32_t signed_from_state3(state3 q){
	int32_t a;
	memcpy(&a, q.state, 4);
	return a;
}
static inline state3 float_to_state3(float a){
	state3 q;
	memcpy(q.state, &a, 4);
	return q;
}
static inline float float_from_state3(state3 q){
	float a;
	memcpy(&a, q.state, 4);
	return a;
}

KERNELCONV(2,3);
//Array functions.
SUBSTATE_ARRAY(1,3);
SUBSTATE_ARRAY(2,3);
//We need to define some hardware-accelerated kernels!
//These kernels operate on the two halves of a state machine.
//The return value is the result of the operation FOR THE HALF TYPE.
//state3 has two uint16 or int16's in it, and as such, operations are performed as if they are 16 bit, not 32.
static state3 k_add3(state3 a){
	uint16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 + arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_sadd3(state3 a){
	int16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 + arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_sub3(state3 a){
	uint16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 - arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_ssub3(state3 a){
	int16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 - arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_mult3(state3 a){
	uint16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 * arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_smult3(state3 a){
	int16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 * arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_div3(state3 a){
	uint16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 / arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_sdiv3(state3 a){
	int16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 / arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_mod3(state3 a){
	uint16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 % arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
static state3 k_smod3(state3 a){
	int16_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 2);
	memcpy(&arg2, a.state+2, 2);
	ret = arg1 % arg2;
	memcpy(a.state, &ret, 2);
	return a;
}
//Only floating point function.
//Fast Inverse Square Root.
static state3 k_fisr(state3 xx){
	int32_t x = from_state3(xx);
	int32_t i; 
	float x2;
	memcpy(&i, xx.state, 4);
	i = 0x5F1FFFF9 - (i>>1);
	memcpy(&x2, &i, 4);
	x2 *= 0.703952253f * (2.38924456f - x * x2 * x2);
	memcpy(xx.state, &x2, 4);
	return xx;
}

KERNELB(4,8);
//The to and from functions can't be used unless we have uint64_t
#ifdef UINT64_MAX
static inline state4 to_state4(uint64_t a){
	state4 q;
	memcpy(q.state, &a, 8);
	return q;
}
static inline uint64_t from_state4(state4 q){
	uint64_t a;
	memcpy(&a, q.state, 8);
	return a;
}

static inline state4 signed_to_state4(int64_t a){
	state4 q;
	memcpy(q.state, &a, 8);
	return q;
}
static inline int64_t signed_from_state4(state4 q){
	int64_t a;
	memcpy(&a, q.state, 8);
	return a;
}

static inline state4 double_to_state4(double a){
	state4 q;
	memcpy(q.state, &a, 8);
	return q;
}
static inline double double_from_state4(state4 q){
	double a;
	memcpy(&a, q.state, 8);
	return a;
}
#endif
//Define the hardware accelerated kernels.
static state4 k_add4(state4 a){
	uint32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 + arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_sadd4(state4 a){
	int32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 + arg2;
	memcpy(a.state, &ret, 4);
	return a;
}

static state4 k_sub4(state4 a){
	uint32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 - arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_ssub4(state4 a){
	int32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 - arg2;
	memcpy(a.state, &ret, 4);
	return a;
}

static state4 k_mult4(state4 a){
	uint32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 * arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_smult4(state4 a){
	int32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 * arg2;
	memcpy(a.state, &ret, 4);
	return a;
}


static state4 k_div4(state4 a){
	uint32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 / arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_sdiv4(state4 a){
	int32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 / arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_mod4(state4 a){
	uint32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 % arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_smod4(state4 a){
	int32_t arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 % arg2;
	memcpy(a.state, &ret, 4);
	return a;
}

//For the first and only time, floating point!
//Doubles are not implemented.
static state4 k_fadd4(state4 a){
	float arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 + arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_fsub4(state4 a){
	float arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 - arg2;
	memcpy(a.state, &ret, 4);
	return a;
}

static state4 k_fmult4(state4 a){
	float arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 * arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
static state4 k_fdiv4(state4 a){
	float arg1, arg2, ret;
	memcpy(&arg1, a.state, 4);
	memcpy(&arg2, a.state+4, 4);
	ret = arg1 / arg2;
	memcpy(a.state, &ret, 4);
	return a;
}
KERNELCONV(3,4);
//Array functions.
SUBSTATE_ARRAY(1,4);
SUBSTATE_ARRAY(2,4);
SUBSTATE_ARRAY(3,4);

//larger kernels.
KERNELB(5,16);
KERNELCONV(4,5);
//Array functions.
SUBSTATE_ARRAY(1,5);
SUBSTATE_ARRAY(2,5);
SUBSTATE_ARRAY(3,5);
SUBSTATE_ARRAY(4,5);
KERNELB(6,16);
KERNELCONV(5,6);
//Array functions.
SUBSTATE_ARRAY(1,6);
SUBSTATE_ARRAY(2,6);
SUBSTATE_ARRAY(3,6);
SUBSTATE_ARRAY(4,6);
SUBSTATE_ARRAY(5,6);
KERNELB(7,16);
KERNELCONV(6,7);
//Array functions.
SUBSTATE_ARRAY(1,7);
SUBSTATE_ARRAY(2,7);
SUBSTATE_ARRAY(3,7);
SUBSTATE_ARRAY(4,7);
SUBSTATE_ARRAY(5,7);
SUBSTATE_ARRAY(6,7);
KERNELB(8,16);
KERNELCONV(7,8);
//Array functions.
SUBSTATE_ARRAY(1,8);
SUBSTATE_ARRAY(2,8);
SUBSTATE_ARRAY(3,8);
SUBSTATE_ARRAY(4,8);
SUBSTATE_ARRAY(5,8);
SUBSTATE_ARRAY(6,8);
SUBSTATE_ARRAY(7,8);
KERNELB(9,16);
KERNELCONV(8,9);
//Array functions.
SUBSTATE_ARRAY(1,9);
SUBSTATE_ARRAY(2,9);
SUBSTATE_ARRAY(3,9);
SUBSTATE_ARRAY(4,9);
SUBSTATE_ARRAY(5,9);
SUBSTATE_ARRAY(6,9);
SUBSTATE_ARRAY(7,9);
SUBSTATE_ARRAY(8,9);
KERNELB(10,16);
KERNELCONV(9,10);
//Array functions.
SUBSTATE_ARRAY(1,10);
SUBSTATE_ARRAY(2,10);
SUBSTATE_ARRAY(3,10);
SUBSTATE_ARRAY(4,10);
SUBSTATE_ARRAY(5,10);
SUBSTATE_ARRAY(6,10);
SUBSTATE_ARRAY(7,10);
SUBSTATE_ARRAY(8,10);
SUBSTATE_ARRAY(9,10);

//The eleventh order kernel holds 2^(11-1) bytes, or 1024 bytes.
KERNELB(11,16);
KERNELCONV(10,11);
//Array functions.
SUBSTATE_ARRAY(1,11);
SUBSTATE_ARRAY(2,11);
SUBSTATE_ARRAY(3,11);
SUBSTATE_ARRAY(4,11);
SUBSTATE_ARRAY(5,11);
SUBSTATE_ARRAY(6,11);
SUBSTATE_ARRAY(7,11);
SUBSTATE_ARRAY(8,11);
SUBSTATE_ARRAY(9,11);
SUBSTATE_ARRAY(10,11);
KERNELB(12,16);
KERNELCONV(11,12);
//Array functions.
SUBSTATE_ARRAY(1,12);
SUBSTATE_ARRAY(2,12);
SUBSTATE_ARRAY(3,12);
SUBSTATE_ARRAY(4,12);
SUBSTATE_ARRAY(5,12);
SUBSTATE_ARRAY(6,12);
SUBSTATE_ARRAY(7,12);
SUBSTATE_ARRAY(8,12);
SUBSTATE_ARRAY(9,12);
SUBSTATE_ARRAY(10,12);
SUBSTATE_ARRAY(11,12);
KERNELB(13,16);
KERNELCONV(12,13);
//Array functions.
SUBSTATE_ARRAY(1,13);
SUBSTATE_ARRAY(2,13);
SUBSTATE_ARRAY(3,13);
SUBSTATE_ARRAY(4,13);
SUBSTATE_ARRAY(5,13);
SUBSTATE_ARRAY(6,13);
SUBSTATE_ARRAY(7,13);
SUBSTATE_ARRAY(8,13);
SUBSTATE_ARRAY(9,13);
SUBSTATE_ARRAY(10,13);
SUBSTATE_ARRAY(11,13);
SUBSTATE_ARRAY(12,13);
KERNELB(14,16);
KERNELCONV(13,14);
//Array functions.
SUBSTATE_ARRAY(1,14);
SUBSTATE_ARRAY(2,14);
SUBSTATE_ARRAY(3,14);
SUBSTATE_ARRAY(4,14);
SUBSTATE_ARRAY(5,14);
SUBSTATE_ARRAY(6,14);
SUBSTATE_ARRAY(7,14);
SUBSTATE_ARRAY(8,14);
SUBSTATE_ARRAY(9,14);
SUBSTATE_ARRAY(10,14);
SUBSTATE_ARRAY(11,14);
SUBSTATE_ARRAY(12,14);
SUBSTATE_ARRAY(13,14);
KERNELB(15,16);
KERNELCONV(14,15);
//Array functions.
SUBSTATE_ARRAY(1,15);
SUBSTATE_ARRAY(2,15);
SUBSTATE_ARRAY(3,15);
SUBSTATE_ARRAY(4,15);
SUBSTATE_ARRAY(5,15);
SUBSTATE_ARRAY(6,15);
SUBSTATE_ARRAY(7,15);
SUBSTATE_ARRAY(8,15);
SUBSTATE_ARRAY(9,15);
SUBSTATE_ARRAY(10,15);
SUBSTATE_ARRAY(11,15);
SUBSTATE_ARRAY(12,15);
SUBSTATE_ARRAY(13,15);
SUBSTATE_ARRAY(14,15);
KERNELB(16,16);
KERNELCONV(15,16);
//Array functions.
SUBSTATE_ARRAY(1,16);
SUBSTATE_ARRAY(2,16);
SUBSTATE_ARRAY(3,16);
SUBSTATE_ARRAY(4,16);
SUBSTATE_ARRAY(5,16);
SUBSTATE_ARRAY(6,16);
SUBSTATE_ARRAY(7,16);
SUBSTATE_ARRAY(8,16);
SUBSTATE_ARRAY(9,16);
SUBSTATE_ARRAY(10,16);
SUBSTATE_ARRAY(11,16);
SUBSTATE_ARRAY(12,16);
SUBSTATE_ARRAY(13,16);
SUBSTATE_ARRAY(14,16);
SUBSTATE_ARRAY(15,16);
KERNELB(17,16);
KERNELCONV(16,17);
//Array functions.
SUBSTATE_ARRAY(1,17);
SUBSTATE_ARRAY(2,17);
SUBSTATE_ARRAY(3,17);
SUBSTATE_ARRAY(4,17);
SUBSTATE_ARRAY(5,17);
SUBSTATE_ARRAY(6,17);
SUBSTATE_ARRAY(7,17);
SUBSTATE_ARRAY(8,17);
SUBSTATE_ARRAY(9,17);
SUBSTATE_ARRAY(10,17);
SUBSTATE_ARRAY(11,17);
SUBSTATE_ARRAY(12,17);
SUBSTATE_ARRAY(13,17);
SUBSTATE_ARRAY(14,17);
SUBSTATE_ARRAY(15,17);
SUBSTATE_ARRAY(16,17);

KERNELB(18,16);
KERNELCONV(17,18);
//Array functions.
SUBSTATE_ARRAY(1,18);
SUBSTATE_ARRAY(2,18);
SUBSTATE_ARRAY(3,18);
SUBSTATE_ARRAY(4,18);
SUBSTATE_ARRAY(5,18);
SUBSTATE_ARRAY(6,18);
SUBSTATE_ARRAY(7,18);
SUBSTATE_ARRAY(8,18);
SUBSTATE_ARRAY(9,18);
SUBSTATE_ARRAY(10,18);
SUBSTATE_ARRAY(11,18);
SUBSTATE_ARRAY(12,18);
SUBSTATE_ARRAY(13,18);
SUBSTATE_ARRAY(14,18);
SUBSTATE_ARRAY(15,18);
SUBSTATE_ARRAY(16,18);
SUBSTATE_ARRAY(17,18);
KERNELB(19,16);
KERNELCONV(18,19);
//Array functions.
SUBSTATE_ARRAY(1,19);
SUBSTATE_ARRAY(2,19);
SUBSTATE_ARRAY(3,19);
SUBSTATE_ARRAY(4,19);
SUBSTATE_ARRAY(5,19);
SUBSTATE_ARRAY(6,19);
SUBSTATE_ARRAY(7,19);
SUBSTATE_ARRAY(8,19);
SUBSTATE_ARRAY(9,19);
SUBSTATE_ARRAY(10,19);
SUBSTATE_ARRAY(11,19);
SUBSTATE_ARRAY(12,19);
SUBSTATE_ARRAY(13,19);
SUBSTATE_ARRAY(14,19);
SUBSTATE_ARRAY(15,19);
SUBSTATE_ARRAY(16,19);
SUBSTATE_ARRAY(17,19);
SUBSTATE_ARRAY(18,19);

//Henceforth it is no longer safe to have the Op functions since
//it'd be straight up bad practice to put shit on the stack.
KERNELB_NO_OP(20,16);
KERNELCONV(19,20);
//Array functions.
SUBSTATE_ARRAY(1,20);
SUBSTATE_ARRAY(2,20);
SUBSTATE_ARRAY(3,20);
SUBSTATE_ARRAY(4,20);
SUBSTATE_ARRAY(5,20);
SUBSTATE_ARRAY(6,20);
SUBSTATE_ARRAY(7,20);
SUBSTATE_ARRAY(8,20);
SUBSTATE_ARRAY(9,20);
SUBSTATE_ARRAY(10,20);
SUBSTATE_ARRAY(11,20);
SUBSTATE_ARRAY(12,20);
SUBSTATE_ARRAY(13,20);
SUBSTATE_ARRAY(14,20);
SUBSTATE_ARRAY(15,20);
SUBSTATE_ARRAY(16,20);
SUBSTATE_ARRAY(17,20);
SUBSTATE_ARRAY(18,20);
SUBSTATE_ARRAY(19,20);
//Holds an entire megabyte.
KERNELB_NO_OP(21,16);
KERNELCONV(20,21);
//Array functions.
SUBSTATE_ARRAY(1,21);
SUBSTATE_ARRAY(2,21);
SUBSTATE_ARRAY(3,21);
SUBSTATE_ARRAY(4,21);
SUBSTATE_ARRAY(5,21);
SUBSTATE_ARRAY(6,21);
SUBSTATE_ARRAY(7,21);
SUBSTATE_ARRAY(8,21);
SUBSTATE_ARRAY(9,21);
SUBSTATE_ARRAY(10,21);
SUBSTATE_ARRAY(11,21);
SUBSTATE_ARRAY(12,21);
SUBSTATE_ARRAY(13,21);
SUBSTATE_ARRAY(14,21);
SUBSTATE_ARRAY(15,21);
SUBSTATE_ARRAY(16,21);
SUBSTATE_ARRAY(17,21);
SUBSTATE_ARRAY(18,21);
SUBSTATE_ARRAY(19,21);
SUBSTATE_ARRAY(20,21);
//TWO ENTIRE MEGABYTES
KERNELB_NO_OP(22,16);
KERNELCONV(21,22);
//Array functions.
SUBSTATE_ARRAY(1,22);
SUBSTATE_ARRAY(2,22);
SUBSTATE_ARRAY(3,22);
SUBSTATE_ARRAY(4,22);
SUBSTATE_ARRAY(5,22);
SUBSTATE_ARRAY(6,22);
SUBSTATE_ARRAY(7,22);
SUBSTATE_ARRAY(8,22);
SUBSTATE_ARRAY(9,22);
SUBSTATE_ARRAY(10,22);
SUBSTATE_ARRAY(11,22);
SUBSTATE_ARRAY(12,22);
SUBSTATE_ARRAY(13,22);
SUBSTATE_ARRAY(14,22);
SUBSTATE_ARRAY(15,22);
SUBSTATE_ARRAY(16,22);
SUBSTATE_ARRAY(17,22);
SUBSTATE_ARRAY(18,22);
SUBSTATE_ARRAY(19,22);
SUBSTATE_ARRAY(20,22);
SUBSTATE_ARRAY(21,22);
//FOUR ENTIRE MEGABYTES
KERNELB_NO_OP(23,16);
KERNELCONV(22,23);
//Array functions.
SUBSTATE_ARRAY(1,23);
SUBSTATE_ARRAY(2,23);
SUBSTATE_ARRAY(3,23);
SUBSTATE_ARRAY(4,23);
SUBSTATE_ARRAY(5,23);
SUBSTATE_ARRAY(6,23);
SUBSTATE_ARRAY(7,23);
SUBSTATE_ARRAY(8,23);
SUBSTATE_ARRAY(9,23);
SUBSTATE_ARRAY(10,23);
SUBSTATE_ARRAY(11,23);
SUBSTATE_ARRAY(12,23);
SUBSTATE_ARRAY(13,23);
SUBSTATE_ARRAY(14,23);
SUBSTATE_ARRAY(15,23);
SUBSTATE_ARRAY(16,23);
SUBSTATE_ARRAY(17,23);
SUBSTATE_ARRAY(18,23);
SUBSTATE_ARRAY(19,23);
SUBSTATE_ARRAY(20,23);
SUBSTATE_ARRAY(21,23);
SUBSTATE_ARRAY(22,23);
//EIGHT ENTIRE MEGABYTES. As much as the Dreamcast.
KERNELB_NO_OP(24,16);
KERNELCONV(23,24);
//Array functions.
SUBSTATE_ARRAY(1,24);
SUBSTATE_ARRAY(2,24);
SUBSTATE_ARRAY(3,24);
SUBSTATE_ARRAY(4,24);
SUBSTATE_ARRAY(5,24);
SUBSTATE_ARRAY(6,24);
SUBSTATE_ARRAY(7,24);
SUBSTATE_ARRAY(8,24);
SUBSTATE_ARRAY(9,24);
SUBSTATE_ARRAY(10,24);
SUBSTATE_ARRAY(11,24);
SUBSTATE_ARRAY(12,24);
SUBSTATE_ARRAY(13,24);
SUBSTATE_ARRAY(14,24);
SUBSTATE_ARRAY(15,24);
SUBSTATE_ARRAY(16,24);
SUBSTATE_ARRAY(17,24);
SUBSTATE_ARRAY(18,24);
SUBSTATE_ARRAY(19,24);
SUBSTATE_ARRAY(20,24);
SUBSTATE_ARRAY(21,24);
SUBSTATE_ARRAY(22,24);
SUBSTATE_ARRAY(23,24);
//16 megs
KERNELB_NO_OP(25,16);
KERNELCONV(24,25);
//Array functions.
SUBSTATE_ARRAY(1,25);
SUBSTATE_ARRAY(2,25);
SUBSTATE_ARRAY(3,25);
SUBSTATE_ARRAY(4,25);
SUBSTATE_ARRAY(5,25);
SUBSTATE_ARRAY(6,25);
SUBSTATE_ARRAY(7,25);
SUBSTATE_ARRAY(8,25);
SUBSTATE_ARRAY(9,25);
SUBSTATE_ARRAY(10,25);
SUBSTATE_ARRAY(11,25);
SUBSTATE_ARRAY(12,25);
SUBSTATE_ARRAY(13,25);
SUBSTATE_ARRAY(14,25);
SUBSTATE_ARRAY(15,25);
SUBSTATE_ARRAY(16,25);
SUBSTATE_ARRAY(17,25);
SUBSTATE_ARRAY(18,25);
SUBSTATE_ARRAY(19,25);
SUBSTATE_ARRAY(20,25);
SUBSTATE_ARRAY(21,25);
SUBSTATE_ARRAY(22,25);
SUBSTATE_ARRAY(23,25);
SUBSTATE_ARRAY(24,25);
//32 megs
KERNELB_NO_OP(26,16);
KERNELCONV(25,26);
//Array functions.
SUBSTATE_ARRAY(1,26);
SUBSTATE_ARRAY(2,26);
SUBSTATE_ARRAY(3,26);
SUBSTATE_ARRAY(4,26);
SUBSTATE_ARRAY(5,26);
SUBSTATE_ARRAY(6,26);
SUBSTATE_ARRAY(7,26);
SUBSTATE_ARRAY(8,26);
SUBSTATE_ARRAY(9,26);
SUBSTATE_ARRAY(10,26);
SUBSTATE_ARRAY(11,26);
SUBSTATE_ARRAY(12,26);
SUBSTATE_ARRAY(13,26);
SUBSTATE_ARRAY(14,26);
SUBSTATE_ARRAY(15,26);
SUBSTATE_ARRAY(16,26);
SUBSTATE_ARRAY(17,26);
SUBSTATE_ARRAY(18,26);
SUBSTATE_ARRAY(19,26);
SUBSTATE_ARRAY(20,26);
SUBSTATE_ARRAY(21,26);
SUBSTATE_ARRAY(22,26);
SUBSTATE_ARRAY(23,26);
SUBSTATE_ARRAY(24,26);
SUBSTATE_ARRAY(25,26);
//64 megs
KERNELB_NO_OP(27,16);
KERNELCONV(26,27);
//Array functions.
SUBSTATE_ARRAY(1,27);
SUBSTATE_ARRAY(2,27);
SUBSTATE_ARRAY(3,27);
SUBSTATE_ARRAY(4,27);
SUBSTATE_ARRAY(5,27);
SUBSTATE_ARRAY(6,27);
SUBSTATE_ARRAY(7,27);
SUBSTATE_ARRAY(8,27);
SUBSTATE_ARRAY(9,27);
SUBSTATE_ARRAY(10,27);
SUBSTATE_ARRAY(11,27);
SUBSTATE_ARRAY(12,27);
SUBSTATE_ARRAY(13,27);
SUBSTATE_ARRAY(14,27);
SUBSTATE_ARRAY(15,27);
SUBSTATE_ARRAY(16,27);
SUBSTATE_ARRAY(17,27);
SUBSTATE_ARRAY(18,27);
SUBSTATE_ARRAY(19,27);
SUBSTATE_ARRAY(20,27);
SUBSTATE_ARRAY(21,27);
SUBSTATE_ARRAY(22,27);
SUBSTATE_ARRAY(23,27);
SUBSTATE_ARRAY(24,27);
SUBSTATE_ARRAY(25,27);
SUBSTATE_ARRAY(26,27);
//128 megs
KERNELB_NO_OP(28,16);
KERNELCONV(27,28);
//Array functions.
SUBSTATE_ARRAY(1,28);
SUBSTATE_ARRAY(2,28);
SUBSTATE_ARRAY(3,28);
SUBSTATE_ARRAY(4,28);
SUBSTATE_ARRAY(5,28);
SUBSTATE_ARRAY(6,28);
SUBSTATE_ARRAY(7,28);
SUBSTATE_ARRAY(8,28);
SUBSTATE_ARRAY(9,28);
SUBSTATE_ARRAY(10,28);
SUBSTATE_ARRAY(11,28);
SUBSTATE_ARRAY(12,28);
SUBSTATE_ARRAY(13,28);
SUBSTATE_ARRAY(14,28);
SUBSTATE_ARRAY(15,28);
SUBSTATE_ARRAY(16,28);
SUBSTATE_ARRAY(17,28);
SUBSTATE_ARRAY(18,28);
SUBSTATE_ARRAY(19,28);
SUBSTATE_ARRAY(20,28);
SUBSTATE_ARRAY(21,28);
SUBSTATE_ARRAY(22,28);
SUBSTATE_ARRAY(23,28);
SUBSTATE_ARRAY(24,28);
SUBSTATE_ARRAY(25,28);
SUBSTATE_ARRAY(26,28);
SUBSTATE_ARRAY(27,28);
//256 megs.
KERNELB_NO_OP(29,16);
KERNELCONV(28,29);
//Array functions.
SUBSTATE_ARRAY(1,29);
SUBSTATE_ARRAY(2,29);
SUBSTATE_ARRAY(3,29);
SUBSTATE_ARRAY(4,29);
SUBSTATE_ARRAY(5,29);
SUBSTATE_ARRAY(6,29);
SUBSTATE_ARRAY(7,29);
SUBSTATE_ARRAY(8,29);
SUBSTATE_ARRAY(9,29);
SUBSTATE_ARRAY(10,29);
SUBSTATE_ARRAY(11,29);
SUBSTATE_ARRAY(12,29);
SUBSTATE_ARRAY(13,29);
SUBSTATE_ARRAY(14,29);
SUBSTATE_ARRAY(15,29);
SUBSTATE_ARRAY(16,29);
SUBSTATE_ARRAY(17,29);
SUBSTATE_ARRAY(18,29);
SUBSTATE_ARRAY(19,29);
SUBSTATE_ARRAY(20,29);
SUBSTATE_ARRAY(21,29);
SUBSTATE_ARRAY(22,29);
SUBSTATE_ARRAY(23,29);
SUBSTATE_ARRAY(24,29);
SUBSTATE_ARRAY(25,29);
SUBSTATE_ARRAY(26,29);
SUBSTATE_ARRAY(27,29);
SUBSTATE_ARRAY(28,29);

//512 megs.
KERNELB_NO_OP(30,16);
KERNELCONV(29,30);
//Array functions.
SUBSTATE_ARRAY(1,30);
SUBSTATE_ARRAY(2,30);
SUBSTATE_ARRAY(3,30);
SUBSTATE_ARRAY(4,30);
SUBSTATE_ARRAY(5,30);
SUBSTATE_ARRAY(6,30);
SUBSTATE_ARRAY(7,30);
SUBSTATE_ARRAY(8,30);
SUBSTATE_ARRAY(9,30);
SUBSTATE_ARRAY(10,30);
SUBSTATE_ARRAY(11,30);
SUBSTATE_ARRAY(12,30);
SUBSTATE_ARRAY(13,30);
SUBSTATE_ARRAY(14,30);
SUBSTATE_ARRAY(15,30);
SUBSTATE_ARRAY(16,30);
SUBSTATE_ARRAY(17,30);
SUBSTATE_ARRAY(18,30);
SUBSTATE_ARRAY(19,30);
SUBSTATE_ARRAY(20,30);
SUBSTATE_ARRAY(21,30);
SUBSTATE_ARRAY(22,30);
SUBSTATE_ARRAY(23,30);
SUBSTATE_ARRAY(24,30);
SUBSTATE_ARRAY(25,30);
SUBSTATE_ARRAY(26,30);
SUBSTATE_ARRAY(27,30);
SUBSTATE_ARRAY(28,30);
SUBSTATE_ARRAY(29,30);

//1G
KERNELB_NO_OP(31,16);
KERNELCONV(30,31);
//Array functions.
SUBSTATE_ARRAY(1,31);
SUBSTATE_ARRAY(2,31);
SUBSTATE_ARRAY(3,31);
SUBSTATE_ARRAY(4,31);
SUBSTATE_ARRAY(5,31);
SUBSTATE_ARRAY(6,31);
SUBSTATE_ARRAY(7,31);
SUBSTATE_ARRAY(8,31);
SUBSTATE_ARRAY(9,31);
SUBSTATE_ARRAY(10,31);
SUBSTATE_ARRAY(11,31);
SUBSTATE_ARRAY(12,31);
SUBSTATE_ARRAY(13,31);
SUBSTATE_ARRAY(14,31);
SUBSTATE_ARRAY(15,31);
SUBSTATE_ARRAY(16,31);
SUBSTATE_ARRAY(17,31);
SUBSTATE_ARRAY(18,31);
SUBSTATE_ARRAY(19,31);
SUBSTATE_ARRAY(20,31);
SUBSTATE_ARRAY(21,31);
SUBSTATE_ARRAY(22,31);
SUBSTATE_ARRAY(23,31);
SUBSTATE_ARRAY(24,31);
SUBSTATE_ARRAY(25,31);
SUBSTATE_ARRAY(26,31);
SUBSTATE_ARRAY(27,31);
SUBSTATE_ARRAY(28,31);
SUBSTATE_ARRAY(29,31);
SUBSTATE_ARRAY(30,31);

#endif
