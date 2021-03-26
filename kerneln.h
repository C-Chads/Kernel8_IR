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

#define KERNEL_FAKE_ALIGN(n) /*a comment*/

//define a 2^(n-1) byte state, and kernel type,
//as well as common operations.
//These are the state member declarations...
//so you can do state30.state10s[3]
#define STATE_MEMBERS(n,alignment) STATE_MEMBERS_##n(alignment)
#define STATE_MEMBERS_1(alignment) /*a comment*/
#define STATE_MEMBERS_2(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<1];
#define STATE_MEMBERS_3(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<1];
#define STATE_MEMBERS_4(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s[1<<1];
#define STATE_MEMBERS_5(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s[1<<1];
#define STATE_MEMBERS_6(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s[1<<1];
#define STATE_MEMBERS_7(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s[1<<1];
#define STATE_MEMBERS_8(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s[1<<1];
#define STATE_MEMBERS_9(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s[1<<1];
#define STATE_MEMBERS_10(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s[1<<1];
#define STATE_MEMBERS_11(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s	[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s	[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s	[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s	[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s	[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s	[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s	[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s	[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s	[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<1];
#define STATE_MEMBERS_12(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s	[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s	[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s	[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s	[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s	[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s	[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s	[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s	[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s	[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<1];
#define STATE_MEMBERS_13(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s	[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s	[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s	[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s	[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s	[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s	[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s	[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s	[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s	[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<1];
#define STATE_MEMBERS_14(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s	[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s	[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s	[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s	[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s	[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s	[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s	[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s	[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s	[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<1];
#define STATE_MEMBERS_15(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s	[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s	[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s	[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s	[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s	[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s	[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s	[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s	[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s	[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<1];
#define STATE_MEMBERS_16(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s	[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s	[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s	[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s	[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s	[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s	[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s	[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s	[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s	[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<1];
#define STATE_MEMBERS_17(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s	[1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s	[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s	[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s	[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s	[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s	[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s	[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s	[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s	[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<1];


#define STATE_MEMBERS_18(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<1];

#define STATE_MEMBERS_19(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<<1];

#define STATE_MEMBERS_20(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<9];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<8];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<7];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<6];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<5];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<4];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<3];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<<2];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<<1];

#define STATE_MEMBERS_21(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 1];

#define STATE_MEMBERS_22(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 1];

#define STATE_MEMBERS_23(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 1];
#define STATE_MEMBERS_24(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 1];

#define STATE_MEMBERS_25(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<24];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state24 state24s[1<< 1];

#define STATE_MEMBERS_26(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<25];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<24];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state24 state24s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state25 state25s[1<< 1];

#define STATE_MEMBERS_27(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<26];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<25];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<24];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state24 state24s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state25 state25s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state26 state26s[1<< 1];

#define STATE_MEMBERS_28(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<27];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<26];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<25];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<24];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state24 state24s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state25 state25s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state26 state26s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state27 state27s[1<< 1];

#define STATE_MEMBERS_29(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<28];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<27];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<26];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<25];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<24];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state24 state24s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state25 state25s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state26 state26s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state27 state27s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state28 state28s[1<< 1];

#define STATE_MEMBERS_30(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<29];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<28];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<27];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<26];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<25];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<24];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state24 state24s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state25 state25s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state26 state26s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state27 state27s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state28 state28s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state29 state29s[1<< 1];

#define STATE_MEMBERS_31(alignment)\
	KERNEL_FAKE_ALIGN(alignment) state1 state1s  [1<<30];\
	KERNEL_FAKE_ALIGN(alignment) state2 state2s  [1<<29];\
	KERNEL_FAKE_ALIGN(alignment) state3 state3s  [1<<28];\
	KERNEL_FAKE_ALIGN(alignment) state4 state4s  [1<<27];\
	KERNEL_FAKE_ALIGN(alignment) state5 state5s  [1<<26];\
	KERNEL_FAKE_ALIGN(alignment) state6 state6s  [1<<25];\
	KERNEL_FAKE_ALIGN(alignment) state7 state7s  [1<<24];\
	KERNEL_FAKE_ALIGN(alignment) state8 state8s  [1<<23];\
	KERNEL_FAKE_ALIGN(alignment) state9 state9s  [1<<22];\
	KERNEL_FAKE_ALIGN(alignment) state10 state10s[1<<21];\
	KERNEL_FAKE_ALIGN(alignment) state11 state11s[1<<20];\
	KERNEL_FAKE_ALIGN(alignment) state12 state12s[1<<19];\
	KERNEL_FAKE_ALIGN(alignment) state13 state13s[1<<18];\
	KERNEL_FAKE_ALIGN(alignment) state14 state14s[1<<17];\
	KERNEL_FAKE_ALIGN(alignment) state15 state15s[1<<16];\
	KERNEL_FAKE_ALIGN(alignment) state16 state16s[1<<15];\
	KERNEL_FAKE_ALIGN(alignment) state17 state17s[1<<14];\
	KERNEL_FAKE_ALIGN(alignment) state18 state18s[1<<13];\
	KERNEL_FAKE_ALIGN(alignment) state19 state19s[1<<12];\
	KERNEL_FAKE_ALIGN(alignment) state20 state20s[1<<11];\
	KERNEL_FAKE_ALIGN(alignment) state21 state21s[1<<10];\
	KERNEL_FAKE_ALIGN(alignment) state22 state22s[1<< 9];\
	KERNEL_FAKE_ALIGN(alignment) state23 state23s[1<< 8];\
	KERNEL_FAKE_ALIGN(alignment) state24 state24s[1<< 7];\
	KERNEL_FAKE_ALIGN(alignment) state25 state25s[1<< 6];\
	KERNEL_FAKE_ALIGN(alignment) state26 state26s[1<< 5];\
	KERNEL_FAKE_ALIGN(alignment) state27 state27s[1<< 4];\
	KERNEL_FAKE_ALIGN(alignment) state28 state28s[1<< 3];\
	KERNEL_FAKE_ALIGN(alignment) state29 state29s[1<< 2];\
	KERNEL_FAKE_ALIGN(alignment) state30 state30s[1<< 1];

#define KERNELB_NO_OP(n, alignment)\
typedef union{\
  KERNEL_ALIGN(alignment) uint8_t state[(size_t)1<<(n-1)];\
  STATE_MEMBERS(n, alignment);\
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
	state##nm ret;\
	ret.state##nn##s[0] = a;\
	ret.state##nn##s[1] = b;\
	return ret;\
}\
static inline void statemixp##nn(state##nn *a, state##nn *b, state##nm *ret){\
	ret->state##nn##s[0] = *a;\
	ret->state##nn##s[1] = *b;\
}\
/*Duplicate */\
static inline state##nm statedup##nn(state##nn a){\
	return statemix##nn(a,a);\
}\
/*Retrieve the highest precision bits*/\
static inline state##nn k_high##nm(state##nm a){\
	return a.state##nn##s[0];\
}\
static inline void k_highp##nm(state##nm *a, state##nn *ret){\
	*ret = a->state##nn##s[0];\
}\
/*Retrieve the lowest precision bits*/\
static inline state##nn k_low##nm(state##nm a){\
	return a.state##nn##s[1];\
}\
static inline void k_lowp##nm(state##nm *a, state##nn *ret){\
	*ret = a->state##nn##s[1];\
}


#define KERNEL_MULTIPLEX_CALL(iscopy, func, nn) KERNEL_MULTIPLEX_CALL_##iscopy(func, nn)
#define KERNEL_MULTIPLEX_CALL_1(func, nn) a.state##nn##s[i] = func(a.state##nn##s[i]);
#define KERNEL_MULTIPLEX_CALL_0(func, nn) func(a.state##nn##s + i);

#define KERNEL_MULTIPLEX_CALLP(iscopy, func, nn) KERNEL_MULTIPLEX_CALLP_##iscopy(func, nn)
#define KERNEL_MULTIPLEX_CALLP_1(func, nn) a->state##nn##s[i] = func(a->state##nn##s[i]);
#define KERNEL_MULTIPLEX_CALLP_0(func, nn) func(a->state##nn##s + i);

//Multiplex a low level kernel to a higher level.
//The last argument specifies what type of kernel it is-
//if it is a type1 or type 2 kernel
#define KERNEL_MULTIPLEX(name, func, nn, nm, iscopy)\
static state##nm name(state##nm a){\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
		KERNEL_MULTIPLEX_CALL(iscopy, func, nn);\
	return a;\
}

#define KERNEL_MULTIPLEX_SIMD(name, func, nn, nm, iscopy)\
static state##nm name(state##nm a){\
	PRAGMA_SIMD\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
			KERNEL_MULTIPLEX_CALL(iscopy, func, nn);\
	return a;\
}

#define KERNEL_MULTIPLEX_NP(name, func, nn, nm, iscopy)\
static state##nm name(state##nm a){\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
			KERNEL_MULTIPLEX_CALL(iscopy, func, nn);\
	return a;\
}
//Multiplex a low level kernel to a higher level, by POINTER
//Useful for applying a kernel to an extremely large state which is perhaps hundreds of megabytes.
#define KERNEL_MULTIPLEX_POINTER(name, func, nn, nm, iscopy)\
static void name(state##nm *a){\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
			KERNEL_MULTIPLEX_CALLP(iscopy, func, nn);\
}

#define KERNEL_MULTIPLEX_POINTER_SIMD(name, func, nn, nm, iscopy)\
static void name(state##nm *a){\
	PRAGMA_SIMD\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
		KERNEL_MULTIPLEX_CALLP(iscopy, func, nn);\
}

#define KERNEL_MULTIPLEX_POINTER_NP(name, func, nn, nm, iscopy)\
static void name(state##nm *a){\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
			KERNEL_MULTIPLEX_CALLP(iscopy, func, nn);\
}

//pointer version
#define KERNEL_MULTIPLEX_ICALLP(iscopy, func) KERNEL_MULTIPLEX_ICALLP_##iscopy(func)
#define KERNEL_MULTIPLEX_ICALLP_1(func) *current_indexed = func(*current_indexed);
#define KERNEL_MULTIPLEX_ICALLP_0(func) func(current_indexed);
//non pointer version.
#define KERNEL_MULTIPLEX_ICALL(iscopy, func) KERNEL_MULTIPLEX_ICALL_##iscopy(func)
#define KERNEL_MULTIPLEX_ICALL_1(func) current_indexed = func(current_indexed);
#define KERNEL_MULTIPLEX_ICALL_0(func) func(&current_indexed);

//Multiplex a low level kernel to a higher level, with index in the upper half.
//Your kernel must operate on statennn but the input array will be treated as statenn's
#define KERNEL_MULTIPLEX_INDEXED_POINTER(name, func, nn, nnn, nm, iscopy)\
static void name(state##nm *a){\
	state##nn *current =NULL, *index = NULL; \
	state##nnn *current_indexed = NULL;\
	current = malloc(sizeof(state##nn));\
	index = calloc(1, sizeof(state##nn));\
	current_indexed = malloc(sizeof(state##nnn));\
	if(!current) goto end;\
	if(!index) goto end;\
	if(!current_indexed) goto end;\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		*current = a->state##nn##s[i];\
		if(nn == 1)/*Single byte indices.*/\
			memcpy(index->state, &ind8, 1);\
		else if (nn == 2)/*Two byte indices*/\
			memcpy(index->state, &ind16, 2);\
		else if (nn == 3)/*Three byte indices*/\
			memcpy(index->state, &ind32, 4);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index->state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		statemixp##nn(index,current, current_indexed);\
		KERNEL_MULTIPLEX_ICALLP(iscopy, func);\
		/*Run the function on the indexed thing and return the low */\
		*current = current_indexed->state##nn##s[1];\
		memcpy(a->state + i*(1<<(nn-1)), current->state, (1<<(nn-1)) );\
	}\
	end:\
	free(current); free(index); free(current_indexed);\
}


#define KERNEL_MULTIPLEX_INDEXED(name, func, nn, nnn, nm, iscopy)\
static name(state##nm a){\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		current = a.state##nn##s[i];\
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

#define KERNEL_MULTIPLEX_INDEXED_NP_POINTER(name, func, nn, nnn, nm, iscopy)\
static void name(state##nm *a){\
	state##nn *current =NULL, *index = NULL; \
	state##nnn *current_indexed = NULL;\
	current = malloc(sizeof(state##nn));\
	index = calloc(1, sizeof(state##nn));\
	current_indexed = malloc(sizeof(state##nnn));\
	if(!current) goto end;\
	if(!index) goto end;\
	if(!current_indexed) goto end;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		*current = a->state##nn##s[i];\
		if(nn == 1)/*Single byte indices.*/\
			memcpy(index->state, &ind8, 1);\
		else if (nn == 2)/*Two byte indices*/\
			memcpy(index->state, &ind16, 2);\
		else if (nn == 3)/*Three byte indices*/\
			memcpy(index->state, &ind32, 4);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index->state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		statemixp##nn(index,current, current_indexed);\
		KERNEL_MULTIPLEX_ICALLP(iscopy, func);\
		/*Run the function on the indexed thing and return the low */\
		*current = current_indexed->state##nn##s[1];\
		memcpy(a->state + i*(1<<(nn-1)), current->state, (1<<(nn-1)) );\
	}\
	end:\
	free(current); free(index); free(current_indexed);\
}


#define KERNEL_MULTIPLEX_INDEXED_NP(name, func, nn, nnn, nm, iscopy)\
static state##nm name(state##nm a){\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		current = a.state##nn##s[i];\
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
#define KERNEL_MULTIPLEX_INDEXED_EMPLACE(name, func, nn, nnn, nm, iscopy)\
static state##nm name(state##nm a){\
	state##nm ret = {0};\
	static const size_t emplacemask = (1<<(nm-1)) / (1<<(nn-1)) - 1;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		state##nn current, index = {0}; state##nnn current_indexed;\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		current = a.state##nn##s[i];\
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


#define KERNEL_MULTIPLEX_POINTER_INDEXED_EMPLACE(name, func, nn, nnn, nm, iscopy)\
static void name(state##nm* a){\
	state##nm* ret = malloc(sizeof(state##nm));\
	state##nn *current =NULL, *index = NULL; \
	state##nnn *current_indexed = NULL;\
	if(!ret) return;\
	current = malloc(sizeof(state##nn));\
	index = calloc(1, sizeof(state##nn));\
	current_indexed = malloc(sizeof(state##nnn));\
	if(!current) goto end;\
	if(!index) goto end;\
	if(!current_indexed) goto end;\
	memcpy(ret, a, sizeof(state##nm));\
	static const size_t emplacemask = (1<<(nm-1)) / (1<<(nn-1)) - 1;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
	{	\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		*current = a->state##nn##s[i];\
		if(nn == 1)/*Single byte indices.*/\
			*index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			*index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			*index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index->state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		statemixp##nn(index,current, current_indexed);\
		/*Run the function on the indexed thing and return the low */\
		KERNEL_MULTIPLEX_ICALLP(iscopy, func);\
		*index = current_indexed->state##nn##s[0];\
		*current = current_indexed->state##nn##s[1];\
		if(nn == 1){/*Single byte indices.*/\
			memcpy(&ind8, index->state, 1);\
			ind8 &= emplacemask;\
			memcpy(ret->state + ind8*(1<<(nn-1)), current->state, (1<<(nn-1)) );\
		}else if (nn == 2){/*Two byte indices*/\
			memcpy(&ind16, index->state, 2);\
			ind16 &= emplacemask;\
			memcpy(ret->state + ind16*(1<<(nn-1)), current->state, (1<<(nn-1)) );\
		}else if (nn == 3){/*Three byte indices*/\
			memcpy(&ind32, index->state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret->state + ind32*(1<<(nn-1)), current->state, (1<<(nn-1)) );\
		}else{	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(&ind32, index->state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret->state + ind32*(1<<(nn-1)), current->state, (1<<(nn-1)) );\
		}\
	}\
	memcpy(a, ret, sizeof(state##nm));\
	end:\
	free(current); free(index); free(current_indexed);\
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
#define KERNEL_SHARED_STATE_POINTER(name, func, nn, nnn, nm, iscopy)\
static void name(state##nm *a){\
	state##nnn *passed = malloc(sizeof(state##nnn));\
	if(!passed) return;\
	/*memcpy(passed->state, a->state, sizeof(state##nn));*/\
	passed->state##nn##s[0] = a->state##nn##s[0];\
	/*i = 1 because the 0'th element is shared.*/\
	for(size_t i = 1; i < (1<<(nm-1)) / (1<<(nn-1)); i++){\
		passed->state##nn##s[1] = a->state##nn##s[i];\
		KERNEL_SHARED_CALL(iscopy, func)\
		a->state##nn##s[i] = passed->state##nn##s[1];\
	}\
	/*Copy the shared state back.*/\
	memcpy(a->state, passed->state, sizeof(state##nn));\
	free(passed);\
}\


#define KERNEL_MULTIKERNEL_CALL(iscopy, funcarr, nn) KERNEL_MULTIKERNEL_CALL_##iscopy(funcarr, nn)
#define KERNEL_MULTIKERNEL_CALL_1(funcarr, nn) a->state##nn##s[i] = (funcarr[i])(a->state##nn##s[i]);
#define KERNEL_MULTIKERNEL_CALL_0(funcarr, nn) (funcarr[i])(a->state##nn##s +i);
//Create a multiplexed kernel which taks in an array of function pointers
//
#define KERNEL_MULTIPLEX_MULTIKERNEL_POINTER(name, funcarr, nn, nm, iscopy)\
static void name(state##nm *a){\
	PRAGMA_PARALLEL\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
		KERNEL_MULTIKERNEL_CALL(iscopy, funcarr, nn);\
}

#define KERNEL_MULTIPLEX_MULTIKERNEL_NP_POINTER(name, funcarr, nn, nm, iscopy)\
static void name(state##nm *a){\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
		KERNEL_MULTIKERNEL_CALL(iscopy, funcarr, nn);\
}

#define KERNEL_MULTIPLEX_MULTIKERNEL_SIMD_POINTER(name, funcarr, nn, nm, iscopy)\
static void name(state##nm *a){\
	PRAGMA_SIMD\
	for(size_t i = 1; i < (1<<(nm-1)) / (1<<(nn-1)); i++)\
		KERNEL_MULTIKERNEL_CALL(iscopy, funcarr);\
}

#define KERNEL_MULTIPLEX_NLOGN_CALLP(func, iscopy) KERNEL_MULTIPLEX_NLOGN_CALLP_##iscopy(func)
#define KERNEL_MULTIPLEX_NLOGN_CALLP_1(func) *current_b = func(*current_b);
#define KERNEL_MULTIPLEX_NLOGN_CALLP_0(func) func(current_b);

#define KERNEL_MULTIPLEX_NLOGN_CALL(func, iscopy) KERNEL_MULTIPLEX_NLOGN_CALL_##iscopy(func)
#define KERNEL_MULTIPLEX_NLOGN_CALL_1(func) current_b = func(current_b);
#define KERNEL_MULTIPLEX_NLOGN_CALL_0(func) func(&current_b);

#define KERNEL_MULTIPLEX_NLOGN_POINTER(name, func, nn, nnn, nm, iscopy)\
static void name(state##nm *a){\
	state##nnn *current_b = NULL;\
	current_b = malloc(sizeof(state##nnn));\
	if(!current_b) goto end;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)) - 1; i++){\
		current_b->state##nn##s[0] = a->state##nn##s[i];\
		for(size_t j = i+1; j < (1<<(nm-1)) / (1<<(nn-1)); j++)\
		{\
			current_b->state##nn##s[1] = a->state##nn##s[j];\
			KERNEL_MULTIPLEX_NLOGN_CALLP(func, iscopy)\
			a->state##nn##s[j] = current_b->state##nn##s[1];\
		}\
		/*Write back elem i*/\
		a->state##nn##s[i] = current_b->state##nn##s[0];\
	}\
	end:\
	free(current_b);\
}


#define KERNEL_MULTIPLEX_NLOGN(name, func, nn, nnn, nm, iscopy)\
static state##nm name(state##nm a){\
	state##nnn current_b;\
	for(size_t i = 0; i < (1<<(nm-1)) / (1<<(nn-1)) - 1; i++){\
		current_b.state##nn##s[0] = a.state##nn##s[i];\
		for(size_t j = i+1; j < (1<<(nm-1)) / (1<<(nn-1)); j++)\
		{\
			current_b.state##nn##s[1] = a.state##nn##s[j];/*Take ownership.*/\
			KERNEL_MULTIPLEX_NLOGN_CALL(func, iscopy) /*Use*/\
			a.state##nn##s[j] = current_b.state##nn##s[1]; /*Write back.*/\
		}\
		/*Write back elem i*/\
		a.state##nn##s[i] = current_b.state##nn##s[0];\
	}\
	return a;\
}
//Fetch a lower state out of a higher state by index.
//these are not kernels.
#define SUBSTATE_ARRAY(nn, nm) /* a comment */


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

//larger kernels.
KERNELB(5,16);
KERNELCONV(4,5);
KERNELB(6,16);
KERNELCONV(5,6);
KERNELB(7,16);
KERNELCONV(6,7);
KERNELB(8,16);
KERNELCONV(7,8);
KERNELB(9,16);
KERNELCONV(8,9);
KERNELB(10,16);
KERNELCONV(9,10);

//The eleventh order kernel holds 2^(11-1) bytes, or 1024 bytes.
KERNELB(11,16);
KERNELCONV(10,11);
KERNELB(12,16);
KERNELCONV(11,12);
KERNELB(13,16);
KERNELCONV(12,13);
KERNELB(14,16);
KERNELCONV(13,14);
KERNELB(15,16);
KERNELCONV(14,15);
KERNELB(16,16);
KERNELCONV(15,16);
KERNELB(17,16);
KERNELCONV(16,17);
KERNELB(18,16);
KERNELCONV(17,18);
KERNELB(19,16);
KERNELCONV(18,19);
//Henceforth it is no longer safe to have the Op functions since
//it'd be straight up bad practice to put shit on the stack.
KERNELB_NO_OP(20,16);
KERNELCONV(19,20);
//Holds an entire megabyte.
KERNELB_NO_OP(21,16);
KERNELCONV(20,21);
//TWO ENTIRE MEGABYTES
KERNELB_NO_OP(22,16);
KERNELCONV(21,22);
//FOUR ENTIRE MEGABYTES
KERNELB_NO_OP(23,16);
KERNELCONV(22,23);
//EIGHT ENTIRE MEGABYTES. As much as the Dreamcast.
KERNELB_NO_OP(24,16);
KERNELCONV(23,24);
//16 megs
KERNELB_NO_OP(25,16);
KERNELCONV(24,25);
//32 megs
KERNELB_NO_OP(26,16);
KERNELCONV(25,26);
//64 megs
KERNELB_NO_OP(27,16);
KERNELCONV(26,27);
//128 megs
KERNELB_NO_OP(28,16);
KERNELCONV(27,28);
//256 megs.
KERNELB_NO_OP(29,16);
KERNELCONV(28,29);
//512 megs.
KERNELB_NO_OP(30,16);
KERNELCONV(29,30);
//1G
KERNELB_NO_OP(31,16);
KERNELCONV(30,31);

#endif
