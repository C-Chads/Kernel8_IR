//DMHSW's "Kernel 8"
//A notation for state machine computing which reflects underlying hardware.
//Metaprogramming language
//Functional Programming language
//Implemented in and fully compatible with C11.

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
8) it should be possible to write a static inline analyzer for your kernel code which can
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

#ifndef K8_FAST_FLOAT_MATH
#define K8_FAST_FLOAT_MATH 1
#endif

#ifndef KERNELN_H
#define KERNELN_H

#ifndef PRAGMA_NOPARALLEL
#define PRAGMA_NOPARALLEL /*a comment*/
#endif


#if defined(_OPENMP)

#ifndef PRAGMA_PARALLEL
#define PRAGMA_PARALLEL _Pragma("omp parallel for")
#endif

#ifndef PRAGMA_SUPARA

#ifdef __clang__
//Mitigate issue compiling with clang- openmp offloading is BUGGED!
#define PRAGMA_SUPARA _Pragma("omp parallel for")
#else
#define PRAGMA_SUPARA _Pragma("omp target teams distribute parallel for")
#endif
//#define PRAGMA_SUPARA _Pragma("omp target parallel for")
#endif

#ifndef PRAGMA_SIMD
#define PRAGMA_SIMD _Pragma("omp simd")
#endif

#ifndef K_IO
#define K_IO _Pragma("omp critical") {
#define K_END_IO }
#endif

#else
//#define PRAGMA_PARALLEL _Pragma("acc loop")
#define PRAGMA_PARALLEL /*a comment */
#define PRAGMA_SUPARA /*a comment*/
#define PRAGMA_SIMD /*a comment*/
#define K_IO {
#define K_END_IO }
#endif

//TODO: use compiler optimization hints to tell the compiler that values are never used for the duration of a function.
//Indicate to the compiler that state variables go unused AND unmodified.
#ifndef K8_UNUSED
#define K8_UNUSED(x) (void)x;
#endif
//Indicate to the compiler that a value is unmodified 
#ifndef K8_CONST
#define K8_CONST(x) /*a comment*/
#endif


#include <stdint.h>
#include <float.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#ifndef K8_DEBUG_PRINT

#ifdef K8_DEBUG
#define K8_DEBUG_PRINT(...) fprintf (stderr, __VA_ARGS__)
#else
#define K8_DEBUG_PRINT(...) /*a comment*/
#endif

#endif


#ifndef K8_ASSERT

#ifdef K8_DEBUG
#define K8_ASSERT(t) assert(t)
#else
#define K8_ASSERT(t) /*a comment.*/
#endif

#endif

#ifndef K8_STATIC_ASSERT

#if defined(static_assert)
#define K8_STATIC_ASSERT(t) static_assert(t, "K8_STATIC_ASSERT failed!")
#else
#define K8_STATIC_ASSERT(t) /*a comment.*/
#warning "You are not compiling with C11, cannot use static_assert"
#endif

#endif

#ifndef __STDC_IEC_559__
#warning "Nonconformant float implementation, floating point may not work correctly. Run floatmath tests."
#endif


#ifndef K8_NO_ALIGN
#include <stdalign.h>
#define K8_ALIGN(n) alignas(n)
#else
#define K8_ALIGN(n) /*a comment*/
#endif

//define a 2^(n-1) byte state, and kernel type,
//as well as common operations.
//These are the state member declarations...
//so you can do state30.state10s[3]
#define STATE_MEMBERS(n,alignment) STATE_MEMBERS_##n(alignment)
#define STATE_MEMBERS_1(alignment)\
	K8_ALIGN(alignment) uint8_t u;\
	K8_ALIGN(alignment) int8_t i;
#define STATE_MEMBERS_2(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<1];\
	K8_ALIGN(alignment) uint16_t u;\
	K8_ALIGN(alignment) int16_t i;
#define STATE_MEMBERS_3(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<1];\
	K8_ALIGN(alignment) float f;\
	K8_ALIGN(alignment) uint32_t u;\
	K8_ALIGN(alignment) int32_t i;
#ifndef INT64_MAX

#define STATE_MEMBERS_4(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<1];

#else

#define STATE_MEMBERS_4(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<1];\
	uint64_t u;\
	int64_t i;\

#endif
#define STATE_MEMBERS_5(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state4 state4s[(ssize_t)1<<1];
#define STATE_MEMBERS_6(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state4 state4s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state5 state5s[(ssize_t)1<<1];
#define STATE_MEMBERS_7(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state4 state4s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state5 state5s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state6 state6s[(ssize_t)1<<1];
#define STATE_MEMBERS_8(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state4 state4s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state5 state5s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state6 state6s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state7 state7s[(ssize_t)1<<1];
#define STATE_MEMBERS_9(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state4 state4s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state5 state5s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state6 state6s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state7 state7s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state8 state8s[(ssize_t)1<<1];
#define STATE_MEMBERS_10(alignment)\
	K8_ALIGN(alignment) state1 state1s[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state2 state2s[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state3 state3s[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state4 state4s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state5 state5s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state6 state6s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state7 state7s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state8 state8s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state9 state9s[(ssize_t)1<<1];
#define STATE_MEMBERS_11(alignment)\
	K8_ALIGN(alignment) state1 state1s	[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state2 state2s	[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state3 state3s	[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state4 state4s	[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state5 state5s	[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state6 state6s	[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state7 state7s	[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state8 state8s	[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state9 state9s	[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<1];
#define STATE_MEMBERS_12(alignment)\
	K8_ALIGN(alignment) state1 state1s	[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state2 state2s	[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state3 state3s	[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state4 state4s	[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state5 state5s	[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state6 state6s	[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state7 state7s	[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state8 state8s	[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state9 state9s	[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<1];
#define STATE_MEMBERS_13(alignment)\
	K8_ALIGN(alignment) state1 state1s	[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state2 state2s	[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state3 state3s	[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state4 state4s	[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state5 state5s	[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state6 state6s	[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state7 state7s	[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state8 state8s	[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state9 state9s	[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<1];
#define STATE_MEMBERS_14(alignment)\
	K8_ALIGN(alignment) state1 state1s	[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state2 state2s	[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state3 state3s	[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state4 state4s	[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state5 state5s	[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state6 state6s	[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state7 state7s	[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state8 state8s	[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state9 state9s	[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<1];
#define STATE_MEMBERS_15(alignment)\
	K8_ALIGN(alignment) state1 state1s	[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state2 state2s	[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state3 state3s	[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state4 state4s	[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state5 state5s	[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state6 state6s	[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state7 state7s	[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state8 state8s	[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state9 state9s	[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<1];
#define STATE_MEMBERS_16(alignment)\
	K8_ALIGN(alignment) state1 state1s	[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state2 state2s	[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state3 state3s	[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state4 state4s	[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state5 state5s	[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state6 state6s	[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state7 state7s	[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state8 state8s	[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state9 state9s	[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<1];
#define STATE_MEMBERS_17(alignment)\
	K8_ALIGN(alignment) state1 state1s	[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state2 state2s	[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state3 state3s	[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state4 state4s	[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state5 state5s	[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state6 state6s	[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state7 state7s	[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state8 state8s	[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state9 state9s	[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<1];


#define STATE_MEMBERS_18(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<15];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<14];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<13];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<12];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<11];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<10];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<9];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<1];

#define STATE_MEMBERS_19(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<15];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<14];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<13];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<12];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<11];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<10];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<1];

#define STATE_MEMBERS_20(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<15];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<14];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<13];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<12];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<11];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<9];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<8];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<7];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<6];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<5];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<4];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<3];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<2];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<1];

#define STATE_MEMBERS_21(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<15];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<14];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<13];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<12];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 1];

#define STATE_MEMBERS_22(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<15];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<14];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<13];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 1];

#define STATE_MEMBERS_23(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<15];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<14];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 1];
#define STATE_MEMBERS_24(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<15];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 1];

#define STATE_MEMBERS_25(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<16];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 1];

#define STATE_MEMBERS_26(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<17];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 1];

#define STATE_MEMBERS_27(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<18];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 1];

#define STATE_MEMBERS_28(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<19];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 1];

#define STATE_MEMBERS_29(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<28];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<20];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<19];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state28 state28s[(ssize_t)1<< 1];

#define STATE_MEMBERS_30(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<29];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<28];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<21];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<20];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<19];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state28 state28s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state29 state29s[(ssize_t)1<< 1];

#define STATE_MEMBERS_31(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<30];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<29];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<28];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<22];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<21];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<20];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<19];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state28 state28s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state29 state29s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state30 state30s[(ssize_t)1<< 1];

#define STATE_MEMBERS_32(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<31];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<30];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<29];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<28];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<23];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<22];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<21];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<20];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<19];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state28 state28s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state29 state29s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state30 state30s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state31 state31s[(ssize_t)1<< 1];

#define STATE_MEMBERS_33(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<32];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<31];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<30];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<29];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<28];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<24];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<23];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<22];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<21];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<20];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<19];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state28 state28s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state29 state29s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state30 state30s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state31 state31s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state32 state32s[(ssize_t)1<< 1];

#define STATE_MEMBERS_34(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<33];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<32];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<31];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<30];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<29];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<28];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<25];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<24];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<23];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<22];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<21];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<20];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<19];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state28 state28s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state29 state29s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state30 state30s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state31 state31s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state32 state32s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state33 state33s[(ssize_t)1<< 1];

#define STATE_MEMBERS_35(alignment)\
	K8_ALIGN(alignment) state1 state1s  [(ssize_t)1<<34];\
	K8_ALIGN(alignment) state2 state2s  [(ssize_t)1<<33];\
	K8_ALIGN(alignment) state3 state3s  [(ssize_t)1<<32];\
	K8_ALIGN(alignment) state4 state4s  [(ssize_t)1<<31];\
	K8_ALIGN(alignment) state5 state5s  [(ssize_t)1<<30];\
	K8_ALIGN(alignment) state6 state6s  [(ssize_t)1<<29];\
	K8_ALIGN(alignment) state7 state7s  [(ssize_t)1<<28];\
	K8_ALIGN(alignment) state8 state8s  [(ssize_t)1<<27];\
	K8_ALIGN(alignment) state9 state9s  [(ssize_t)1<<26];\
	K8_ALIGN(alignment) state10 state10s[(ssize_t)1<<25];\
	K8_ALIGN(alignment) state11 state11s[(ssize_t)1<<24];\
	K8_ALIGN(alignment) state12 state12s[(ssize_t)1<<23];\
	K8_ALIGN(alignment) state13 state13s[(ssize_t)1<<22];\
	K8_ALIGN(alignment) state14 state14s[(ssize_t)1<<21];\
	K8_ALIGN(alignment) state15 state15s[(ssize_t)1<<20];\
	K8_ALIGN(alignment) state16 state16s[(ssize_t)1<<19];\
	K8_ALIGN(alignment) state17 state17s[(ssize_t)1<<18];\
	K8_ALIGN(alignment) state18 state18s[(ssize_t)1<<17];\
	K8_ALIGN(alignment) state19 state19s[(ssize_t)1<<16];\
	K8_ALIGN(alignment) state20 state20s[(ssize_t)1<<15];\
	K8_ALIGN(alignment) state21 state21s[(ssize_t)1<<14];\
	K8_ALIGN(alignment) state22 state22s[(ssize_t)1<<13];\
	K8_ALIGN(alignment) state23 state23s[(ssize_t)1<<12];\
	K8_ALIGN(alignment) state24 state24s[(ssize_t)1<<11];\
	K8_ALIGN(alignment) state25 state25s[(ssize_t)1<<10];\
	K8_ALIGN(alignment) state26 state26s[(ssize_t)1<< 9];\
	K8_ALIGN(alignment) state27 state27s[(ssize_t)1<< 8];\
	K8_ALIGN(alignment) state28 state28s[(ssize_t)1<< 7];\
	K8_ALIGN(alignment) state29 state29s[(ssize_t)1<< 6];\
	K8_ALIGN(alignment) state30 state30s[(ssize_t)1<< 5];\
	K8_ALIGN(alignment) state31 state31s[(ssize_t)1<< 4];\
	K8_ALIGN(alignment) state32 state32s[(ssize_t)1<< 3];\
	K8_ALIGN(alignment) state33 state33s[(ssize_t)1<< 2];\
	K8_ALIGN(alignment) state34 state34s[(ssize_t)1<< 1];
typedef unsigned char BYTE;

#define STATE_ZERO {{0}}

#define KNLB_NO_OP(n, alignment)\
typedef union{\
  K8_ALIGN(alignment) BYTE state[(ssize_t)1<<(n-1)];\
  STATE_MEMBERS(n, alignment);\
} state##n;\
typedef state##n 	(* kernelb##n )( state##n);\
typedef void 		(* kernelpb##n )( state##n*);\
static inline state##n state##n##_zero() {return (state##n)STATE_ZERO;}\
static inline state##n mem_to_state##n(void* p){state##n a; memcpy(a.state, p, (ssize_t)1<<(n-1)); return a;}\
static inline void mem_to_statep##n(void* p, state##n *a){memcpy(a->state, p, (ssize_t)1<<(n-1));}\
static inline void k_nullpb##n(state##n *c){c = NULL; c++; return;}\
static inline state##n k_nullb##n(state##n c){return c;}\
/*inline kernelpb call*/\
static inline state##n ikpb##n(state##n s, kernelpb##n func){\
	func(&s);\
	return s;\
}\
static inline void state_bigswap##n(state##n *a, state##n *b){\
	for(ssize_t i = 0; i < ((ssize_t)1<<(n-1)); i++)\
	{BYTE temp = a->state[i];\
		a->state[i] = b->state[i];\
		b->state[i] = temp;\
	}\
}\
static inline void state_smallswap##n(state##n *a, state##n *b){\
	state##n c;\
	c = *a;\
	*a = *b;\
	*b = c;\
}\
static inline void state_swap##n(state##n *a, state##n *b){\
	if(n < 17)\
		state_smallswap##n(a,b);\
	else\
		state_bigswap##n(a,b);\
}

#define ACCESS_MASK(nn, nm) (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)) - 1)


#define k_at(arr, i, n, nm) ((state##nm)arr).state##n##s[i & ACCESS_MASK(n, nm)]
#define k_pat(arr, i, n, nm) ((state##nm)*arr).state##n##s[i & ACCESS_MASK(n, nm)]
#define k_off(arr, i, n, nm) (((state##nm)arr).state##n##s + (i & ACCESS_MASK(n, nm)))
//For consistency.
#define k_poff(arr, i, n, nm) (((state##nm)*arr).state##n##s + (i & ACCESS_MASK(n, nm)))

#define KNLB(n, alignment)\
KNLB_NO_OP(n, alignment)\
/*perform the operation between the two halves and return it*/\
static inline void k_and##n (state##n *a){\
	for(ssize_t i = 0; i < ((ssize_t)1<<(n-1))/2; i++)\
		k_pat(a, i, 1, n).u = k_pat(a, i, 1, n).u &  	k_pat(a, i+((ssize_t)1<<(n-2)), 1, n).u;\
}\
static inline void k_or##n (state##n *a){\
	for(ssize_t i = 0; i < ((ssize_t)1<<(n-1))/2; i++)\
		k_pat(a, i, 1, n).u = k_pat(a, i, 1, n).u | 	k_pat(a, i+((ssize_t)1<<(n-2)), 1, n).u;\
}\
static inline void k_xor##n (state##n *a){\
	for(ssize_t i = 0; i < ((ssize_t)1<<(n-1))/2; i++)\
		k_pat(a, i, 1, n).u = k_pat(a, i, 1, n).u ^ 	k_pat(a, i+((ssize_t)1<<(n-2)), 1, n).u;\
}\
static inline void k_byteswap##n (state##n *a){\
	for(ssize_t i = 0; i < ((ssize_t)1<<(n-1))/2; i++){\
		uint8_t c = k_pat(a, i, 1, n).u;\
		k_pat(a, i, 1, n).u = k_pat(a, ((ssize_t)1<<(n-1))-1-i, 1, n).u;\
		k_pat(a, ((ssize_t)1<<(n-1))-1-i, 1, n).u = c;\
	}\
}\
static inline void k_endian_cond_byteswap##n (state##n *a){\
	volatile const ssize_t i = 1;\
	if(*((char*)&i))\
		k_byteswap##n(a);\
}

//Define functions which need to know nn and nm.
#define KNLCONV(nn, nm)\
/*Retrieve the highest precision bits*/\
static inline state##nm statemix##nn(state##nn a, state##nn b){\
	state##nm ret;\
	k_at(ret, 0, nn, nm) = a;\
	k_at(ret, 1, nn, nm) = b;\
	return ret;\
}\
static inline void statemixp##nn(state##nn *a, state##nn *b, state##nm *ret){\
	k_pat(ret, 0, nn, nm) = *a;\
	k_pat(ret, 1, nn, nm) = *b;\
}\
/*Duplicate */\
static inline state##nm statedup##nn(state##nn a){\
	return statemix##nn(a,a);\
}\
static inline state##nn state_high##nm(state##nm a){\
	return k_at(a, 0, nn, nm);\
}\
static inline void state_highp##nm(state##nm *a, state##nn *ret){\
	*ret = k_pat(a, 0, nn, nm);\
}\
/*Retrieve the lowest precision bits*/\
static inline state##nn state_low##nm(state##nm a){\
	return k_at(a, 1, nn, nm);\
}\
static inline void state_lowp##nm(state##nm *a, state##nn *ret){\
	*ret =k_pat(a, 1, nn, nm);\
}\
static inline state##nn* state_ptr_low##nm(state##nm *a){\
	return k_poff(a, 1, nn, nm);\
}\
static inline state##nn* state_ptr_high##nm(state##nm *a){\
	return k_poff(a, 0, nn, nm);\
}\
/*Kernels*/\
/*Swap the upper and lower halves.*/\
static inline void k_smallswap##nm(state##nm *a){\
	state##nn c = k_pat(a, 0, nn, nm);\
	k_pat(a, 0, nn, nm) = k_pat(a, 1, nn, nm);\
	k_pat(a, 1, nn, nm) = c;\
}\
/*Large swap*/\
static inline void k_bigswap##nm(state##nm *a){\
	for(ssize_t i = 0; i < (ssize_t)1<<(nn-1); i++){\
		uint8_t c = k_pat(a, 0, nn, nm).state[i];\
		k_pat(a, 0, nn, nm).state[i] = k_pat(a, 1, nn, nm).state[i];\
		k_pat(a, 1, nn, nm).state[i] = c;\
	}\
}\
/*swap for this type*/\
static inline void k_swap##nm(state##nm *a){\
	if(nm < 17) k_smallswap##nm(a);\
	else k_bigswap##nm(a);\
}\
/*VLINT- Very Large Integer*/\
/*The most significant bits are in lower end.*/\
static inline void k_vlint_add##nn(state##nm *q){\
	uint8_t carry = 0;\
	for(ssize_t i = 0; i < ((ssize_t)1<<(nn-1)); i++){\
		uint16_t a = k_pat(q, 0, nn, nm).state[i];\
		uint16_t b = k_pat(q, 1, nn, nm).state[i];\
		a += carry; carry = 0;\
		a += b;\
		k_pat(q, 0, nn, nm).state[i] = a & 255;\
		carry = a/256;\
	}\
}\
static inline void k_vlint_twoscomplement##nn(state##nn *q){\
	uint8_t carry = 1;\
	for(ssize_t i = 0; i < ((ssize_t)1<<(nn-1)); i++){\
		q->state[i] = ~q->state[i];\
		uint16_t a = q->state[i];\
		a+=carry; carry = 0;\
		q->state[i] = a & 255;\
		carry = a/256;\
	}\
}\
static inline void k_vlint_sub##nn(state##nm *q){\
	k_vlint_twoscomplement##nn(q->state##nn##s + 1);\
	k_vlint_add##nn(q);\
}\
static inline void k_vlint_shr1_##nn(state##nn *q){\
	uint8_t carry = 0;\
	for(ssize_t i = ((ssize_t)1<<(nn-1)) - 1; i >= 0; i--){\
		uint8_t nextcarry = (q->state[i] & 1)<<7;\
		q->state[i] /= 2;\
		q->state[i] |= carry;\
		carry = nextcarry;\
	}\
}\
static inline void k_vlint_shl1_##nn(state##nn *q){\
	uint8_t carry = 0;\
	for(ssize_t i = 0; i < ((ssize_t)1<<(nn-1)); i++){\
		uint8_t nextcarry = (q->state[i] & 128)/128;\
		q->state[i] *= 2;\
		q->state[i] |= carry;\
		carry = nextcarry;\
	}\
}\
/*Get a state from a string.*/\
static inline void state##nm##_from_string(char* str, state##nm *q){\
	ssize_t len = strlen(str);\
	if(len > (ssize_t)1<<(nm-1))\
		len =(ssize_t)1<<(nm-1);\
	memcpy(q->state, str, len);\
	q->state[((ssize_t)1<<(nm-1)) - 1] = '\0';\
}\
/*Take in a string, read this many bytes from a file. Binary and text versions.*/\
static inline void fk_io_rbfile_##nm(state##nm *q){\
	int32_t maxbytes = 0;\
	int32_t offbytes = 0;\
	ssize_t maxbytes_temp = 0;\
	ssize_t offbytes_temp = 0;\
			FILE* f = NULL; \
	/*Error out on invalid size.*/\
	if(nn == 1 || nn == 2 || nn == 3){*q = (state##nm)STATE_ZERO; return;}\
	/*Handle valid cases- state4*/\
	memcpy(&maxbytes, q->state + ((ssize_t)1<<(nn-1))	 , 4);\
	memcpy(&offbytes, q->state + ((ssize_t)1<<(nn-1)) + 4, 4);\
	maxbytes_temp = maxbytes;\
	offbytes_temp = offbytes;\
	if(maxbytes_temp > ((ssize_t)1<<(nm-1)) || maxbytes_temp < 0)\
		{*q = (state##nm)STATE_ZERO; return;}\
	if(offbytes_temp > ((ssize_t)1<<(nm-1)) || offbytes_temp < 0)\
		{*q = (state##nm)STATE_ZERO; return;}\
	q->state##nn##s[0].state[((ssize_t)1<<(nn-1)) - 1] = '\0'; /*The string.*/\
	K_IO\
		f = fopen((char*)q->state, "rb");\
	K_END_IO\
		if(!f) {*q = (state##nm)STATE_ZERO; return;}\
	K_IO\
		fseek(f, 0, SEEK_END);\
		ssize_t len = ftell(f) - offbytes;\
		fseek(f,offbytes,SEEK_SET);\
		if(len > maxbytes)\
			len = maxbytes;\
		if(len > 0)\
			fread(q->state, 1, len, f);\
		if(len <= 0) {*q = (state##nm)STATE_ZERO;}\
		fclose(f);\
	K_END_IO\
}\
static inline void fk_io_rtfile_##nm(state##nm *q){\
	int32_t maxbytes = 0;\
	int32_t offbytes = 0;\
	ssize_t maxbytes_temp = 0;\
	ssize_t offbytes_temp = 0;\
			FILE* f = NULL; \
	/*Error out on invalid size.*/\
	if(nn == 1 || nn == 2 || nn == 3) {*q = (state##nm)STATE_ZERO; return;}\
	/*Handle valid cases*/\
	if(nn > 3){\
		memcpy(&maxbytes, q->state + ((ssize_t)1<<(nn-1))	 , 4);\
		memcpy(&offbytes, q->state + ((ssize_t)1<<(nn-1)) + 4, 4);\
	}\
	maxbytes_temp = maxbytes;\
	offbytes_temp = offbytes;\
	if(maxbytes_temp > ((ssize_t)1<<(nm-1)) || maxbytes_temp < 0)\
		{*q = (state##nm)STATE_ZERO; return;}\
	if(offbytes_temp > ((ssize_t)1<<(nm-1)) || offbytes_temp < 0)\
		{*q = (state##nm)STATE_ZERO; return;}\
	q->state##nn##s[0].state[((ssize_t)1<<(nn-1)) - 1] = '\0'; /*The string.*/\
	K_IO\
		f = fopen((char*)q->state, "r");\
	K_END_IO\
		if(!f) {*q = (state##nm)STATE_ZERO; return;}\
	K_IO\
		fseek(f, 0, SEEK_END);\
		ssize_t len = ftell(f) - offbytes;\
		fseek(f,offbytes,SEEK_SET);\
		if(len > maxbytes)\
			len = maxbytes;\
		if(len > 0)\
			fread(q->state, 1, len, f);\
		else {*q = (state##nm)STATE_ZERO;}\
		fclose(f);\
	K_END_IO\
	q->state[((ssize_t)1<<(nm-1)) - 1] = '\0'; /*Text files must be null terminated strings.*/\
}\
/*Take in a string in the upper half, write the lower half to file.*/\
static inline void fk_io_wbfile_##nm(state##nm *q){\
	/*Error out on invalid size.*/\
	K8_CONST(q->state##nn##s[1]);\
	FILE* f = NULL;\
	if(nn == 1 || nn == 2){return;}\
	q->state##nn##s[0].state[((ssize_t)1<<(nn-1)) - 1] = '\0'; /*The string.*/\
	K_IO\
		f = NULL; \
		f = fopen((char*)q->state, "wb");\
	K_END_IO\
		if(!f) {return;}\
	K_IO\
		fwrite(q->state, 1, ((ssize_t)1<<(nn-1)), f);\
		fclose(f);\
	K_END_IO\
}\
/*Take in a string in the upper half, write the lower half to file.*/\
static inline void fk_io_wtfile_##nm(state##nm *q){\
	/*Error out on invalid size.*/\
	K8_CONST(q->state##nn##s[1]);\
	FILE* f = NULL;\
	if(nn == 1 || nn == 2){return;}\
	q->state##nn##s[0].state[((ssize_t)1<<(nn-1)) - 1] = '\0'; /*The string.*/\
	K_IO\
		f = NULL; \
		f = fopen((char*)q->state, "w");\
	K_END_IO\
		if(!f) {return;}\
	K_IO\
		fwrite(q->state, 1, ((ssize_t)1<<(nn-1)), f);\
		fclose(f);\
	K_END_IO\
}\
static inline void fk_io_print_##nm(state##nm *q){\
	q->state[((ssize_t)1<<(nm-1))-1] = '\0';\
	K_IO\
		fwrite(q->state, strlen((char*)q->state), 1, stdout);\
	K_END_IO\
}




//Iterate over an entire container calling a kernel.
#define K8_FOREACH(func, arr, nn, nm)\
for(ssize_t i = 0; i < ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)); i++)\
	arr.state##nn##s[i] = func(arr.state##nn##s[i]);

#define K8_PFOREACH(func, arr, nn, nm)\
for(ssize_t i = 0; i < ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)); i++)\
	arr->state##nn##s[i] = func(arr->state##nn##s[i]);

#define K8_FOREACHP(func, arr, nn, nm)\
for(ssize_t i = 0; i < ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)); i++)\
	func(arr.state##nn##s +i);

#define K8_PFOREACHP(func, arr, nn, nm)\
for(ssize_t i = 0; i < ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)); i++)\
	func(arr->state##nn##s +i);

#define TRAVERSAL_INTERN_FETCH(i, arr, nn, arb) TRAVERSAL_INTERN_FETCH_##arb(i,arr,nn)
#define TRAVERSAL_INTERN_FETCH_PP(i,arr,nn) state##nn *elem_##i = arr->state##nn##s + i;
#define TRAVERSAL_INTERN_FETCH_VP(i,arr,nn) state##nn *elem_##i = arr.state##nn##s + i;

//Traverse a portion of a container, using a variable.
//THIS IS WHAT C++'s "for each" should look like,
//it's bullshit that it doesn't look like this.
#define FORWARD_TRAVERSAL_ARB(arr, nn, nm, i, start_in, end_in, incr_in, arb)\
{\
const ssize_t start__##i = start_in;\
const ssize_t end__##i = end_in;\
const ssize_t incr__##i = (ssize_t)incr_in;\
K8_ASSERT(incr_in <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) && incr_in > 0);\
if(\
	/*Well-formed range of iteration- The loop will never access out-of-bounds.*/\
	start__##i <= end__##i && incr__##i >0 && /*Valid Forward traversal?*/\
	start__##i <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) && (start__##i >= 0) &&	/**/\
	end__##i <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) && (end__##i >= 0) 		/**/\
){\
for(ssize_t i = start__##i; i<end__##i; i+=incr__##i){\
TRAVERSAL_INTERN_FETCH(i, arr, nn, arb)

#define BACKWARD_TRAVERSAL_ARB(arr, nn, nm, i, start_in, end_in, incr_in, arb)\
{\
const ssize_t start__##i = start_in;\
const ssize_t end__##i = end_in;\
const ssize_t incr__##i = incr_in;\
K8_ASSERT(incr_in <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) && incr_in > 0);\
if(\
	/*Well-formed range of iteration- The loop will never access out-of-bounds.*/\
	start__##i >= end__##i && incr__##i >0 && /*Valid backward traversal?*/\
	start__##i < (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) && (start__##i >= 0) &&	/*Notice Less than, not Less than or equal*/\
	end__##i <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) && (end__##i >= -1) 	/**/\
){\
for(ssize_t i = start__##i; i>end__##i; i-=incr__##i){\
TRAVERSAL_INTERN_FETCH(i, arr, nn, arb)

#define FORWARD_TRAVERSAL(arr, nn, nm, i, start, end, incr)\
FORWARD_TRAVERSAL_ARB(arr, nn, nm, i, start, end, incr, VP)

#define FORWARD_PTRAVERSAL(arr, nn, nm, i, start, end, incr)\
FORWARD_TRAVERSAL_ARB(arr, nn, nm, i, start, end, incr, PP)

#define BACKWARD_TRAVERSAL(arr, nn, nm, i, start, end, incr)\
BACKWARD_TRAVERSAL_ARB(arr, nn, nm, i, start, end, incr, VP)

#define BACKWARD_PTRAVERSAL(arr, nn, nm, i, start, end, incr)\
BACKWARD_TRAVERSAL_ARB(arr, nn, nm, i, start, end, incr, PP)

#define TRAVERSAL_END }} else {\
		K8_DEBUG_PRINT("\nK8_DEBUG: TRAVERSAL uses invalid range.");\
		K8_ASSERT(0);}\
}

#define K8_CHAINP(name, func1, func2, n)\
static inline void name(state##n *c){\
	func1(c);\
	func2(c);\
}

#define K8_CHAIN(name, func1, func2, n)\
static inline void name(state##n *c){\
	*c = func1(*c);\
	*c = func2(*c);\
}

#define K8_MULTIPLEX_CALLP(iscopy, func, nn) K8_MULTIPLEX_CALLP_##iscopy(func, nn)
#define K8_MULTIPLEX_CALLP_1(func, nn) a->state##nn##s[i] = func(a->state##nn##s[i]);
#define K8_MULTIPLEX_CALLP_0(func, nn) func(a->state##nn##s + i);

//Multiplex a low level kernel to a higher level.
//The most basic implementation, with parallelism.
//The last argument specifies what type of kernel it is-
//if it is a type1 or type 2 kernel.
//These macros ALWAYS produce pointer kernels, they produce better bytecode.

#define K8_MULTIPLEX_PARTIAL_ALIAS(name, func, nn, nm, start, end, iscopy, alias)\
static inline void name(state##nm *a){\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) );\
	PRAGMA_##alias\
	for(ssize_t i = start; i < end; i++)\
		K8_MULTIPLEX_CALLP(iscopy, func, nn);\
}

#define K8_MULTIPLEX_PARTIAL(name, func, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_PARTIAL_ALIAS(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy, PARALLEL)

#define K8_MULTIPLEX(name, func, nn, nm, iscopy)\
K8_MULTIPLEX_PARTIAL(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

//SUPER parallel
#define K8_MULTIPLEX_PARTIAL_SUPARA(name, func, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_PARTIAL_ALIAS(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy, SUPARA)

#define K8_MULTIPLEX_SUPARA(name, func, nn, nm, iscopy)\
K8_MULTIPLEX_PARTIAL_SUPARA(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)


#define K8_MULTIPLEX_PARTIAL_SIMD(name, func, nn, nm, start, end, iscopy)\
	K8_MULTIPLEX_PARTIAL_ALIAS(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy, SIMD)

#define K8_MULTIPLEX_SIMD(name, func, nn, nm, iscopy)\
	K8_MULTIPLEX_PARTIAL_SIMD(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_MULTIPLEX_PARTIAL_NP(name, func, nn, nm, start, end, iscopy)\
	K8_MULTIPLEX_PARTIAL_ALIAS(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy, NOPARALLEL)

#define K8_MULTIPLEX_NP(name, func, nn, nm, iscopy)\
	K8_MULTIPLEX_PARTIAL_NP(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

//pointer version
#define K8_MULTIPLEX_ICALLP(iscopy, func) K8_MULTIPLEX_ICALLP_##iscopy(func)
#define K8_MULTIPLEX_ICALLP_1(func) current_indexed = func(current_indexed);
#define K8_MULTIPLEX_ICALLP_0(func) func(&current_indexed);

//Multiplex a low level kernel to a higher level, with index in the upper half.
//Your kernel must operate on statennn but the input array will be treated as statenn's
#define K8_MULTIPLEX_INDEXED_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, alias)\
static inline void name(state##nm *a){\
	state##nn current, index; \
	state##nnn current_indexed;\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	K8_STATIC_ASSERT(nnn == (nn + 1));\
	PRAGMA_##alias\
	for(ssize_t i = start; i < end; i++)\
	{\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		current = a->state##nn##s[i];\
		if(nn == 1)/*Single byte indices.*/\
			memcpy(index.state, &ind8, 1);\
		else if (nn == 2)/*Two byte indices*/\
			memcpy(index.state, &ind16, 2);\
		else if (nn == 3)/*Three byte indices*/\
			memcpy(index.state, &ind32, 4);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed.state##nn##s[0] = index;\
		current_indexed.state##nn##s[1] = current;\
		K8_MULTIPLEX_ICALLP(iscopy, func);\
		/*Run the function on the indexed thing and return the low */\
		current = current_indexed.state##nn##s[1];\
		memcpy(a->state + i*((ssize_t)1<<(nn-1)), current.state, ((ssize_t)1<<(nn-1)) );\
	}\
}

#define K8_MULTIPLEX_INDEXED_PARTIAL(name, func, nn, nnn, nm, start, end, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, PARALLEL)

#define K8_MULTIPLEX_INDEXED_PARTIAL_SUPARA(name, func, nn, nnn, nm, start, end, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, SUPARA)

#define K8_MULTIPLEX_INDEXED_PARTIAL_SIMD(name, func, nn, nnn, nm, start, end, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, SIMD)

#define K8_MULTIPLEX_INDEXED_PARTIAL_NP(name, func, nn, nnn, nm, start, end, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, NOPARALLEL)

	

#define K8_MULTIPLEX_INDEXED(name, func, nn, nnn, nm, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_MULTIPLEX_INDEXED_SUPARA(name, func, nn, nnn, nm, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL_SUPARA(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_MULTIPLEX_INDEXED_SIMD(name, func, nn, nnn, nm, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL_SIMD(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_MULTIPLEX_INDEXED_NP(name, func, nn, nnn, nm, iscopy)\
	K8_MULTIPLEX_INDEXED_PARTIAL_NP(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)





#define K8_SHUFFLE_CALL(func, iscopy) K8_SHUFFLE_CALL_##iscopy (func)
#define K8_SHUFFLE_CALL_1(func) index = func(index);
#define K8_SHUFFLE_CALL_0(func) func(&index);

#define K8_SHUFFLE_IND32_PARTIAL(name, func, nn, nm, start, end, iscopy)\
static inline void name(state##nm* a){\
	state##nm ret;\
	state3 index; \
	const size_t emplacemask = ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)) - 1;\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	for(ssize_t i = start; i < end; i++)\
	{\
		index=to_state3(i);\
		K8_SHUFFLE_CALL(func, iscopy);\
		ret.state##nn##s[from_state3(index) & emplacemask] = \
		a->state##nn##s[i];\
	}\
	*a = ret;\
}

#define K8_SHUFFLE_IND32(name, func, nn, nm, iscopy)\
K8_SHUFFLE_IND32_PARTIAL(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_SHUFFLE_IND16_PARTIAL(name, func, nn, nm, start, end, iscopy)\
static inline void name(state##nm* a){\
	state##nm ret;\
	state2 index; \
	const size_t emplacemask = ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)) - 1;\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	for(ssize_t i = start; i < end; i++)\
	{	\
		index=to_state2(i);\
		K8_SHUFFLE_CALL(func, iscopy);\
		ret.state##nn##s[from_state3(index) & emplacemask] = \
		a->state##nn##s[i];\
	}\
	*a = ret;\
}

#define K8_SHUFFLE_IND16(name, func, nn, nm, iscopy)\
K8_SHUFFLE_IND16_PARTIAL(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_SHUFFLE_IND8_PARTIAL(name, func, nn, nm, start, end, iscopy)\
static inline void name(state##nm* a){\
	state##nm ret;\
	state1 index; \
	const size_t emplacemask = ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)) - 1;\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	for(ssize_t i = start; i < end; i++)\
	{\
		index=to_state1(i);\
		K8_SHUFFLE_CALL(func, iscopy);\
		ret.state##nn##s[from_state3(index) & emplacemask] = \
		a->state##nn##s[i];\
	}\
	*a = ret;\
}

#define K8_SHUFFLE_IND8(name, func, nn, nm, iscopy)\
K8_SHUFFLE_IND8_PARTIAL(name, func, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)




/*Multiplex a low level kernel to a higher level, extracting elements and putting the index
in the upper half.

The index returned in the upper half is used to place in the result.
*/
#define K8_MULTIPLEX_INDEXED_EMPLACE_PARTIAL(name, func, nn, nnn, nm, start, end, iscopy)\
static inline void name(state##nm *a){\
	state##nm ret;\
	state##nn current, index; \
	state##nnn current_indexed;\
	memcpy(&ret, a, sizeof(state##nm));\
	const size_t emplacemask = ((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)) - 1;\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	K8_STATIC_ASSERT(nnn == (nn + 1));\
	for(ssize_t i = start; i < end; i++)\
	{\
		uint32_t ind32 = i; uint16_t ind16 = i; uint8_t ind8 = i;\
		current = a->state##nn##s[i];\
		if(nn == 1)/*Single byte indices.*/\
			index = mem_to_state##nn(&ind8);\
		else if (nn == 2)/*Two byte indices*/\
			index = mem_to_state##nn(&ind16);\
		else if (nn == 3)/*Three byte indices*/\
			index = mem_to_state##nn(&ind32);\
		else	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(index.state, &ind32, 4);\
		/*We have the current and the index, combine them.*/\
		current_indexed.state##nn##s[0] = index;\
		current_indexed.state##nn##s[1] = current;\
		/*Run the function on the indexed thing and return the low */\
		K8_MULTIPLEX_ICALLP(iscopy, func);\
		index = current_indexed.state##nn##s[0];\
		current = current_indexed.state##nn##s[1];\
		if(nn == 1){/*Single byte indices.*/\
			memcpy(&ind8, index.state, 1);\
			ind8 &= emplacemask;\
			memcpy(ret.state + ind8*((ssize_t)1<<(nn-1)), current.state, ((ssize_t)1<<(nn-1)) );\
		}else if (nn == 2){/*Two byte indices*/\
			memcpy(&ind16, index.state, 2);\
			ind16 &= emplacemask;\
			memcpy(ret.state + ind16*((ssize_t)1<<(nn-1)), current.state, ((ssize_t)1<<(nn-1)) );\
		}else if (nn == 3){/*Three byte indices*/\
			memcpy(&ind32, index.state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret.state + ind32*((ssize_t)1<<(nn-1)), current.state, ((ssize_t)1<<(nn-1)) );\
		}else{	/*We must copy the 32 bit index into the upper half.*/\
			memcpy(&ind32, index.state, 4);\
			ind32 &= emplacemask;\
			memcpy(ret.state + ind32*((ssize_t)1<<(nn-1)), current.state, ((ssize_t)1<<(nn-1)) );\
		}\
	}\
	memcpy(a, &ret, sizeof(state##nm));\
}

#define K8_MULTIPLEX_INDEXED_EMPLACE(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_INDEXED_EMPLACE_PARTIAL(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1))/((ssize_t)1<<(nn-1))), iscopy)


//The shared state function.
//func must take in nnn of state.
//the very first statenn within the statenm is considered "shared"
//every single nn thereafter is looped over.
//The shared state and the i'th statenn is merged into a single statennn
//which is passed to your function.
//nnn must be nn + 1
//The shared state is presumed to be very large, so this is all done with pointers and heap memory.
//All that said, you *can* pass a copy-kernel.
#define K8_SHARED_CALL(iscopy, func) K8_SHARED_CALL_##iscopy(func)
#define K8_SHARED_CALL_1(func) passed = func(passed);
#define K8_SHARED_CALL_0(func) func(&passed);


//the parameters nwind, whereind, doind specify
//where in the shared state to write
//the current index,
//but only if "doind" is one.
#define K8_SHARED_STATE_PARTIAL_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy)\
static inline void name(state##nm *a){\
	state##nnn passed; state##nwind saved;\
	passed.state##nn##s[0] = a->state##nn##s[sharedind];\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))) );\
	K8_STATIC_ASSERT(nnn == (nn + 1));\
	K8_STATIC_ASSERT(!(sharedind >= start && sharedind < end));\
	K8_STATIC_ASSERT(whereind >= 0);\
	K8_STATIC_ASSERT(nwind <= nn);\
	K8_STATIC_ASSERT(whereind >= 0);\
	K8_STATIC_ASSERT(whereind < (((ssize_t)1<<(nn-1)) / ((ssize_t)1<<(nwind-1))) );/*There's actually a spot.*/\
	if(doind) saved = passed.state##nn##s[0].state##nwind##s[whereind];/*Don't lose data!*/\
	for(ssize_t i = start; i < end; i++){\
		passed.state##nn##s[1] = a->state##nn##s[i];\
		if(doind){\
			state##nwind index; index.u = i;\
			memcpy(passed.state##nn##s[0].state##nwind##s + whereind, index.state, sizeof(index));\
		}\
		K8_SHARED_CALL(iscopy, func)\
		a->state##nn##s[i] = passed.state##nn##s[1];\
	}\
	if(doind){ /*Write back the useful data.*/\
		passed.state##nn##s[0].state##nwind##s[whereind] = saved;\
	}\
	memcpy(a->state + sharedind, passed.state, sizeof(state##nn));\
}

//WIND version.
#define K8_SHARED_STATE_WIND(name, func, nn, nnn, nm,              					   nwind, whereind, doind, iscopy)\
K8_SHARED_STATE_PARTIAL_WIND(name, func, nn, nnn, nm, 1, (((ssize_t)1<<(nm-1))/((ssize_t)1<<(nn-1))), 0, nwind, whereind, doind, iscopy)

#define K8_SHARED_STATE_PARTIAL(name, func, nn, nnn, nm, start, end, sharedind, iscopy)\
 K8_SHARED_STATE_PARTIAL_WIND(name, func, nn, nnn, nm, start, end, sharedind, 1, 0, 0, iscopy)

#define K8_SHARED_STATE(name, func, nn, nnn, nm, iscopy)\
K8_SHARED_STATE_PARTIAL(name, func, nn, nnn, nm, 1, (((ssize_t)1<<(nm-1))/((ssize_t)1<<(nn-1))), 0, iscopy)

//Variant in which the shared state is "read only"

#define K8_RO_SHARED_STATE_PARTIAL_ALIAS_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy, alias)\
static inline void name(state##nm *a){\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));/*End is valid*/\
	K8_STATIC_ASSERT(nnn == (nn + 1));\
	K8_STATIC_ASSERT(!(sharedind >= start && sharedind < end));\
	K8_STATIC_ASSERT(whereind >= 0);\
	K8_STATIC_ASSERT(nwind <= nn);\
	K8_STATIC_ASSERT(whereind >= 0);\
	K8_STATIC_ASSERT(whereind < (((ssize_t)1<<(nn-1)) / ((ssize_t)1<<(nwind-1))) );/*There's actually a spot.*/\
	K8_CONST(a->state##nn##s[sharedind]);\
	PRAGMA_##alias\
	for(ssize_t i = start; i < end; i++){\
		state##nnn passed;\
		passed.state##nn##s[0] = a->state##nn##s[sharedind];\
		if(doind){\
			state##nwind index; index.u = i;\
			memcpy(passed.state##nn##s[0].state##nwind##s + whereind, index.state, sizeof(index));\
		}\
		passed.state##nn##s[1] = a->state##nn##s[i];\
		K8_SHARED_CALL(iscopy, func)\
		a->state##nn##s[i] = passed.state##nn##s[1];\
	}\
}


//Define WIND variants.
#define K8_RO_SHARED_STATE_PARTIAL_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy, PARALLEL)

#define K8_RO_SHARED_STATE_PARTIAL_WIND_SUPARA(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy, SUPARA)

#define K8_RO_SHARED_STATE_PARTIAL_WIND_SIMD(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy, SIMD)

#define K8_RO_SHARED_STATE_PARTIAL_WIND_NP(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy, NOPARALLEL)



#define K8_RO_SHARED_STATE_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, sharedind, iscopy, fuck)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS_WIND(name, func, nn, nnn, nm, start, end, sharedind, 1, 0, 0, iscopy, fuck)



#define K8_RO_SHARED_STATE_PARTIAL(name, func, nn, nnn, nm, start, end, sharedind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, sharedind, iscopy, PARALLEL)

#define K8_RO_SHARED_STATE_PARTIAL_SUPARA(name, func, nn, nnn, nm, start, end, sharedind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, sharedind, iscopy, SUPARA)


#define K8_RO_SHARED_STATE(name, func, nn, nnn, nm, iscopy)\
K8_RO_SHARED_STATE_PARTIAL(name, func, nn, nnn, nm, 1, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), 0, iscopy)

#define K8_RO_SHARED_STATE_SUPARA(name, func, nn, nnn, nm, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_SUPARA(name, func, nn, nnn, nm, 1, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), 0, iscopy)



#define K8_RO_SHARED_STATE_PARTIAL_NP(name, func, nn, nnn, nm, start, end, sharedind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, sharedind, iscopy, NOPARALLEL)

#define K8_RO_SHARED_STATE_NP(name, func, nn, nnn, nm, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_NP(name, func, nn, nnn, nm, 1, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), 0, iscopy)

//Variant in which the shared state is "read only"

#define K8_RO_SHARED_STATE_PARTIAL_SIMD(name, func, nn, nnn, nm, start, end, sharedind, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, sharedind, iscopy, SIMD)

#define K8_RO_SHARED_STATE_SIMD(name, func, nn, nnn, nm, iscopy)\
K8_RO_SHARED_STATE_PARTIAL_SIMD(name, func, nn, nnn, nm, 1, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), 0, iscopy)

#define K8_MHALVES_CALLP(iscopy, func) K8_MHALVES_CALLP_##iscopy(func)
#define K8_MHALVES_CALLP_1(func) passed = func(passed);
#define K8_MHALVES_CALLP_0(func) func(&passed);


#define K8_MHALVES_CALL(iscopy, func) K8_MHALVES_CALL_##iscopy(func)
#define K8_MHALVES_CALL_1(func) passed = func(passed);
#define K8_MHALVES_CALL_0(func) func(&passed);
//Multiplex on halves.
#define K8_MULTIPLEX_HALVES_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, alias)\
static inline void name(state##nm *a){\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= ((((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1)))/2));\
	K8_STATIC_ASSERT(nnn == (nn + 1));\
	PRAGMA_##alias\
	for(ssize_t i = start; i < end; i++){\
		state##nnn passed;\
		passed.state##nn##s[0] = state_ptr_high##nm(a)->state##nn##s[i];\
		passed.state##nn##s[1] = state_ptr_low##nm(a)->state##nn##s[i];\
		K8_MHALVES_CALLP(iscopy, func)\
		state_ptr_high##nm(a)->state##nn##s[i] = passed.state##nn##s[0];\
		state_ptr_low##nm(a)->state##nn##s[i] = passed.state##nn##s[1];\
	}\
}

#define K8_MULTIPLEX_HALVES_PARTIAL(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, PARALLEL)

#define K8_MULTIPLEX_HALVES(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL(name, func, nn, nnn, nm, 0, ((((ssize_t)1<<(nm-1))/((ssize_t)1<<(nn-1)))/2), iscopy)

#define K8_MULTIPLEX_HALVES_PARTIAL_SUPARA(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, SUPARA)

#define K8_MULTIPLEX_HALVES_SUPARA(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL_SUPARA(name, func, nn, nnn, nm, 0, ((((ssize_t)1<<(nm-1))/((ssize_t)1<<(nn-1)))/2), iscopy)

#define K8_MULTIPLEX_HALVES_PARTIAL_SIMD(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, SIMD)

#define K8_MULTIPLEX_HALVES_SIMD(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL_SIMD(name, func, nn, nnn, nm, 0, ((((ssize_t)1<<(nm-1))/((ssize_t)1<<(nn-1)))/2), iscopy)

#define K8_MULTIPLEX_HALVES_PARTIAL_NP(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, NOPARALLEL)

#define K8_MULTIPLEX_HALVES_NP(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_HALVES_PARTIAL_NP(name, func, nn, nnn, nm, 0, ((((ssize_t)1<<(nm-1))/((ssize_t)1<<(nn-1)))/2), iscopy)


#define K8_MULTIK8_CALL(iscopy, funcarr, nn) K8_MULTIK8_CALL_##iscopy(funcarr, nn)
#define K8_MULTIK8_CALL_1(funcarr, nn) a->state##nn##s[i] = (funcarr[i])(a->state##nn##s[i]);
#define K8_MULTIK8_CALL_0(funcarr, nn) (funcarr[i])(a->state##nn##s +i);
//Create a multiplexed kernel which taks in an array of function pointers
//

#define K8_MULTIPLEX_MULTIK8_PARTIAL_ALIAS(name, funcarr, nn, nm, start, end, iscopy, alias)\
static inline void name(state##nm *a){\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	PRAGMA_##alias\
	for(ssize_t i = start; i < end; i++)\
		K8_MULTIK8_CALL(iscopy, funcarr, nn);\
}

#define K8_MULTIPLEX_MULTIK8_PARTIAL(name, funcarr, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL_ALIAS(name, funcarr, nn, nm, start, end, iscopy, PARALLEL)

#define K8_MULTIPLEX_MULTIK8_PARTIAL_SUPARA(name, funcarr, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL_ALIAS(name, funcarr, nn, nm, start, end, iscopy, SUPARA)

#define K8_MULTIPLEX_MULTIKERNEL(name, funcarr, nn, nm, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL(name, funcarr, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_MULTIPLEX_MULTIK8_SUPARA(name, funcarr, nn, nm, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL_SUPARA(name, funcarr, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_MULTIPLEX_MULTIK8_PARTIAL_SIMD(name, funcarr, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL_ALIAS(name, funcarr, nn, nm, start, end, iscopy, SIMD)

#define K8_MULTIPLEX_MULTIK8_SIMD(name, funcarr, nn, nm, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL_SIMD(name, funcarr, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)

#define K8_MULTIPLEX_MULTIK8_PARTIAL_NP(name, funcarr, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL_ALIAS(name, funcarr, nn, nm, start, end, iscopy, NOPARALLEL)

#define K8_MULTIPLEX_MULTIK8_NP(name, funcarr, nn, nm, iscopy)\
K8_MULTIPLEX_MULTIK8_PARTIAL_NP(name, funcarr, nn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)






#define K8_MULTIPLEX_NLOGN_CALLP(func, iscopy) K8_MULTIPLEX_NLOGN_CALLP_##iscopy(func)
#define K8_MULTIPLEX_NLOGN_CALLP_1(func) current_b = func(current_b);
#define K8_MULTIPLEX_NLOGN_CALLP_0(func) func(&current_b);

//NLOGN implementation.
//Parallelism cannot be used.
#define K8_MULTIPLEX_NLOGN_PARTIAL(name, func, nn, nnn, nm, start, end, iscopy)\
static inline void name(state##nm *a){\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	K8_STATIC_ASSERT(nnn == (nn+1));\
	for(ssize_t i = start; i < end - 1; i++){\
		state##nnn current_b;\
		current_b.state##nn##s[0] = a->state##nn##s[i];\
		for(ssize_t j = i+1; j < end; j++)\
		{\
			current_b.state##nn##s[1] = a->state##nn##s[j];\
			K8_MULTIPLEX_NLOGN_CALLP(func, iscopy)\
			a->state##nn##s[j] = current_b.state##nn##s[1];\
		}\
		/*Write back elem i*/\
		a->state##nn##s[i] = current_b.state##nn##s[0];\
	}\
}
#define K8_MULTIPLEX_NLOGN(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_NLOGN_PARTIAL(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)


//NLOGN but parallel, the i element is considered "read only"
//This is useful in situations where you want NLOGN functionality, but you dont want to modify i element.
//Simd variant.
#define K8_MULTIPLEX_NLOGNRO_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, alias)\
static inline void name(state##nm *a){\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))));\
	K8_STATIC_ASSERT(nnn == (nn+1));\
	for(ssize_t i = start; i < end - 1; i++){\
		state##nn shared = a->state##nn##s[i];\
		PRAGMA_##alias\
		for(ssize_t j = i+1; j < end; j++)\
		{\
			state##nnn current_b;\
			current_b.state##nn##s[0] = shared;\
			current_b.state##nn##s[1] = a->state##nn##s[j];\
			K8_MULTIPLEX_NLOGN_CALLP(func, iscopy)\
			a->state##nn##s[j] = current_b.state##nn##s[1];\
		}\
	}\
}
#define K8_MULTIPLEX_NLOGNRO_PARTIAL_NP(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, NOPARALLEL)

#define K8_MULTIPLEX_NLOGNRO_NP(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL_NP(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)


#define K8_MULTIPLEX_NLOGNRO_PARTIAL(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, PARALLEL)

#define K8_MULTIPLEX_NLOGNRO(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)


#define K8_MULTIPLEX_NLOGNRO_PARTIAL_SUPARA(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, SUPARA)

#define K8_MULTIPLEX_NLOGNRO_SUPARA(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL_SUPARA(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)


#define K8_MULTIPLEX_NLOGNRO_PARTIAL_SIMD(name, func, nn, nnn, nm, start, end, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, SIMD)

#define K8_MULTIPLEX_NLOGNRO_SIMD(name, func, nn, nnn, nm, iscopy)\
K8_MULTIPLEX_NLOGNRO_PARTIAL_SIMD(name, func, nn, nnn, nm, 0, (((ssize_t)1<<(nm-1)) / ((ssize_t)1<<(nn-1))), iscopy)


#define K8_MULTIPLEX_DE_CALLP(func, iscopy) K8_MULTIPLEX_DE_CALLP_##iscopy(func)
#define K8_MULTIPLEX_DE_CALLP_1(func) data = func(data);
#define K8_MULTIPLEX_DE_CALLP_0(func) func(&data);


/*
Multiplex extracting arbitrary data

Extract "nproc" bytes and put it in a state##nn, 
which is then passed to func.
*/
#define K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_ALIAS(name, func, nproc, nn, nm, start, end, iscopy, alias)\
static inline void name(state##nm *a){\
	K8_STATIC_ASSERT(start >= 0);\
	K8_STATIC_ASSERT(start <= end);\
	K8_STATIC_ASSERT(end <= (((ssize_t)1<<(nm-1))-nproc+1) );\
	PRAGMA_##alias\
	for(ssize_t i = start; i < end; i += nproc){\
		state##nn data;\
		memcpy(data.state, a->state+i, nproc);\
		K8_MULTIPLEX_DE_CALLP(func, iscopy)\
		memcpy(a->state+i, data.state, nproc);\
	}\
}

/*Partials*/
#define K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL(name, func, nproc, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_ALIAS(name, func, nproc, nn, nm, start, end, iscopy, PARALLEL)

#define K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_SUPARA(name, func, nproc, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_ALIAS(name, func, nproc, nn, nm, start, end, iscopy, SUPARA)

#define K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_SIMD(name, func, nproc, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_ALIAS(name, func, nproc, nn, nm, start, end, iscopy, SIMD)

#define K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_NP(name, func, nproc, nn, nm, start, end, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_ALIAS(name, func, nproc, nn, nm, start, end, iscopy, NOPARALLEL)

/*Automatic start and end calculation*/
#define K8_MULTIPLEX_DATA_EXTRACTION(name, func, nproc, nn, nm, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL(name, func, nproc, nn, nm, 0, ((ssize_t)1<<(nm-1))-nproc+1, iscopy)

#define K8_MULTIPLEX_DATA_EXTRACTION_SUPARA(name, func, nproc, nn, nm, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_SUPARA(name, func, nproc, nn, nm, 0, ((ssize_t)1<<(nm-1))-nproc+1, iscopy)

#define K8_MULTIPLEX_DATA_EXTRACTION_SIMD(name, func, nproc, nn, nm, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_SIMD(name, func, nproc, nn, nm, 0, ((ssize_t)1<<(nm-1))-nproc+1, iscopy)

#define K8_MULTIPLEX_DATA_EXTRACTION_NP(name, func, nproc, nn, nm, iscopy)\
K8_MULTIPLEX_DATA_EXTRACTION_PARTIAL_NP(name, func, nproc, nn, nm, 0, ((ssize_t)1<<(nm-1))-nproc+1, iscopy)

#define K8_WRAP_OP2(name, n, nn)\
static inline state##nn kb_##name##_s##n(state##nn c) {k_##name##_s##n(&c); return c;}
#define K8_WRAP_OP1(name, n, nn)\
static inline state##n kb_##name##_s##n(state##n c) {k_##name##_s##n(&c); return c;}

#define K8_COMPLETE_ARITHMETIC(n, nn, bb)\
static inline void k_shl_s##n(state##nn *q){\
	uint##bb##_t a = from_state##n(q->state##n##s[0]);\
	uint##bb##_t b = from_state##n(q->state##n##s[1]);\
	b &= bb - 1;\
	a <<= b;\
	q->state##n##s[0] = to_state##n(a);\
}\
K8_WRAP_OP2(shl, n, nn);\
static inline void k_shr_s##n(state##nn *q){\
	uint##bb##_t a = from_state##n(q->state##n##s[0]);\
	uint##bb##_t b = from_state##n(q->state##n##s[1]);\
	b &= bb - 1;\
	a >>= b;\
	q->state##n##s[0] = to_state##n(a);\
}\
K8_WRAP_OP2(shr, n, nn);\
static inline void k_and_s##n(state##nn *q){\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) & from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(and, n, nn);\
static inline void k_or_s##n(state##nn *q){\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) | from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(or, n, nn);\
static inline void k_xor_s##n(state##nn *q){\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) ^ from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(xor, n, nn);\
static inline void k_add_s##n(state##nn *q){\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) + from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(add, n, nn);\
static inline void k_sub_s##n(state##nn *q){\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) - from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(sub, n, nn);\
static inline void k_mul_s##n(state##nn *q){\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) * from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(mul, n, nn);\
static inline void k_div_s##n(state##nn *q){\
	if(from_state##n(q->state##n##s[1]) == 0) {q->state##n##s[0] = to_state##n(0); return;}\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) / from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(div, n, nn);\
static inline void k_mod_s##n(state##nn *q){\
	if(from_state##n(q->state##n##s[1]) == 0) {q->state##n##s[0] = signed_to_state##n(0); return;}\
	q->state##n##s[0] = to_state##n( from_state##n(q->state##n##s[0]) % from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(mod, n, nn);\
static inline void k_sneg_s##n(state##n *q){\
	*q = signed_to_state##n( -1 * signed_from_state##n(*q));\
}\
K8_WRAP_OP1(sneg, n, nn);\
static inline void k_abs_s##n(state##n *q){\
	*q = signed_to_state##n( labs(signed_from_state##n(*q)) );\
}\
K8_WRAP_OP1(abs, n, nn);\
static inline void k_neg_s##n(state##n *q){\
	*q = to_state##n( ~(from_state##n(*q)) );\
}\
K8_WRAP_OP1(neg, n, nn);\
static inline void k_incr_s##n(state##n *q){\
	*q = to_state##n( (from_state##n(*q))+1 );\
}\
K8_WRAP_OP1(incr, n, nn);\
static inline void k_decr_s##n(state##n *q){\
	*q = to_state##n( (from_state##n(*q))-1 );\
}\
K8_WRAP_OP1(decr, n, nn);\
static inline void k_sadd_s##n(state##nn *q){\
	k_add_s##n(q);\
}\
K8_WRAP_OP2(sadd, n, nn);\
static inline void k_ssub_s##n(state##nn *q){\
	k_sub_s##n(q);\
}\
K8_WRAP_OP2(ssub, n, nn);\
static inline void k_smul_s##n(state##nn *q){\
	k_mul_s##n(q);\
}\
K8_WRAP_OP2(smul, n, nn);\
static inline void k_sdiv_s##n(state##nn *q){\
	if(signed_from_state##n(q->state##n##s[1]) == 0) {q->state##n##s[0] = signed_to_state##n(0); return;}\
	q->state##n##s[0] = to_state##n( signed_from_state##n(q->state##n##s[0]) / signed_from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(sdiv, n, nn);\
static inline void k_smod_s##n(state##nn *q){\
	if(signed_from_state##n(q->state##n##s[1]) == 0) {q->state##n##s[0] = signed_to_state##n(0); return; }\
	q->state##n##s[0] = to_state##n( signed_from_state##n(q->state##n##s[0]) % signed_from_state##n(q->state##n##s[1]) );\
}\
K8_WRAP_OP2(smod, n, nn);
/*

Implementer's note: acos and asin have a domain restriction,
[-1,1]

They are not implemented
*/

#define K8_COMPLETE_FLOATING_ARITHMETIC(n, nn, type)\
static inline void k_fadd_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(a+b);\
		return;\
	}\
	if((isfinite(a) && isfinite(b)))\
		q->state##n##s[0] = type##_to_state##n(a+b);\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fadd, n, nn);\
static inline void k_fsub_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(a-b);\
		return;\
	}\
	if(isfinite(a) && isfinite(b))\
		q->state##n##s[0] = type##_to_state##n(a-b);\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fsub, n, nn);\
static inline void k_fmul_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(a*b);\
		return;\
	}\
	if(isfinite(a) && isfinite(b))\
		q->state##n##s[0] = type##_to_state##n(a*b);\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fmul, n, nn);\
static inline void k_fdiv_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(a/b);\
		return;\
	}\
	if(isfinite(a) && isnormal(b))\
		q->state##n##s[0] = type##_to_state##n(a/b);\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fdiv, n, nn);\
static inline void k_fmod_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(fmod(a,b));\
		return;\
	}\
	if(isfinite(a) && isnormal(b))\
		q->state##n##s[0] = type##_to_state##n(fmod(a,b));\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fmod, n, nn);\
static inline void k_fmodf_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(fmodf(a,b));\
		return;\
	}\
	if(isfinite(a) && isnormal(b))\
		q->state##n##s[0] = type##_to_state##n(fmodf(a,b));\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fmodf, n, nn);\
static inline void k_fceil_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(ceil(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(ceil(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fceil, n, nn);\
static inline void k_fceilf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(ceilf(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(ceilf(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fceilf, n, nn);\
static inline void k_ffloor_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(floor(a));\
		return;\
	}\
	if(K8_FAST_FLOAT_MATH || isfinite(a))\
		*q = type##_to_state##n(floor(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(ffloor, n, nn);\
static inline void k_ffloorf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(floorf(a));\
		return;\
	}\
	if(K8_FAST_FLOAT_MATH || isfinite(a))\
		*q = type##_to_state##n(floorf(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(ffloorf, n, nn);\
static inline void k_fabs_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q= type##_to_state##n(fabs(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(fabs(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fabs, n, nn);\
static inline void k_fabsf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q= type##_to_state##n(fabsf(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(fabsf(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fabsf, n, nn);\
static inline void k_fsqrt_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(sqrt(fabs(a)));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(sqrt(fabs(a)));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fsqrt, n, nn);\
static inline void k_fsqrtf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(sqrtf(fabsf(a)));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(sqrtf(fabsf(a)));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fsqrtf, n, nn);\
static inline void k_fsin_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(sin(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(sin(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fsin, n, nn);\
static inline void k_fsinf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(sinf(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(sinf(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fsinf, n, nn);\
static inline void k_fcos_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(cos(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(cos(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fcos, n, nn);\
static inline void k_fcosf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(cosf(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(cosf(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fcosf, n, nn);\
static inline void k_ftan_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(tan(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(tan(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(ftan, n, nn);\
static inline void k_ftanf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(tanf(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(tanf(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(ftanf, n, nn);\
static inline void k_fatan_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(atan(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(atan(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fatan, n, nn);\
static inline void k_fatanf_s##n(state##n *q){\
	type a = type##_from_state##n(*q);\
	if(K8_FAST_FLOAT_MATH){\
		*q = type##_to_state##n(atanf(a));\
		return;\
	}\
	if(isfinite(a))\
		*q = type##_to_state##n(atanf(a));\
	else\
		*q = type##_to_state##n(0);\
}\
K8_WRAP_OP1(fatanf, n, nn);\
static inline void k_fatan2_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(atan2(a,b));\
		return;\
	}\
	if(isfinite(a) && isfinite(b))\
		q->state##n##s[0] = type##_to_state##n(atan2(a,b));\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fatan2, n, nn);\
static inline void k_fatan2f_s##n(state##nn *q){\
	type a = type##_from_state##n(q->state##n##s[0]);\
	type b = type##_from_state##n(q->state##n##s[1]);\
	if(K8_FAST_FLOAT_MATH){\
		q->state##n##s[0] = type##_to_state##n(atan2f(a,b));\
		return;\
	}\
	if(isfinite(a) && isfinite(b))\
		q->state##n##s[0] = type##_to_state##n(atan2f(a,b));\
	else\
		q->state##n##s[0] = type##_to_state##n(0);\
}\
K8_WRAP_OP2(fatan2f, n, nn);\
static inline void k_fsqr_s##n(state##n *q){\
	state##nn p;\
	p.state##n##s[0] = *q;\
	p.state##n##s[1] = *q;\
	k_fmul_s##n(&p);\
	*q = p.state##n##s[0];\
}\
K8_WRAP_OP1(fsqr, n, nn);\
static inline void k_fneg_s##n(state##n *q){\
	state##nn p;\
	p.state##n##s[0] = *q;\
	p.state##n##s[1] = type##_to_state##n(-1);\
	k_fmul_s##n(&p);\
	*q = p.state##n##s[0];\
}\
K8_WRAP_OP1(fneg, n, nn);

//There is no relevant op for 1.
KNLB_NO_OP(1,1);
//helper function.
static inline state1 to_state1(uint8_t a){
	state1 q; 
	memcpy(&q, &a, 1);
	return q;
}
static inline state1 signed_to_state1(int8_t a){
	state1 q;
	memcpy(&q, &a, 1);
	return q;
}

static inline uint8_t from_state1(state1 a){
	K8_STATIC_ASSERT(sizeof(uint8_t) == 1);
	uint8_t q;
	memcpy(&q, &a, 1);
	return q;
}
static inline int8_t signed_from_state1(state1 a){
	K8_STATIC_ASSERT(sizeof(int8_t) == 1);
	int8_t q;
	memcpy(&q, &a, 1);
	return q;
}

static inline void fk_io_getc(state1 *q){
	K_IO
		q->state[0] = fgetc(stdin);
	K_END_IO
}

//state2. Contains 2^(2-1) bytes, or 2 bytes.
KNLB(2,2);
//Conversion function to up from 1 byte to 2 bytes.
KNLCONV(1,2);
static inline state2 to_state2(uint16_t a){
	K8_STATIC_ASSERT(sizeof(uint16_t) == 2);
	state2 q;
	memcpy(&q, &a, 2);
	return q;
}
static inline state2 signed_to_state2(int16_t a){
	K8_STATIC_ASSERT(sizeof(int16_t) == 2);
	state2 q;
	memcpy(&q, &a, 2);
	return q;
}


static inline uint16_t from_state2(state2 a){
	K8_STATIC_ASSERT(sizeof(uint16_t) == 2);
	uint16_t q;
	memcpy(&q, &a, 2);
	return q;
}
static inline int16_t signed_from_state2(state2 a){
	K8_STATIC_ASSERT(sizeof(int16_t) == 2);
	int16_t q;
	memcpy(&q, &a, 2);
	return q;
}
K8_COMPLETE_ARITHMETIC(1,2, 8)


//state3. contains 4 bytes- so, most of your typical types go here.
KNLB(3,4);
KNLCONV(2,3);

static inline uint32_t from_state3(state3 a){
	uint32_t u;
	K8_STATIC_ASSERT(sizeof(uint32_t) == 4);
	memcpy(&u, &a, 4);
	return u;
}
static inline int32_t signed_from_state3(state3 a){
	int32_t i;
	K8_STATIC_ASSERT(sizeof(int32_t) == 4);
	memcpy(&i, &a, 4);
	return i;
}

static inline state3 to_state3(uint32_t a){
	state3 q; memcpy(&q, &a, 4);
	return q;
}
static inline state3 signed_to_state3(int32_t a){
	state3 q; memcpy(&q, &a, 4);
	return q;
}

static inline state3 float_to_state3(float a){
	K8_STATIC_ASSERT(sizeof(float) == 4);
	K8_STATIC_ASSERT(sizeof(state3) == 4);
	state3 q; 
	memcpy(&q, &a, 4);
	return q;
}
static inline float float_from_state3(state3 a){
	K8_STATIC_ASSERT(sizeof(float) == 4);
	K8_STATIC_ASSERT(sizeof(state3) == 4);
	float q;
	memcpy(&q, &a, 4);
	return q;
}
K8_COMPLETE_ARITHMETIC(2,3, 16)

//Fast Inverse Square Root.
static inline void k_fisr(state3 *xx){
	const int32_t x = signed_from_state3(*xx);
	int32_t i; 
	float x2;
	memcpy(&i, xx->state, 4);
	i = 0x5F1FFFF9 - (i>>1);
	memcpy(&x2, &i, 4);
	x2 *= 0.703952253f * (2.38924456f - x * x2 * x2);
	memcpy(xx->state, &x2, 4);
}

KNLB(4,8);
KNLCONV(3,4);
//The to and from functions can't be used unless we have uint64_t
#ifdef UINT64_MAX
static inline state4 to_state4(uint64_t a){
	state4 q;
	memcpy(&q, &a, 8);
	return q;
}
static inline uint64_t from_state4(state4 a){
	uint64_t q;
	memcpy(&q, &a, 8);
	return q;
}

static inline state4 signed_to_state4(int64_t a){
	state4 q;
	memcpy(&q, &a, 8);
	return q;
}
static inline int64_t signed_from_state4(state4 a){
	int64_t q;
	memcpy(&q, &a, 8);
	return q;
}

static inline state4 double_to_state4(double a){
	K8_STATIC_ASSERT(sizeof(double) == 8);
	state4 q;
	memcpy(&q, &a, 8);
	return q;
}
static inline double double_from_state4(state4 a){
	K8_STATIC_ASSERT(sizeof(float) == 4);
	double q;
	memcpy(&q, &a, 8);
	return q;
}
#endif


K8_COMPLETE_ARITHMETIC(3,4, 32)
K8_COMPLETE_FLOATING_ARITHMETIC(3, 4, float)


//larger kernels.
//Enough for a vec4
KNLB(5,16);
KNLCONV(4,5);
#ifdef UINT64_MAX
K8_COMPLETE_ARITHMETIC(4,5, 64)
K8_COMPLETE_FLOATING_ARITHMETIC(4, 5, double)
#endif

#ifdef __FLT128_MANT_DIG__
typedef __float128 float128;
static inline float128 float128_from_state5(state5 a){
	K8_STATIC_ASSERT(sizeof(float128) == 16);
	float128 q;
	memcpy(&q, &a, 16);
	return q;
}
static inline state5 float128_to_state5(float128 a){
	K8_STATIC_ASSERT(sizeof(float128) == 16);
	state5 q;
	memcpy(&q, &a, 16);
	return q;
}
#endif

static inline void k_muladdmul_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fadd_s3(
		statemix3(
			kb_fmul_s3(c->state4s[0]).state3s[0],
			kb_fmul_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}
static inline void k_add3_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fadd_s3(
		statemix3(
			kb_fadd_s3(c->state4s[0]).state3s[0],
			kb_fadd_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}
static inline void k_sumv4(state5 *c){k_add3_v4(c);}
static inline void k_mul3_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fmul_s3(
		statemix3(
			kb_fmul_s3(c->state4s[0]).state3s[0],
			kb_fmul_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}
static inline void k_mulsubmul_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fsub_s3(
		statemix3(
			kb_fmul_s3(c->state4s[0]).state3s[0],
			kb_fmul_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}
static inline void k_sub3_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fsub_s3(
		statemix3(
			kb_fsub_s3(c->state4s[0]).state3s[0],
			kb_fsub_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}
static inline void k_divadddiv_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fadd_s3(
		statemix3(
			kb_fdiv_s3(c->state4s[0]).state3s[0],
			kb_fdiv_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}
static inline void k_divsubdiv_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fsub_s3(
		statemix3(
			kb_fdiv_s3(c->state4s[0]).state3s[0],
			kb_fdiv_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}
static inline void k_div3_v4(state5 *c){
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
	c->state3s[0] = kb_fdiv_s3(
		statemix3(
			kb_fdiv_s3(c->state4s[0]).state3s[0],
			kb_fdiv_s3(c->state4s[1]).state3s[1]
		)
	).state3s[0];
}


K8_CHAINP(k_fmul_s3_answer_lower, k_fmul_s3, k_swap4, 4)
K8_MULTIPLEX_HALVES_NP(k_addv2, k_fadd_s3, 3, 4, 5, 0)
K8_MULTIPLEX_HALVES_NP(k_subv2, k_fsub_s3, 3, 4, 5, 0)
K8_MULTIPLEX_HALVES_NP(k_dotv2, k_fmul_s3, 3, 4, 5, 0)
//notice the arguments
//3 means "this is an array of state3's", iterate over it.
//4 is the type twice as large as 3, room enough for the shared state
K8_RO_SHARED_STATE_PARTIAL_NP(k_scalev2, k_fmul_s3_answer_lower, 3, 4, 5, 0, 2, 2, 0)

K8_RO_SHARED_STATE_NP(k_scalev3_scale_in_first, k_fmul_s3_answer_lower, 3, 4, 5, 0)
K8_RO_SHARED_STATE_PARTIAL_NP(k_scalev3, k_fmul_s3_answer_lower, 3, 4, 5, 0, 3, 3, 0)

//Variant where the scale is in the last element.
static inline void k_scalev3_scale_in_last(state5 *c){k_scalev3(c);}
//K8_SHARED_STATE(k_sumv4, k_fadd_s3, 3, 4, 5, 0)
//K8_MULTIPLEX_SIMD(name, func, nn, nm, iscopy)
K8_MULTIPLEX_SIMD(k_sqrv4, k_fsqr_s3, 3, 5, 0)
K8_CHAINP(k_sqrlengthv4, k_sqrv4, k_sumv4, 5)
static inline void k_lengthv4(state5 *c){
	k_sqrlengthv4(c);
	k_fsqrt_s3(c->state3s);
}
static inline void k_normalizev4(state5 *c){
	state3 length;
	{state5 tempc = *c;
			k_lengthv4(&tempc);
			length = tempc.state3s[0];
	}
	//for(ssize_t i = 0; i<4; i++)
	FORWARD_PTRAVERSAL(c, 3, 5, i, 0, 4, 1)
	{
		state4 worker;
		//worker.state3s[0] = c->state3s[i];
		worker.state3s[0] = *elem_i;
		worker.state3s[1] = length;
		k_fdiv_s3(&worker);
		c->state3s[i] = worker.state3s[0];
	}
	TRAVERSAL_END
}
#if K8_FAST_FLOAT_MATH
static inline void k_fisrnormalizev4(state5 *c){
	state3 length;
	{
		state5 tempc = *c;
			k_sqrlengthv4(&tempc);
			length = (tempc.state3s[0]);
			k_fisr(&length);
	}
	//for(ssize_t i = 0; i<4; i++)
	FORWARD_PTRAVERSAL(c, 3, 5, i, 0, 4, 1)
		*elem_i = float_to_state3(float_from_state3(*elem_i) * float_from_state3(length));
	TRAVERSAL_END
}
#endif
static inline void k_clampf(state5* c){
	const float a = float_from_state3(c->state3s[0]);
	const float min = float_from_state3(c->state3s[1]);
	const float max = float_from_state3(c->state3s[2]);
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state4s[1]);
#if K8_FAST_FLOAT_MATH == 0
	if(!isfinite(a) || !isfinite(min) || !isfinite(max)) return;
#endif
	if(a<min) {c->state3s[0] = float_to_state3(min); return;}
	if(a>max) {c->state3s[0] = float_to_state3(max); return;}
	return;
}
//Enough for a mat2x4 or 4x2
KNLB(6,32);
KNLCONV(5,6);

#ifdef __FLT128_MANT_DIG__
K8_COMPLETE_FLOATING_ARITHMETIC(5, 6, float128)
#endif

//newname, oldname, isarrayof, type taken by kernel, arraysize, start, end(exclusive), sharedind, iscopy
K8_RO_SHARED_STATE_PARTIAL_NP(k_scalev4, k_fmul_s3_answer_lower, 3, 4, 6,   0, 4,   4,    0)
static inline state6 kb_scalev4(state6 c){
	k_scalev4(&c);
	return c;
}
K8_MULTIPLEX_HALVES_NP(k_addv4, k_fadd_s3, 3, 4, 6, 0)
K8_MULTIPLEX_HALVES_NP(k_subv4, k_fsub_s3, 3, 4, 6, 0)
K8_MULTIPLEX_HALVES_NP(k_mulv4, k_fmul_s3, 3, 4, 6, 0)
K8_MULTIPLEX_HALVES_NP(k_divv4, k_fdiv_s3, 3, 4, 6, 0)

K8_MULTIPLEX_HALVES_PARTIAL_NP(k_addv3, k_fadd_s3, 3, 4, 6, 0,3, 0)
K8_MULTIPLEX_HALVES_PARTIAL_NP(k_subv3, k_fsub_s3, 3, 4, 6, 0,3, 0)
K8_MULTIPLEX_HALVES_PARTIAL_NP(k_mulv3, k_fmul_s3, 3, 4, 6, 0,3, 0)
K8_MULTIPLEX_HALVES_PARTIAL_NP(k_divv3, k_fdiv_s3, 3, 4, 6, 0,3, 0)
static inline void k_dotv4(state6 *c){
	k_mulv4(c);
	k_sumv4(c->state5s+0);
}
static inline state6 kb_dotv4(state6 c){
    k_dotv4(&c);
    return c;
}
//Enough for a 4x4. TODO implement SIMD-accelerated matrix math.
KNLB(7,32);
KNLCONV(6,7);
/*Limited memory version which works in-place.*/
static inline void k_mat4_transpose(state7 *c){
	K8_CONST(c->state3s[0]);
	K8_CONST(c->state5s[1].state3s[1]);
	K8_CONST(c->state5s[2].state3s[2]);
	K8_CONST(c->state5s[3].state3s[3]);
	//for(ssize_t row = 1; row < 4; row++)
	//for(ssize_t col = 0; col < row; col++)
	FORWARD_PTRAVERSAL(c, 3, 7, row, 1,   4, 1)
	FORWARD_PTRAVERSAL(c, 3, 7, col, 0, row, 1)
	{
		state3 temp;
		temp = c->state3s[row*4 + col];
		c->state3s[row*4 + col] = c->state3s[col*4 + row];
		c->state3s[col*4 + row] = temp;
	}
	TRAVERSAL_END
	TRAVERSAL_END
}


static inline void k_mat4_det(state7 *c){
	K8_CONST(c->state6s[1]);
	K8_CONST(c->state5s[1]);
	K8_CONST(c->state4s[1]);
	//0 is used.
	K8_CONST(c->state3s[1]);
	K8_CONST(c->state3s[2]);
	K8_CONST(c->state3s[3]);
	const state3 a00 = (c->state3s[0]), 	a01 = (c->state3s[1]), 	a02 = (c->state3s[2]), 	a03 = (c->state3s[3]),
			a10 = (c->state3s[4]), 	a11 = (c->state3s[5]), 	a12 = (c->state3s[6]), 	a13 = (c->state3s[7]),
			a20 = (c->state3s[8]), 	a21 = (c->state3s[9]), 	a22 = (c->state3s[10]), a23 = (c->state3s[11]),
			a30 = (c->state3s[12]), a31 = (c->state3s[13]), a32 = (c->state3s[14]), a33 = (c->state3s[15]);
	state5 worker;

	/*dest00 = a00 * a11 - 
			   a01 * a10,*/
			worker.state3s[0] = a00; 	worker.state3s[1] = a11;
			worker.state3s[2] = a01; 	worker.state3s[3] = a10;
		k_mulsubmul_v4(&worker);
	state3 dest00 = worker.state3s[0];
		    /*dest01 = a00 * a12 - 
		    		   a02 * a10,*/
		worker.state3s[0] = a00; 	worker.state3s[1] = a12;
		worker.state3s[2] = a02; 	worker.state3s[3] = a10;
			k_mulsubmul_v4(&worker);
	state3 dest01 = worker.state3s[0];
		    /*dest02 = a00 * a13 -
		    		 	a03 * a10,*/
		worker.state3s[0] = a00; 	worker.state3s[1] = a13;
		worker.state3s[2] = a03; 	worker.state3s[3] = a10;
			k_mulsubmul_v4(&worker);
	state3 dest02 = worker.state3s[0];
		    /*dest03 = a01 * a12 -
		    		 a02 * a11,*/
		worker.state3s[0] = a01; 	worker.state3s[1] = a12;
		worker.state3s[2] = a02; 	worker.state3s[3] = a11;
			k_mulsubmul_v4(&worker);
	state3 dest03 = worker.state3s[0];


		    /*dest04 = a01 * a13 -
		    		 a03 * a11,*/
		worker.state3s[0] = a01; 	worker.state3s[1] = a13;
		worker.state3s[2] = a03; 	worker.state3s[3] = a11;
			k_mulsubmul_v4(&worker);
	state3 dest04 = worker.state3s[0];
		    /*dest05 = a02 * a13 -
		    		 	a03 * a12,*/
	worker.state3s[0] = a02; 	worker.state3s[1] = a13;
	worker.state3s[2] = a03; 	worker.state3s[3] = a12;
		k_mulsubmul_v4(&worker);
	state3 dest05 = worker.state3s[0];
	/*
		dest06 = a20 * a31 -
				 a21 * a30,
	*/

		worker.state3s[0] = a20; 	worker.state3s[1] = a31;
		worker.state3s[2] = a21; 	worker.state3s[3] = a30;
			k_mulsubmul_v4(&worker);
	state3 dest06 = worker.state3s[0];
/*
		    dest07 = a20 * a32 -
		    		 a22 * a30,
*/

		worker.state3s[0] = a20; 	worker.state3s[1] = a32;
		worker.state3s[2] = a22; 	worker.state3s[3] = a30;
			k_mulsubmul_v4(&worker);
	state3 dest07 = worker.state3s[0];
/*
		    dest08 = a20 * a33 -
		    		 a23 * a30,

*/
		worker.state3s[0] = a20; 	worker.state3s[1] = a33;
		worker.state3s[2] = a23; 	worker.state3s[3] = a30;
			k_mulsubmul_v4(&worker);
	state3 dest08 = worker.state3s[0];

		    /*dest09 = a21 * a32 -
		    		 a22 * a31,*/
		worker.state3s[0] = a21; 	worker.state3s[1] = a32;
		worker.state3s[2] = a22; 	worker.state3s[3] = a31;
			k_mulsubmul_v4(&worker);
	state3 dest09 = worker.state3s[0];
		   /*
		    dest10 = a21 * a33 -
		    		 a23 * a31,
		   */
		worker.state3s[0] = a21; 	worker.state3s[1] = a33;
		worker.state3s[2] = a23; 	worker.state3s[3] = a31;
			k_mulsubmul_v4(&worker);
	state3 dest10 = worker.state3s[0];
	/*
		    dest11 = a22 * a33 -
		    		 a23 * a32;
	*/
		worker.state3s[0] = a22; 	worker.state3s[1] = a33;
		worker.state3s[2] = a23; 	worker.state3s[3] = a32;
			k_mulsubmul_v4(&worker);
	state3 dest11 = worker.state3s[0];

//as it so happens, worker.state3 is already
//set to dest11.
//the following line will therefore do nothing.
//TERM 1
	worker.state3s[0] = dest11;
	worker.state3s[1] = dest00;
	k_fmul_s3(worker.state4s);
	//store our result... this is the first term, we aren't
	//adding or subtracting yet
	worker.state4s[1].state3s[0] = worker.state3s[0];
//TERM 2
	worker.state3s[0] = dest01;
	worker.state3s[1] = dest10;
	k_fmul_s3(worker.state4s);
	//subtract off our result (this is a subtracting term.)
	worker.state4s[1].state3s[1] = worker.state3s[0];
	k_fsub_s3(worker.state4s+1);
//TERM 3
	worker.state3s[0] = dest02;
	worker.state3s[1] = dest09;
	k_fmul_s3(worker.state4s);
	//add our result (this is an adding term.)
	worker.state4s[1].state3s[1] = worker.state3s[0];
	k_fadd_s3(worker.state4s+1);
//TERM 4
	worker.state3s[0] = dest03;
	worker.state3s[1] = dest08;
	k_fmul_s3(worker.state4s);
	//add our result (this is an adding term.)
	worker.state4s[1].state3s[1] = worker.state3s[0];
	k_fadd_s3(worker.state4s+1);
//TERM 5
	worker.state3s[0] = dest04;
	worker.state3s[1] = dest07;
	k_fmul_s3(worker.state4s);
	//sub our result
	worker.state4s[1].state3s[1] = worker.state3s[0];
	k_fsub_s3(worker.state4s+1);
//TERM 6
	worker.state3s[0] = dest05;
	worker.state3s[1] = dest06;
	k_fmul_s3(worker.state4s);
	//add our result (this is an adding term.)
	worker.state4s[1].state3s[1] = worker.state3s[0];
	k_fadd_s3(worker.state4s+1);
	c->state3s[0] = worker.state4s[1].state3s[0];
}

static inline void k_mat4_det_old(state7 *c){
	float a00 = float_from_state3(c->state3s[0]),
			a01 = float_from_state3(c->state3s[1]),
			a02 = float_from_state3(c->state3s[2]), 
			a03 = float_from_state3(c->state3s[3]),
			a10 = float_from_state3(c->state3s[4]), 
			a11 = float_from_state3(c->state3s[5]), 
			a12 = float_from_state3(c->state3s[6]), 
			a13 = float_from_state3(c->state3s[7]),
			a20 = float_from_state3(c->state3s[8]), 
			a21 = float_from_state3(c->state3s[9]), 
			a22 = float_from_state3(c->state3s[10]), 
			a23 = float_from_state3(c->state3s[11]),
			a30 = float_from_state3(c->state3s[12]), 
			a31 = float_from_state3(c->state3s[13]), 
			a32 = float_from_state3(c->state3s[14]), 
			a33 = float_from_state3(c->state3s[15]);
	float dest00 = a00 * a11 - a01 * a10,
		    dest01 = a00 * a12 - a02 * a10,
		    dest02 = a00 * a13 - a03 * a10,
		    dest03 = a01 * a12 - a02 * a11,
		    dest04 = a01 * a13 - a03 * a11,
		    dest05 = a02 * a13 - a03 * a12,
		    dest06 = a20 * a31 - a21 * a30,
		    dest07 = a20 * a32 - a22 * a30,
		    dest08 = a20 * a33 - a23 * a30,
		    dest09 = a21 * a32 - a22 * a31,
		    dest10 = a21 * a33 - a23 * a31,
		    dest11 = a22 * a33 - a23 * a32;
    c->state3s[0] = float_to_state3(
    	dest00 * dest11 - dest01 * dest10 +
    	dest02 * dest09 +
    	dest03 * dest08 - dest04 * dest07 +
    	dest05 * dest06
    );
}
//Enough for TWO 4x4s
KNLB(8,32);
KNLCONV(7,8);

K8_MULTIPLEX_HALVES_NP(k_addmat4, k_fadd_s3, 3, 4, 8, 0)
K8_MULTIPLEX_HALVES_NP(k_submat4, k_fsub_s3, 3, 4, 8, 0)
K8_MULTIPLEX_HALVES_NP(k_mulv16, k_fmul_s3, 3, 4, 8, 0)
K8_MULTIPLEX_HALVES_NP(k_divv16, k_fmul_s3, 3, 4, 8, 0)

static inline void k_mul_mat4(state8 *c){
	/*Matrix multiplication for dummies.
						col
		A 			B   v 			C
		1 0 0 0| 	1 0 0 0|	=	X X X X| 
	row>0 1 0 0| 	0 1 0 0| 	=	X X T X|
		0 0 1 0| 	0 0 1 0| 	=	X X X X|
		0 0 0 1| 	0 0 0 1|	=	X X X X|
		where T = dotv4(row, col)

		These matrices are COLUMN MAJOR, which means state7.state3s looks like this:
		0 4 8 12
		1 5 9 13
		2 6 1014
		3 7 1115
	*/
//	PRAGMA_SIMD //NO, we don't want it.

	
	state7 A = c->state7s[0];
	//state7 B = c->state7s[1];
	K8_CONST(c->state7s[1]);
	//for(ssize_t col = 0; col < 4; col++)
	FORWARD_TRAVERSAL(c->state7s[0], 5, 7, col, 0,   4,  1)
	{
		state8 pairs; 
		//Prepare the pairs to be dotted together.
		//B portions.
		//for(ssize_t row = 0; row < 4; row++)
		FORWARD_TRAVERSAL(pairs, 6, 8, row, 0,   4,  1)
			//pairs.state6s[row].state5s[1] = c->state7s[1].state5s[col];
			elem_row->state5s[1] = c->state7s[1].state5s[col];
		TRAVERSAL_END
		//A portions
		//for(ssize_t row = 0; row < 4; row++)
		//for(ssize_t i = 0; i < 4; i++)
		FORWARD_TRAVERSAL(pairs, 6, 8, row, 0,   4,  1)
		FORWARD_TRAVERSAL(A, 5, 7, i, 0,   4,  1)
/*pairs.state6s[row].*/elem_row->state5s[0].state3s[i] = 
	/*A.state5s[i].*/				elem_i->state3s[row];
		TRAVERSAL_END
		TRAVERSAL_END
		//Perform our dot products.
		//for(ssize_t row = 0; row < 4; row++)
		FORWARD_TRAVERSAL(pairs, 6, 8, row, 0,   4,  1)
			k_dotv4(elem_row);
		TRAVERSAL_END
		//for(ssize_t row = 0; row < 4; row++)
		FORWARD_TRAVERSAL(pairs, 6, 8, row, 0,   4,  1)
			//c->state7s[0].state5s[col].state3s[row] = pairs.state6s[row].state3s[0];
			//c->state7s[0].state5s[col].state3s[row] = elem_row->state3s[0];
			elem_col->state3s[row] = elem_row->state3s[0];
		TRAVERSAL_END
	}
	TRAVERSAL_END
}


static inline void k_mat4xvec4(state8 *c){
	const state7 mat = c->state7s[0];
	const state5 vec = c->state7s[1].state5s[0];
	K8_UNUSED(c->state7s[1].state5s[1]);
	K8_UNUSED(c->state7s[1].state5s[2]);
	K8_UNUSED(c->state7s[1].state5s[3]);
	//for(ssize_t row = 0; row < 4; row++)
	FORWARD_TRAVERSAL(c->state5s[0], 3, 5, row, 0,   4,  1)
	{
		state6 ret;
		//for(ssize_t b = 0; b < 4; b++)
		FORWARD_TRAVERSAL(ret.state5s[0], 3, 5, b, 0,   4,  1)
			*elem_b = mat.state3s[row + 4*b];
		TRAVERSAL_END
		ret.state5s[1] = vec;
		k_dotv4(&ret);
		c->state5s[0].state3s[row] = ret.state3s[0];
	}
	TRAVERSAL_END
}
KNLB(9,32);
KNLCONV(8,9);
KNLB(10,32);
KNLCONV(9,10);

//The eleventh order kernel holds 2^(11-1) bytes, or 1024 bytes.
KNLB(11,32);
KNLCONV(10,11);
KNLB(12,32);
KNLCONV(11,12);
KNLB(13,32);
KNLCONV(12,13);
KNLB(14,32);
KNLCONV(13,14);
KNLB(15,32);
KNLCONV(14,15);
KNLB(16,32);
KNLCONV(15,16);
KNLB(17,32);
KNLCONV(16,17);
KNLB(18,32);
KNLCONV(17,18);
KNLB(19,32);
KNLCONV(18,19);
KNLB(20,32);
KNLCONV(19,20);
//Holds an entire megabyte. 2^(21-1) bytes, 2^20 bytes, 2^10 * 2^10, 1024 * 1024 bytes.
KNLB(21,32);
KNLCONV(20,21);
//TWO ENTIRE MEGABYTES
KNLB(22,32);
KNLCONV(21,22);
//FOUR ENTIRE MEGABYTES
KNLB(23,32);
KNLCONV(22,23);
//EIGHT ENTIRE MEGABYTES. As much as the Dreamcast.
KNLB(24,32);
KNLCONV(23,24);
//16 megs
KNLB(25,32);
KNLCONV(24,25);
//32 megs
KNLB(26,32);
KNLCONV(25,26);
//64 megs
KNLB(27,32);
KNLCONV(26,27);
//128 megs
KNLB(28,32);
KNLCONV(27,28);
//256 megs.
KNLB(29,32);
KNLCONV(28,29);
//512 megs.
KNLB(30,32);
KNLCONV(29,30);
//1G
KNLB(31,32);
KNLCONV(30,31);
//2G
KNLB(32,32);
KNLCONV(31,32);
//4G
KNLB(33,32);
KNLCONV(32,33);
//8G
KNLB(34,32);
KNLCONV(33,34);
//16G
KNLB(35,32);
KNLCONV(34,35);
//math typedefs

typedef state5 k_vec4;
typedef state5 k_vec3;
typedef state4 k_vec2;
typedef state7 k_mat4;


#endif
