Kernel8 Functional, Low-level, Intermediate Representation Language


High performance functional programming and intermediate representation for higher level code.

Research project by DMHSW.




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

* Kernel code written entirely from kernels which have no erroneous inputs always has well-defined behavior.

* Kernel code is extremely memory efficient

* Kernel code is extremely easy to parallelize
	(What can and what cannot be parallelized is extremely clear when written in kernel code)

### What use cases are there for Kernel8?

That's what i'm researching. Possibilities include...

* Hardware description/prototyping language.

* Functional Metaprogramming language

* Compiler IR

* SIMD instruction implementations (Extremely easy to write code which forces the compiler to generate SIMD even at O2)

### What has been written?

* Complete Kernels (Fully defined for their entire domain) for all common floating point and integer operations.

* VLINT (Very Large Integer) addition, subtraction, shift left and shift right.

* A myriad of utility functions

* lots of (non-explicit) SIMD accelerated code including functions which extremely reliably compile into optimal SIMD code.

I've implemented bits and pieces of a computer graphics math library.

I've been testing how Kernel8 code compiles with GCC and clang into x86_64 assembly language using Matt Godbolt's compiler explorer.

So far I am decently pleased with the results, however there are known issues.

1) Copy kernels sporadically do either decently well, or incredibly poorly in the optimizer.

2) The state union typedef for Kernel8, which is the basis for the entire language, bogs down compiletimes.

3) Implementing the exact reference code for an assembly language instruction has variable success rates.

I was able to successfully replicate vfnmadd321s and a couple others, but I cannot get the compiler
to generate a kernel which just does pshufbs... but the compiler *will* generate pshufbs in other code...

4) Clang has more consistent, but overall worse results than GCC

If clang can optimize something, then almost any minor variation of it will compile to the exact same code.

But with gcc, the optimization is very sporadic, it's either incredibly good (50%) same as clang(40%) or it shits the bed(10%)

GCC's optimization is sometimes so good that it generates the same 
SIMD accelerated instructions at O1 as it does at O3. This never happens with clang.

if you want a specific example of a kernel which compiles extremely inconsistently, check out mat4xvec4 and mul_mat4

5) static inline is necessary to achieve optimal results. Everywhere.

6) Quite confusingly, #pragma omp simd will sometimes *disable* SIMD acceleration and even prevent unrolling loops
where it would otherwise be present.

7) GCC inconsistently statically evaluates code. Sometimes it fails to properly perform constant propagation. 

Clang almost always succeeds in performing constant propagation.

8) Clang shits the bed at integer operations

9) in GCC especially, loops flatten differently when loop iterators are signed or unsigned. Signed produces better code.

10) Neither GCC nor clang will inline code called through function pointers using a dispatch table in a for loop
indexed with the iterator, but they will BOTH put in *calls* to those functions, without the iterator.

This means of course that KERNEL_MULTIPLEX_MULTIKERNEL is very poorly optimized by both compilers.

### Why can't a kernel take in more than one argument?

It's the ABI i've chosen.

### API tips

1) Use k_at, k_pat, k_off, and k_offp for non-constant accesses into state.

If/when I implement my own Kernel8 compiler, I will implement the proper wrapping behavior.

### Programming language specification not implemented or unable to be implemented due to restrictions

The API is still very unstable.

I'm still defining the specification and have as-of-yet to write a standard.

1) KERNEL_UNUSED(x) 

should indicate to the compiler that a portion of the state (Either pointer or value passed as X) goes unused

for a scope (inside of curly braces)

2) KERNEL_CONST(x)

should explicitly indicate to the compiler that a portion of the state goes unmodified (Read-only) in a scope.

3) Memory mirroring

It should be impossible to index a state out-of-bounds by virtue of index mirroring.

This also makes implementing some algorithms easier.

you can use k_at, k_pat, k_off, and k_offp to simulate this functionality.

"p" variants take in a pointer to a state, "at" provides an lvalue and "off" provides a pointer.

4) Block memory allocation

A C++ expert friend of mine is working with me on this one, we might be able to figure something out.

I'd like the language to be able to use block allocation implicitly for accesses into large states.

Currently, if you want a state31, you *have* to be able to malloc a gigabyte of (virtually) contiguous memory.

This works for modern processors which have virtual addressing hardware, but

it makes Kernel8 very restrictive in freestanding implementations with out virtual memory mapping.



5) Haskell-like "IO" and optimization restrictions based on IO-propagation

Currently, the convention for kernels which use some form of external state is to name them "fk_" (for "fake")

I'd like a way of distinguishing fake kernels from real ones at compiletime and warning/erroring based on this.

6) Stack restriction

Currently, Kernel8 does not place any limit on the size of the stack, it can be as large
as your system will allow.

This may be desirable for most cases, but I'd like to be able to control stack allocation limits in the language.

6) Explicit lazy memory allocation

If a block of memory is not used, it should not be allocated.

This is in fact what happens in my tests (My operating system is awfully smart!) with large states in the
data section (And even mallocs, too!) but I don't like depending on this functionality.

7) Arbitrary state implementation

This is another one of those features that could probably be implemented in C++.

Kernel8 should not force you to use contiguous memory or even real memory at all- you should be able to

use disk, compressed RAM, flash, or networked addressable locations (Anything that can be indexed)

8) Forced constant propagation

Currently, i'm using static_assert to try to force constant propagation- it's why this is a C11 codebase and not C99.

9) Forced compiletime memory allocation reduction

GCC and Clang are both smart enough not to allocate memory that Kernel8 does not need even though it declares it,

but I don't like hinging on this.

10) Formal verification of a kernel's "completeness" as a compiletime action for optimization

The gcc "pure" attribute of functions can be used to force this functionality, but there are no tests,
and it only works on copy kernels (pass by pointer does not work)

I want to be able to prove a kernel's completeness by the propagation of its subcomponents,

if you build your kernel out of totally compliant code, it *is* complete.

11) Disabling of C language features inside of Kernel8 code

Specifically:
* Conversion of pointers to integers and storing them in non-pointer memory locations.
* Pointer arithmetic (Other than indexing)
* Arithmetic exceptions (Division by zero, non-quiet nans)
* signed int undefined behavior (Mitigated by kernel8's math code)

12) Lazy evaluation

If a memory location which was, for instance, the location where another kernel wrote something, has not been
accessed, there is no reason to put the data there in the first place until it is needed.

Note that both GCC and clang are smart enough in many cases to do this and change the order of function calls at compile time.
but it is inconsistent and it should be an explicit language feature.

This would allow you to, for instance, write the first 5 million primes to an area of state.

Then another portion of your program could access them as-needed and they wouldn't actually be computed until they were read.
