
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//Kernel definition functions.
//use debugging functionality.
#define KERNEL_DEBUG
//Kernel basic definitions.
#include "kerneln.h"



//The first Kernel8 code ever written.
void and127(state1 *c){ //Reduce to 7 bits of state
	c->state[0] &= 127;
}
void and63(state1 *c){ //Reduce to 6 bits of state
	c->state[0] &= 63;
}

//Kernel32- return c if prime else 0
void is_prime(state3* c){
	uint32_t val = from_state3(*c);
	if(val == 2 || val == 3) return;
	if(val == 1) 
		{*c = state3_zero(); return;}
	if(val%2 == 0) 
		{*c = state3_zero(); return;}
	for(uint32_t i = 3; i < val/2; i+=2){
		if(val%i == 0) {*c = state3_zero(); return;}
	}
	return;
}

void k_incrementhalves4(state4 *c){
	c->state3s[0] = to_state3(from_state3(state_high4(*c))+1);
	c->state3s[1] = to_state3(from_state3(state_low4(*c))+1);
}

void k_upper3_4_increment(state4 *c){
	c->state3s[1] = to_state3(from_state3(c->state3s[0])+1);
	c->state3s[0] = state3_zero();
}

//High state3 is the index, Low state3 is the data at that index.
void k_fillerind(state4 *c){ //Real kernel using the "MultiplexIndexed" syntax.
	uint32_t index = from_state3(state_high4(*c));
	c->state3s[0] = state3_zero();
	c->state3s[1] = to_state3(index);
}
void k_mul5(state3 *c){
	*c = to_state3(from_state3(*c)*5);
}
void k_modsort(state4 *c){ //Use the value at the index to choose its placement.
	c->state3s[0] = c->state3s[1];
}

//Real variant of fk_printer! Using the "indexed" multiplex syntax.
void fk_printerind(state4 *c){
	printf("%u, %u\n", from_state3(c->state3s[0]), from_state3(c->state3s[1]));
}

void fk_printer(state3 *c){
	printf("As uint: %u\n", from_state3(*c));
	printf("Byte 0: %u\n", from_state1(c->state1s[0]));
	printf("Byte 1: %u\n", from_state1(c->state1s[1]));
	printf("Byte 2: %u\n", from_state1(c->state1s[2]));
	printf("Byte 3: %u\n", from_state1(c->state1s[3]));
}
//Print individual bytes, with an 8 bit index.
void fk_printer8ind(state2 *c){
	printf("BP! %u, %u\n", from_state1(state_high2(*c)), from_state1(state_low2(*c)));
}
//Print individual bytes, with a 32 bit index.
void fk_printer8ind32(state4 *c){
	uint32_t ind = from_state3(c->state3s[0])<<2; //We recieved four bytes of data!
	state3 dataseg = c->state3s[1];
	KERNEL_FORWARD_TRAVERSAL(dataseg, 	1, //state you are extracting
										3, //state you are traversing.
										i, //for loop variable
										0, //start
										4, //for i = start, i < end
										1) //increment, i+= 1.
		printf("BP32! %u, %u\n", ind + (uint32_t)i, from_state1(*elem_i));
	KERNEL_TRAVERSAL_END
}

//A large shared state
static inline void big_shared_index(state5* c){
	//we are iterating over state4's
	//we have left a nice gap in the shared portion at the beginning to hold our index.
	//the index is stored in the second state3 of the first state4.
	uint32_t index = c->state4s[0].state3s[1].u;
	//This is actually
	index--;
	index *= 2;
	//if this is the first iteration...
	printf("EXECUTING, INDEX=%u\n", index);
	if(index == 0){
		//puts("First Iteration Detected. This should print exactly once.");
		//write zero to shared variable.
		c->state4s[0].state3s[0].u = 0;
	}
	if(index >= 10 && index < 15000){
		//process our first state3
		c->state4s[1].state3s[0].u = index*100;
		//Our other state3.
		c->state4s[1].state3s[1].u = (index+1)*100;
		//Increment the shared integer.
		c->state4s[0].state3s[0].u+=2;
	}
}
//the new function will be called "big_shared_process"
//it's iterating over an array of state4's, and takes in a state5 (upper is shared.)
//the array to process is a state20.
//We're using a state3 as our index (32 bit unsigned index...)
//We want state3s[1] in the shared portion to be our index,
//the uppermost state3 is a shared variable that we use while processing the array.
//If you're worried about what happens to the data at the location where the index is stored,
//don't worry, it is saved
//it is restored at the end of processing.
//																Arrayof	What we take|  What?|staten of index|where|write index?|iscopy
KERNEL_SHARED_STATE_WIND(big_shared_process20, //new name
							big_shared_index,   //kernel to multiplex
							4, //What is this large state an array of?
							5, //What stateX do we take? (Must be previous argument + 1)
							20,//What is the array?
							3,//What stateX is our index?
							1,//Where in the shared state should the index be written. (NOTE: the value at this location is restored post-call)
							1,//Do we want to enable writing the index? (0 means same as SHARED_STATE)
							0) //is our old kernel a copy-kernel?
//Summer.
void k_sum32(state4 *c){
	uint32_t high = from_state3(state_high4(*c));
	uint32_t low = from_state3(state_low4(*c));
	c->state3s[0] = to_state3(high+low);
	c->state3s[1] = state3_zero();
}

void k_dupe_upper4(state4 *c){
	*state_pointer_low4(c) = *state_pointer_high4(c);
}

void k_ifunc(state3 *c){ //A real kernel.
	//Performs rot right of 4.
	//return to_state3( (from_state3(c)>>4) | (from_state3(c)<< (32 - 4 )) );
	*c = to_state3( from_state3(*c) / 3 );
}
void k_add1_3(state3 *c){
	state4 a;
	a.state3s[0] = *c;
	a.state3s[1] = to_state3(1);
	k_add_s3(&a);
	*c = a.state3s[0];
}

//Kernel8 is designed to use the maximum available parallelism by default,
//if you simply use KERNEL_MULTIPLEX then you will get "medium-high" multithreaded code.
/*Here is a list of levels of parallelism in Kernel8

unspecified- same as PARALLEL if possible, else, NOPARALLEL
PARALLEL- uses medium-high level of multithreading. Equates to normal multithreading on a CPU.
SUPARA- Super parallelism. If GPU parallelism or some form of extreme parallelism is available, use it.
SIMD- use low level parallelism/multithreading. Note that despite the name, GCC will often
optimize NOPARALLEL code better. Watch out for that.
NP- shorthand for NOPARALLEL
NOPARALLEL- Do not explicitly use parallelism; the compiler may still generate SIMD instructions.

if an algorithm cannot be multithreaded or parallelized then it has no parallelism specification,
you muse use the unspecified API and it will be non-parallel (although the compiler may still in fact
generate SIMD instructions by performing wizardry on your code)

if you are writing "fake" kernels which use the kernel8 ABI but do not obey the rules of Kernel8,
then you should always os NOPARALLEL to prevent race conditions.

if you want to, you can use the ALIAS versions of functions which are totally generic.

the more specific versions are generated from them.
'name' - name of new kernel
'func' - kernel you are multiplexing.
'alias' - mode of parallelism. either PARALLEL, SUPARA, SIMD, or NOPARALLEL
'nn' - stateX that the container is considered to be an array of.
'nnn' - must be one greater than nn
'nm' - stateX of the array.
'iscopy' - is func a copy-kernel? 0 means it passes by pointer, 1 means pass by value.

KERNEL_MULTIPLEX_PARTIAL_ALIAS(name, func, nn, nm, start, end, iscopy, alias)
KERNEL_MULTIPLEX_INDEXED_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, alias)
KERNEL_SHUFFLE_IND32_PARTIAL(name, func, nn, nm, start, end, iscopy)
KERNEL_SHUFFLE_IND16_PARTIAL(name, func, nn, nm, start, end, iscopy)
KERNEL_SHUFFLE_IND8_PARTIAL(name, func, nn, nm, start, end, iscopy)
KERNEL_MULTIPLEX_INDEXED_EMPLACE_PARTIAL(name, func, nn, nnn, nm, start, end, iscopy)
//sharedind- index of the shared attribute. Most of the SHARED_STATE declarations default to zero.
//start- the starting iteration of the loop
//end- one greater than the last iteration.
//nwind- stateX of the index, (You might only need a 1 byte index, after all!)
//whereind- where in the shared state to write the index.
//doind- should we even write the index?
KERNEL_SHARED_STATE_PARTIAL_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy)
KERNEL_RO_SHARED_STATE_PARTIAL_ALIAS_WIND(name, func, nn, nnn, nm, start, end, sharedind, nwind, whereind, doind, iscopy, alias)
KERNEL_MULTIPLEX_HALVES_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, alias)
KERNEL_MULTIPLEX_MULTIKERNEL_PARTIAL_ALIAS(name, funcarr, nn, nm, start, end, iscopy, alias)
//Nlogn functionality, an "i,j" nested loop
//i = start; i < end - 1; i++
//j = i+1; j < end; j++
KERNEL_MULTIPLEX_NLOGN_PARTIAL(name, func, nn, nnn, nm, start, end, iscopy)
KERNEL_MULTIPLEX_NLOGNRO_PARTIAL_ALIAS(name, func, nn, nnn, nm, start, end, iscopy, alias)
//just like an ordinary multiplex but with an arbitrary number of bytes retrieved "nproc"
//rather than the array being treated as an array of statenn's
KERNEL_MULTIPLEX_DATA_EXTRACTION_PARTIAL_ALIAS(name, func, nproc, nn, nm, start, end, iscopy, alias)
*/
//Generate a multiplexing of and127 from state1 to state3.
//Notice the SIMD parallelism hint,
KERNEL_MULTIPLEX_SIMD(and127_mt3, //New kernel's name.
						 and127, //What kernel are we multiplexing?
						 1, //What is the large state to be interpreted as an array of?
						 3, //Large stateX
						 0) //is this a copy kernel?


//Endian conditional swaps for byte printing.
KERNEL_MULTIPLEX_SIMD(k_endian_cond_swap3_simd_mtp20, k_endian_cond_byteswap3, 3, 20, 0)

//Multiply unsigned integers by 5.
KERNEL_MULTIPLEX_SIMD(k_mul5_simd_mtp20, k_mul5, 3, 20, 0)

//This one uses the maximum available parallelism on the system
KERNEL_MULTIPLEX_SUPARA(k_mul5_mtp20, k_mul5, 3, 20, 0)

//Multiplex is_prime by pointer to state20.
KERNEL_MULTIPLEX(is_prime_mtp20, is_prime, 3, 20, 0)
//multiplex our divide by 7 function from state3 to state30
KERNEL_MULTIPLEX(k_ifunc_mtp30, k_ifunc, 3, 30, 0)



/*
Kernels to fill large states with indices into them.
Norice the arguments
Newkernel name,
Oldkernel name, 
stateX which the large state is to be treated as an array of.
stateX which your old kernel operates on
stateX which is your "large state" to operate on.
iscopy- is your old kernel a pass-by-copy kernel (0 means it passes by pointer, 1 is pass by value)

These can be parallelized.
*/
KERNEL_MULTIPLEX_INDEXED(k_fillerind_mtpi20, k_fillerind, 3, 4, 20, 0);
KERNEL_MULTIPLEX_INDEXED(k_fillerind_mtpi30, k_fillerind, 3, 4, 30, 0);
//Nonparallel MULTIPLEX_INDEXED.
KERNEL_MULTIPLEX_INDEXED_NP(fk_printerind_np_mtpi20, fk_printerind, 3, 4, 20, 0);
KERNEL_MULTIPLEX_INDEXED_NP(fk_printerind_np_mtpi30, fk_printerind, 3, 4, 30, 0);
//Print as uint8_t's with an 8 bit index.
KERNEL_MULTIPLEX_INDEXED_NP(fk_printer8ind_np_mtpi3, fk_printer8ind, 1, 2, 3, 0);
KERNEL_MULTIPLEX_INDEXED_NP(fk_printer8ind_np_mtpi20, fk_printer8ind, 1, 2, 20, 0);
KERNEL_MULTIPLEX_INDEXED_NP(fk_printer8ind_np_mtpi30, fk_printer8ind, 1, 2, 30, 0);
//Print as uint8_t's with 32 bit index.
KERNEL_MULTIPLEX_INDEXED_NP(fk_printer8ind32_np_mtpi20, fk_printer8ind32, 3, 4, 20, 0);
KERNEL_MULTIPLEX_INDEXED_NP(fk_printer8ind32_np_mtpi30, fk_printer8ind32, 3, 4, 30, 0);
//Emplacing modsort
//mtie stands for "multiplex indexed emplace"
//mtpie stands for "multiplex pointer indexed emplace"
KERNEL_MULTIPLEX_INDEXED_EMPLACE(k_modsort_mtpie20, k_modsort, 3, 4, 20, 0);
//Shuffle.
KERNEL_SHUFFLE_IND32(k_shuffler1_3_20, k_add1_3, 3, 20, 0)
//Shared state worker.
//sharedp stands for "shared pointer". it is a pass-by-pointer shared-type algorithm kernel
KERNEL_SHARED_STATE(k_sum32_sharedp3_20, k_sum32, 3, 4, 20, 0)

//KERNEL_MULTIPLEX_HALVES(name, func, nn, nnn, nm, iscopy)
//Treat the halves of a state20 as two separate arrays.
//Retrieve a state3 from each,
//then feed them as the high and low portions 
KERNEL_MULTIPLEX_HALVES_SUPARA(k_sum32_halvesp20, k_sum32, 3, 4, 20, 0)
//This one uses a read-only shared state, so it can be parallelized.
//I have decided not to include this fact in the name.


KERNEL_RO_SHARED_STATE_SUPARA(k_dupe_upper4_sharedp3_20, k_dupe_upper4, 3, 4, 20, 0)


/*
Argument 1: 
*/
KERNEL_MULTIPLEX_NLOGN(k_incrementhalves4_nlognp20, k_incrementhalves4, 3, 4, 20, 0)


//Nlong
KERNEL_MULTIPLEX_NLOGNRO_SUPARA(k_upper3_4_increment_nlognrop20, k_upper3_4_increment, 3, 4, 20, 0)
//KERNEL_MULTIPLEX_NLOGNRO(k_upper3_4_increment_nlognro20, k_upper3_4_increment, 3, 4, 20, 1);
//KERNEL_MULTIPLEX_NLOGN(k_incrementhalves4_nlogn20, k_incrementhalves4, 3, 4, 20, 1);
//The real magic! We can use our previously generated kernels to
//create NEW kernels.
//This has *infinite possibilities*.
KERNEL_MULTIPLEX_SUPARA(k_dupe_upper4_sharedp3_20_mtp30,k_dupe_upper4_sharedp3_20, 20, 30,0)

//Extract arbitrary data for multiplexing
/*
Argument 1: name of new kernel
Argument 2: name of kernel to multiplex
Argument 3: NUMBER OF BYTES!!!! to retrieve 
Argument 4: stateX which the old kernel operates on.
Argument 5: stateX to operate on.
*/
KERNEL_MULTIPLEX_DATA_EXTRACTION_NP(fk_byteprinter_extract3, //new kernel name
									fk_printer, //old
									2, //NUMBER OF BYTES to extract per iteration.
									3, //stateX that your old kernel takes.
									20, //stateX to iterate over.
									0) //is old a copy kernel? 0 or 1.
//Wow this is really unsafe...
//KERNEL_MULTIPLEX_DATA_EXTRACTION_SUPARA(fk_byteprinter_extract3_supara, fk_printer, 2, 3, 20, 0)

static kernelpb1 and_7667_funcs[4] = {
	and127,
	and63,
	and63,
	and127
};
//Multikernel
KERNEL_MULTIPLEX_MULTIKERNEL_SUPARA(and_7667_big, and_7667_funcs, 1, 30, 0);
KERNEL_MULTIPLEX_MULTIKERNEL_NP(and_7667, and_7667_funcs, 1, 3, 0);
void and_7667_2(state3* c){
	and127(c->state1s);
	and63(c->state1s+1);
	and63(c->state1s+2);
	and127(c->state1s+3);
}

//512 megabyte array.
state30 hughmong; //HUGH MONGOUS

int main(int argc, char** argv){
	union {uint32_t u; float f; int32_t i;} a, b, c;
	srand(time(NULL));
	{
		state3 s; 
		
		a.u = 0xffaa21d8;
		a.u = rand();
		b.u = 0x00;
		s = to_state3(a.u);
		
		//and127_mtp3(&s);
		//s = k_endian_cond_swap3(s);
		and127_mt3(&s);
		c.u = from_state3(s);
		printf("<1>OP ON %x EQUALS %x\n", a.u, c.u);
		printf("Sizeof state10: %zu\n",sizeof(state10));
		fk_printer8ind_np_mtpi3(&s);


		printf("\nTesting backward traversal...\n");
		KERNEL_BACKWARD_TRAVERSAL(s, 1, 3, i, rand()%4, (rand()%4)-1, 1)
			state2 arg;
			arg.state1s[1] = *elem_i;
			arg.state1s[0] = to_state1(i);
			fk_printer8ind(&arg);
		KERNEL_TRAVERSAL_END

		printf("\nTesting forward traversal...\n");
		KERNEL_FORWARD_TRAVERSAL(s, 1, 3, i, rand()%4, 1 + rand()%4, 1)
			state2 arg;
			arg.state1s[1] = *elem_i;
			arg.state1s[0] = to_state1(i);
			fk_printer8ind(&arg);
		KERNEL_TRAVERSAL_END
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		//Another test.
		a.u = rand();
		a.u = 0xffffffff;
		s=to_state3(a.u);
		and_7667(&s);
		c.u = from_state3(s);
		printf("<2>OP ON %x EQUALS %x\n", a.u, c.u);
		fk_printer8ind_np_mtpi3(&s);

		
		puts("Press enter to continue, but don't type anything.");
				fgetc(stdin);
		//Another test.
		s.state[0] = 0x7;
		s.state[1] = 0x6;
		s.state[2] = 0xff;
		s.state[3] = 0x0;
		fk_printer8ind_np_mtpi3(&s);
		k_and3(&s);
		c.u = from_state3(s);
		fk_printer8ind_np_mtpi3(&s);
		puts("Press enter to continue, but don't type anything.");
				fgetc(stdin);
		system("clear");

	}
	{	//Demonstration of state20. 512KB 
		state20 s20;
		
		//k_filler_np_mtp20(&s20);
		k_fillerind_mtpi20(&s20);

		//Run the prime code.
		is_prime_mtp20(&s20);
		//Run the printer.
		//fk_printer_np_mtp20(&s20);
		fk_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear"); //bruh moment
		//Try filling it.
		k_fillerind_mtpi20(&s20);
		k_endian_cond_swap3_simd_mtp20(&s20);
		fk_printer8ind32_np_mtpi20(&s20);

		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear"); //bruh moment
		if(0){
				puts("TESTING DATA EXTRACTION...");

				k_fillerind_mtpi20(&s20);
				fk_byteprinter_extract3(&s20);

				puts("Press enter to continue, but don't type anything.");
				fgetc(stdin);
				system("clear"); //bruh moment
		}
		//Use modsort.
		k_fillerind_mtpi20(&s20);
		//k_mul5_mtp20(&s20);
		k_mul5_simd_mtp20(&s20);
		//s20 = k_mul5_simd_mt20(s20);
		//s20 = k_mul5_mt20(s20);
		k_modsort_mtpie20(&s20);
		//s20 = k_modsort_mtie20(s20);
		fk_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		
		system("clear");
		//Test the shuffler.
		puts("Testing shuffle...");
		k_fillerind_mtpi20(&s20);
		k_shuffler1_3_20(&s20);
		fk_printerind_np_mtpi20(&s20);

		puts("Press enter to continue, but don't type anything.");
				fgetc(stdin);
				
				system("clear");
		//Test the worker.
		puts("Testing worker functions.The next print should be all 1's");
		k_fillerind_mtpi20(&s20);
		s20.state3s[0] = to_state3(1);
		k_dupe_upper4_sharedp3_20(&s20);
		fk_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear");
		if(1)
		{
			puts("Testing nlogn ro (This may take a while...)");
			k_upper3_4_increment_nlognrop20(&s20);
			fk_printerind_np_mtpi20(&s20);
			puts("Press enter to continue, but don't type anything.");
			fgetc(stdin);
			system("clear");
			
			puts("Testing nlogn (non rop, non parallel) this may take a while.");
			s20.state3s[0] = to_state3(1);
			k_dupe_upper4_sharedp3_20(&s20); //Fill it with 1's
			k_incrementhalves4_nlognp20(&s20); //Run our nlogn algo.
			fk_printerind_np_mtpi20(&s20);
		}

		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear");
		puts("Testing sum function...");
		//result should be 1 less than the count.
		s20.state3s[0] = to_state3(1);
		k_dupe_upper4_sharedp3_20(&s20);
		k_sum32_sharedp3_20(&s20);
		printf("Sum is %u",from_state3(s20.state3s[0]));
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear");
		//Test the halves worker.
		puts("Testing the halves worker.");
		s20.state3s[0] = to_state3(1);
		k_dupe_upper4_sharedp3_20(&s20);
		k_sum32_halvesp20(&s20);
		fk_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear");
		s20.state3s[0] = to_state3(24);
		k_dupe_upper4_sharedp3_20(&s20);
		//Run the WIND process.
		puts("TESTING WIND PROCESSING! (As opposed to, say, BLAST processing.)");
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		big_shared_process20(&s20);
		fk_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
				fgetc(stdin);
		system("clear");
	}
	//Perform ifunc on all elements in a huge array. Then, run the duplication function.
	//As you can imagine, this takes a very long time.
	{
		//Fill this 512 megabytes with incrementally increasing integers.
		k_fillerind_mtpi30(&hughmong);
		//We call the kernel.
		k_ifunc_mtp30(&hughmong);
		//We print the results.
		//fk_printerind_np_mtpi30(&hughmong);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		//Use the k_dupe_upper4 kernel on sets of 20 from hughmong
		k_dupe_upper4_sharedp3_20_mtp30(&hughmong);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		fk_printerind_np_mtpi30(&hughmong);
	}
}
