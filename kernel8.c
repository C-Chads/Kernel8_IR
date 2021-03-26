
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
//Kernel definition functions.
#include "kerneln.h"



//The first Kernel8 ever written.
state1 and127(state1 c){
	//printf("Operating on %x", (uint32_t)c.state[0]);
	c = to_state1(from_state1(c) & 127);
	return c;
}
state1 and63(state1 c){
	c = to_state1(from_state1(c) & 63);
	return c;
}

//Kernel32- return if prime else 0
state3 is_prime(state3 c){
	int32_t val = signed_from_state3(c);
	if(val == 2 || val == 3) return c;
	if(val == 1) return to_state3(0);
	if(val%2 == 0) return to_state3(0);
	for(int32_t i = 3; i < val/2; i+=2){
		if(val%i == 0) return to_state3(0);
	}
	return c;
}


//High state3 is the index, Low state3 is the data at that index.
state4 k_fillerind(state4 c){ //Real kernel using the "MultiplexIndexed" syntax.
	state4 ret;
	uint32_t index = from_state3(k_high4(c));
	ret = statemix3(state3_zero(), to_state3(index));
	return ret;
}
state3 k_mul5(state3 c){
	return to_state3(from_state3(c)*5);
}
state4 k_modsort(state4 c){ //Use the value at the index to choose its placement.
	//uint32_t index = from_state3(k_high4(c));
	uint32_t val = from_state3(k_low4(c));
	return statemix3(to_state3(val), to_state3(val));
}

//Real variant of k_printer! Using the "indexed" multiplex syntax.
void k_printerind(state4 *c){
	printf("%u, %u\n", from_state3(k_high4(*c)), from_state3(k_low4(*c)) );
}
//Print individual bytes, with an 8 bit index.
state2 k_printer8ind(state2 c){
	printf("BP! %u, %u\n", from_state1(k_high2(c)), from_state1(k_low2(c)));
	return c;
}
//Print individual bytes, with a 32 bit index.
state4 k_printer8ind32(state4 c){
	uint32_t ind = from_state3(c.state3s[0])<<2; //We recieved four bytes of data!
	state3 dataseg = c.state3s[1];
	uint8_t bytes[4];
#pragma omp simd
	for(size_t i = 0; i < 4; i++)
		bytes[i] = from_state1(dataseg.state1s[i]);
	for(uint32_t i = 0; i < 4; i++)
		printf("BP32! %u, %u\n", ind + i, bytes[i]);
	return c;
}
//Summer.
state4 k_sum32(state4 c){
	uint32_t high = from_state3(k_high4(c));
	uint32_t low = from_state3(k_low4(c));
	return statemix3(
			to_state3(high + low), //The upper half, the shared state.
			k_low4(c)//the lower half, the non-shared state.
		);
}

state4 k_dupe_upper4(state4 c){
	return statemix3(
			k_high4(c), //The upper state3, the shared state.
			k_high4(c)	//the lower state3, the non-shared state.
		);
}

state3 k_ifunc(state3 c){ //A real kernel.
	//Performs rot right of 4.
	//return to_state3( (from_state3(c)>>4) | (from_state3(c)<< (32 - 4 )) );
	return to_state3( from_state3(c) / 3 );
}
//Generate a multiplexing of and127 from state1 to state3.
KERNEL_MULTIPLEX_SIMD(and127_mt3, and127, 1, 3,     1)


//Endian conditional swaps for byte printing.
KERNEL_MULTIPLEX_POINTER_SIMD(k_endian_cond_swap3_simd_mtp20, k_endian_cond_swap3, 3, 20, 1)

//Multiply unsigned integers by 5.
KERNEL_MULTIPLEX_POINTER_SIMD(k_mul5_simd_mtp20, k_mul5, 3, 20, 1)
KERNEL_MULTIPLEX_SIMD(k_mul5_simd_mt20, k_mul5, 3, 20, 1)
KERNEL_MULTIPLEX(k_mul5_mt20, k_mul5, 3, 20, 1)

//Multiplex is_prime by pointer to state20.
KERNEL_MULTIPLEX_POINTER(is_prime_mtp20, is_prime, 3, 20, 1)
//multiplex our divide by 7 function from state3 to state30
KERNEL_MULTIPLEX_POINTER(k_ifunc_mtp30, k_ifunc, 3, 30, 1)
//Fake kernel to fill an array with values
//Real variants of those kernels. Notice that filler can now be parallelized because it no longer
//relies on global state.
KERNEL_MULTIPLEX_INDEXED_POINTER(k_fillerind_mtpi20, k_fillerind, 3, 4, 20, 1);
KERNEL_MULTIPLEX_INDEXED_POINTER(k_fillerind_mtpi30, k_fillerind, 3, 4, 30, 1);
//Notice the last argument- these are NOT pass-by-copy these are PASS-BY-POINTER.
KERNEL_MULTIPLEX_INDEXED_NP_POINTER(k_printerind_np_mtpi20, k_printerind, 3, 4, 20, 0);
KERNEL_MULTIPLEX_INDEXED_NP_POINTER(k_printerind_np_mtpi30, k_printerind, 3, 4, 30, 0);
//Print as uint8_t's
KERNEL_MULTIPLEX_INDEXED_NP_POINTER(k_printer8ind_np_mtpi3, k_printer8ind, 1, 2, 3, 1);
KERNEL_MULTIPLEX_INDEXED_NP_POINTER(k_printer8ind_np_mtpi20, k_printer8ind, 1, 2, 20, 1);
KERNEL_MULTIPLEX_INDEXED_NP_POINTER(k_printer8ind_np_mtpi30, k_printer8ind, 1, 2, 30, 1);
//Print as uint8_t's with 32 bit index.
KERNEL_MULTIPLEX_INDEXED_NP_POINTER(k_printer8ind32_np_mtpi20, k_printer8ind32, 3, 4, 20, 1);
KERNEL_MULTIPLEX_INDEXED_NP_POINTER(k_printer8ind32_np_mtpi30, k_printer8ind32, 3, 4, 30, 1);
//Emplacing modsort
KERNEL_MULTIPLEX_INDEXED_EMPLACE(k_modsort_mtie20, k_modsort, 3, 4, 20, 1);
KERNEL_MULTIPLEX_POINTER_INDEXED_EMPLACE(k_modsort_mtpie20, k_modsort, 3, 4, 20, 1);
//Shared state worker.
KERNEL_SHARED_STATE_POINTER(k_sum32_sharedp3_20, k_sum32, 3, 4, 20, 1)
KERNEL_SHARED_STATE_POINTER(k_dupe_upper4_sharedp3_20, k_dupe_upper4, 3, 4, 20, 1)
//The real magic! We can use our previously generated kernels to
//create NEW kernels.
//This has *infinite possibilities*.
KERNEL_MULTIPLEX_POINTER(k_dupe_upper4_sharedp3_20_mtp30,k_dupe_upper4_sharedp3_20 , 20, 30,0)

static kernelb1 and_7667_funcs[4] = {
	and127,
	and63,
	and63,
	and127
};
//Multikernel
KERNEL_MULTIPLEX_MULTIKERNEL_POINTER(and_7667, and_7667_funcs, 1, 3, 1);


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
		s = and127_mt3(s);
		c.u = from_state3(s);
		printf("OP ON %x EQUALS %x\n", a.u, c.u);
		printf("Sizeof state10: %zu\n",sizeof(state10));
		k_printer8ind_np_mtpi3(&s);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);

		//Another test.
		a.u = rand();
		a.u = 0xffffffff;
		s=to_state3(a.u);
		and_7667(&s);
		c.u = from_state3(s);
		printf("<2>OP ON %x EQUALS %x\n", a.u, c.u);
		k_printer8ind_np_mtpi3(&s);
		puts("Press enter to continue, but don't type anything.");
				fgetc(stdin);
	}
	{	//Demonstration of state20. 512KB 
		state20 s20;
		
		//k_filler_np_mtp20(&s20);
		k_fillerind_mtpi20(&s20);

		//Run the prime code.
		is_prime_mtp20(&s20);
		//Run the printer.
		//k_printer_np_mtp20(&s20);
		k_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear"); //bruh moment
		//Try filling it.
		k_fillerind_mtpi20(&s20);
		k_endian_cond_swap3_simd_mtp20(&s20);
		k_printer8ind32_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear"); //bruh moment
		//Use modsort.
		k_fillerind_mtpi20(&s20);
		//k_mul5_simd_mtp20(&s20);
		//s20 = k_mul5_simd_mt20(s20);
		s20 = k_mul5_mt20(s20);
		k_modsort_mtpie20(&s20);
		//s20 = k_modsort_mtie20(s20);
		k_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		
		system("clear");
		//Test the worker.
		puts("Testing worker functions.The next print should be all 1's");
		k_fillerind_mtpi20(&s20);
		s20.state3s[0] = to_state3(1);
		k_dupe_upper4_sharedp3_20(&s20);

		k_printerind_np_mtpi20(&s20);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		//result should be 1 less than the count.
		k_sum32_sharedp3_20(&s20);
		printf("Sum is %u",from_state3(s20.state3s[0]));
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		system("clear");
	}
	//Perform a division by 7 on 512 megabytes of data as int32's
	//As you can imagine, this takes a very long time.
	{
		//Fill this 512 megabytes with incrementally increasing integers.
		k_fillerind_mtpi30(&hughmong);
		//We call the kernel.
		k_ifunc_mtp30(&hughmong);
		//We print the results.
		//k_printerind_np_mtpi30(&hughmong);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
		//Use the k_dupe_upper4 kernel on sets of 20 from hughmong
		k_dupe_upper4_sharedp3_20_mtp30(&hughmong);
		puts("Press enter to continue, but don't type anything.");
		fgetc(stdin);
				k_printerind_np_mtpi30(&hughmong);
	}
}
