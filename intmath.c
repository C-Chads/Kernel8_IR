#include "kerneln.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv){
	int32_t a1 = atoi(argv[1]);
	int32_t a2 = atoi(argv[2]);
	state4 q;
	puts("Multiplication!");
	printf("Correct result is %d\n", a1 * a2);
	
	q.state3s[0] = signed_to_state3(a1);
	q.state3s[1] = signed_to_state3(a2);
	k_smul_s3(&q);
	printf("Our result is %d\n", signed_from_state3(q.state3s[0]));

	puts("Division!");
	printf("Correct result is %d\n", a1 / a2);
	q.state3s[0] = signed_to_state3(a1);
	q.state3s[1] = signed_to_state3(a2);
	k_sdiv_s3(&q);
	printf("Our result is %d\n", signed_from_state3(q.state3s[0]));

	puts("Addition!");
	printf("Correct result is %d\n", a1 + a2);
	q.state3s[0] = signed_to_state3(a1);
	q.state3s[1] = signed_to_state3(a2);
	k_sadd_s3(&q);
	printf("Our result is %d\n", signed_from_state3(q.state3s[0]));


	puts("Subtraction!");
	printf("Correct result is %d\n", a1 - a2);
	q.state3s[0] = signed_to_state3(a1);
	q.state3s[1] = signed_to_state3(a2);
	k_ssub_s3(&q);
	printf("Our result is %d\n", signed_from_state3(q.state3s[0]));

	puts("Modulo!");
	printf("Correct result is %d\n", a1 % a2);
	q.state3s[0] = signed_to_state3(a1);
	q.state3s[1] = signed_to_state3(a2);
	k_smod_s3(&q);
	printf("Our result is %d\n", signed_from_state3(q.state3s[0]));
}
