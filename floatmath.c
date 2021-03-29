
#define KERNEL_FAST_FLOAT_MATH 0
#include "kerneln.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv){
	float a1 = atof(argv[1]);
	float a2 = atof(argv[2]);
	state4 q;
		puts("Multiplication!");
	printf("Correct result is %f\n", a1 * a2);
	
	q.state3s[0] = float_to_state3(a1);
	q.state3s[1] = float_to_state3(a2);
	k_fmul_s3(&q);
	printf("Our result is %f\n", float_from_state3(q.state3s[0]));

	puts("Division!");
	printf("Correct result is %f\n", a1 / a2);
	q.state3s[0] = float_to_state3(a1);
	q.state3s[1] = float_to_state3(a2);
	k_fdiv_s3(&q);
	printf("Our result is %f\n", float_from_state3(q.state3s[0]));

	puts("Addition!");
	printf("Correct result is %f\n", a1 + a2);
	q.state3s[0] = float_to_state3(a1);
	q.state3s[1] = float_to_state3(a2);
	k_fadd_s3(&q);
	printf("Our result is %f\n", float_from_state3(q.state3s[0]));


	puts("Subtraction!");
	printf("Correct result is %f\n", a1 - a2);
	q.state3s[0] = float_to_state3(a1);
	q.state3s[1] = float_to_state3(a2);
	k_fsub_s3(&q);
	printf("Our result is %f\n", float_from_state3(q.state3s[0]));

	puts("Modulo!");
	printf("Correct result is %f\n", fmodf(a1,a2));
	q.state3s[0] = float_to_state3(a1);
	q.state3s[1] = float_to_state3(a2);
	k_fmodf_s3(&q);
	printf("Our result is %f\n", float_from_state3(q.state3s[0]));
}
