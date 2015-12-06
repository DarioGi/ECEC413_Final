#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>

// includes, kernels
#include "trap_kernel.cu"


#define LEFT_ENDPOINT 10
#define RIGHT_ENDPOINT 1005
#define NUM_TRAPEZOIDS 100000000

double compute_on_device(float, float, int, float);
extern "C" double compute_gold(float, float, int, float);

int 
main(void) 
{
    int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);

	double reference = compute_gold(a, b, n, h);
    printf("Reference solution computed on the CPU = %f \n", reference);

	/* Write this function to complete the trapezoidal on the GPU. */
	double gpu_result = compute_on_device(a, b, n, h);
	printf("Solution computed on the GPU = %f \n", gpu_result);
} 

/* Complete this function to perform the trapezoidal rule on the GPU. */
double 
compute_on_device(float a, float b, int n, float h)
{
    return 0.0;
}



