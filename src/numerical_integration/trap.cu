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
extern float f(float x);

int main(void) 
{
    int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	double reference = compute_gold(a, b, n, h);
    printf("Reference solution computed on the CPU = %f \n", reference);
	gettimeofday(&stop, NULL);
	printf("Gold Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	
	/* Write this function to complete the trapezoidal on the GPU. */
	double gpu_result = compute_on_device(a, b, n, h);
	printf("Solution computed on the GPU = %f \n", gpu_result);
} 

/* Complete this function to perform the trapezoidal rule on the GPU. */
double compute_on_device(float a, float b, int n, float h)
{
	float* dH = NULL;
	int* dN = NULL;
	float* dA = NULL;
	double* dRes = NULL;
	double hRes = (f(a) + f(b))/2.0;
	
	if ( cudaMalloc((void**)&dH, sizeof(float)) != cudaSuccess ||
		cudaMalloc((void**)&dN, sizeof(int))!= cudaSuccess ||
		cudaMalloc((void**)&dA, sizeof(float)) != cudaSuccess ||
		cudaMalloc((void**)&dRes, sizeof(double)) != cudaSuccess )
	{
		printf("Failed allocation.\n");
		return -1;
	}	
	printf("Allocating...");
	cudaMemcpy(dH, &h, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dN, &n, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dA, &a, sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dRes, &hRes, sizeof(double), cudaMemcpyHostToDevice);
	printf("Done.\n");
	int *mutex = NULL;
    cudaMalloc((void **)&mutex, sizeof(int));
    cudaMemset(mutex, 0, sizeof(int));
	
	// Timing
	struct timeval start, stop;	
	gettimeofday(&start, NULL);
	trap_kernel<<<GRID_SIZE, TILE_SIZE>>>(dA, dN, dH, dRes, mutex);
	
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err ) 
	{
		fprintf(stderr, "Kernel execution failed: %s.\n", cudaGetErrorString(err));
		return -1.0;
	}
	
	gettimeofday(&stop, NULL);
	printf("GPU Execution time = %fs. \n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
	cudaMemcpy(&hRes, dRes, sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(dH);
	cudaFree(dN);
	cudaFree(dA);
	cudaFree(dRes);
	cudaFree(mutex);
	
    return hRes * h;
}



