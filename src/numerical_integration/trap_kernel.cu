#include <stdio.h>
#include <math.h>
#include <float.h>
#include "trap.h"


/* Write GPU kernels to compete the functionality of estimating the integral via the trapezoidal rule. */ 

__device__ void lock(int *mutex)
{
	while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex)
{
	atomicExch(mutex, 0);
}

__global__ void trap_kernel(float* dA, int* dN, float* dH, double* dRes, int* mutex)
{
	__shared__ double tRes[TILE_SIZE];
	int tId = threadIdx.x + blockDim.x * blockIdx.x;
	double localSum = 0.0;
	int numElements = *dN;
	float stride = gridDim.x * blockDim.x;
	float a = *dA;
	float h = *dH;
	double temp = 0;
	for ( int i = tId; i < numElements; i += stride )
	{
		temp = a+i*h;
		localSum += (temp + 1)/sqrt(temp*temp + temp + 1);
	}
	
	tRes[threadIdx.x] = localSum;
	__syncthreads();
	
	unsigned int i = TILE_SIZE / 2;
	while ( i != 0 )
	{
		if ( threadIdx.x < i ) 
			tRes[threadIdx.x] += tRes[threadIdx.x + i];
		__syncthreads();
		i /= 2;
	}
	
	if ( threadIdx.x == 0 ) 
	{
        lock(mutex);
        *dRes += tRes[0];
        unlock(mutex);
    }
}