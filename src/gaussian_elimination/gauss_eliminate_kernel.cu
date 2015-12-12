#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_division_kernel(float *U, int k){

    __shared__ float tempDiv;
	
    if ( threadIdx.x == 0 )
        tempDiv = U[k * MATRIX_SIZE + k];
		
    __syncthreads();

    for ( int i = k + threadIdx.x; i < MATRIX_SIZE; i += blockDim.x )
        U[k * MATRIX_SIZE + i] = U[k * MATRIX_SIZE + i] / tempDiv;

    if ( threadIdx.x == 0 )
        U[k * MATRIX_SIZE + k] = 1;
}

__global__ void gauss_eliminate_kernel(float *U, int k)
{
	__shared__ float tempMem[MATRIX_SIZE];
	
    for ( int i = k + threadIdx.x; i < MATRIX_SIZE; i += blockDim.x )
        tempMem[i] = U[k * MATRIX_SIZE + i];

    __syncthreads();

	for ( int i = blockIdx.x + k + 1; i < MATRIX_SIZE; i += gridDim.x )
	{
		for ( int j = threadIdx.x + k+1; j < MATRIX_SIZE; j += blockDim.x )
			U[MATRIX_SIZE * i + j] = U[MATRIX_SIZE * i + j] - (U[MATRIX_SIZE * i + k] * tempMem[j]);
	}

	__syncthreads();

	if ( threadIdx.x == 0 )
	{
		for (int i = blockIdx.x + k + 1; i < MATRIX_SIZE; i += gridDim.x )
		{
			U[MATRIX_SIZE * i + k] = 0;
		}
	}
}
#endif