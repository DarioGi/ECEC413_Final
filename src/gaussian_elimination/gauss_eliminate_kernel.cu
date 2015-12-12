 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(float *U, int k)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	int stride = blockDim.x * gridDim.x;
	int j = 0, i = 0;
	__shared__ float tempPrincipal;
	__shared__ float tempMem[MATRIX_SIZE];
	float temp, temp1;
	if ( threadIdx.x == 0 && threadIdx.y == 0 )
		tempPrincipal = U[MATRIX_SIZE * k + k];
		
	__syncthreads();
	
	for (j = k + threadIdx.x + 1; j < MATRIX_SIZE; j += blockDim.x )
	{
		tempMem[j] = U[MATRIX_SIZE * k + j];
		tempMem[j] = tempMem[j] / tempPrincipal;
	}
	__syncthreads();
	
	for ( i = (k+1); i < MATRIX_SIZE; i++ )
	{
		for ( j = k+tId+1; j < MATRIX_SIZE; j += stride )
		{
			temp = U[MATRIX_SIZE * i + j];
			temp1 = U[MATRIX_SIZE * i + k];
			U[MATRIX_SIZE * i + j] = temp - (temp1 * tempMem[j]); // Elimination step
		}
	}
	__syncthreads();
	for (j = k + tId + 1; j < MATRIX_SIZE; j += stride )
	{
		U[MATRIX_SIZE * k + j] = tempMem[j];
	}
}

__global__ void gauss_eliminate_zeroOut(float *U, int k)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);// + (threadIdx.y*blockDim.y);
	int stride = blockDim.x * gridDim.x;
	
	for (int j = k + tId + 1; j < MATRIX_SIZE; j += stride )
	{
		U[MATRIX_SIZE * j + k] = 0;
	}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
