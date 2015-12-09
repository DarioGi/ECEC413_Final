 /* Device code. */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include "gauss_eliminate.h"

__global__ void gauss_eliminate_kernel(float *U, float* current_row, int offset)
{
	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	int stride = blockDim.x * gridDim.x;
	int j = 0, i = 0;
	
	for ( int k = 0; k < MATRIX_SIZE; k++ )
	{
		for (j = k + tId + 1; j < MATRIX_SIZE; j += stride)
		{ 
			if ( U[MATRIX_SIZE*k + k] == 0 )
				printf("Numerical instability detected. The principal diagonal element is zero. \n");
			U[MATRIX_SIZE * k + j] = (float)(U[MATRIX_SIZE * k + j] / U[MATRIX_SIZE * k + k]); // Division step
		}
		__syncthreads();
		if ( threadIdx.x == 0 && blockIdx.x == 0 )
			U[MATRIX_SIZE * k + k] = 1; // Set the principal diagonal entry in U to be 1 
		__syncthreads();
		
		for (i = (k+1); i < MATRIX_SIZE; i++)
		{
			for (j = (k+tId+1); j < MATRIX_SIZE; j += stride)
				U[MATRIX_SIZE * i + j] = U[MATRIX_SIZE * i + j] - (U[MATRIX_SIZE * i + k] * U[MATRIX_SIZE * k + j]); // Elimination step
			__syncthreads();
			if ( threadIdx.x == 0 && blockIdx.x == 0 )
				U[MATRIX_SIZE * i + k] = 0; 
			__syncthreads();
		} 
	}
}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
