#include <cuda.h>
#include <iostream>
#include <math.h>

// CUDA kernel to add elements of two arrays
__global__
void breaker()
{
	int *ptr = (int*)0xDEADBEEF;
	*ptr = 0;
}

int main(void)
{
        cudaError_t rc;
        cudaStream_t s;

	rc = cudaSetDevice(0);
        if (rc != cudaSuccess) {
                printf("Failed to set device, %s, rc = %d\n", cudaGetErrorString(rc), rc);
                return;
        }

        rc = cudaStreamCreate(&s);
        if (rc != cudaSuccess) {
                printf("Failed to create stream, %s, rc = %d\n", cudaGetErrorString(rc), rc);
                return;
        }

        breaker<<< 1, 32, 0, s>>>();
        rc = cudaStreamSynchronize(s);
        if (rc != cudaSuccess) {
                printf("cudaStreamSynchronize return %d, %s\n", rc, cudaGetErrorString(rc));
        }
	cudaDeviceSynchronize();
	cudaStreamDestory(s);
	return 0;
}
