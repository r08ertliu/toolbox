#include <cuda.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>

#define BLOCK_NR 14
#define BLOCK_SZ 1024

__device__
void clock_block(clock_t clock_count)
{
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	while (clock_offset < clock_count)
		clock_offset = clock() - start_clock;
}

__device__
void sleep_ms(uint64_t ms)
{
	for (int i = 0; i < 1000000; ++i)
		__nanosleep(ms);
}

#define SMEM_SZ (24 << 10)
__global__
__launch_bounds__(BLOCK_SZ, 1)
void busy_loop(int endless)
{
	volatile __shared__ uint8_t s[SMEM_SZ << 1];
	const bool grid_leader = (blockIdx.x == 0 && threadIdx.x == 0);
	const bool block_leader = (threadIdx.x == 0);

	if (block_leader)
		printf("Block %d\n", blockIdx.x);
	if (grid_leader && endless)
		printf("Endless\n");

	if (endless) {
		while(1) {
			for (int i = 0; i < (SMEM_SZ / BLOCK_SZ); i += BLOCK_SZ) {
				if (threadIdx.x + i < SMEM_SZ)
					s[threadIdx.x + i + SMEM_SZ] = s[threadIdx.x + i];
			}
			//clock_block(2^63);
			sleep_ms(1000);
			__syncthreads();
			if (block_leader)
				printf("I'm B[%d]\n", blockIdx.x);
		}
	} else {
		//clock_block(2^63);
		sleep_ms(1000);
		__syncthreads();
	}
}

int main(int argc, char *argv[])
{
	int endless = 0;
	char *c;

	if (argc >= 2) {
		endless = strtol(argv[1], &c, 10);
	}

	// Set desired GPU device ID
	cudaSetDevice(0);

	auto start = std::chrono::system_clock::now();
	busy_loop<<<BLOCK_NR, BLOCK_SZ>>>(endless);
	cudaDeviceSynchronize();
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << std::endl;

	return 0;
}
