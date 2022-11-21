#include <cuda.h>
#include <iostream>
#include <math.h>
#include <chrono>
#include <ctime>
#include <unistd.h>

#define SMEM_SZ (48 << 10)
#define BLOCK_NR 14
#define BLOCK_SZ 1024
#define LOOP_SEC 1

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

__global__
__launch_bounds__(BLOCK_SZ, 1)
void busy_loop(int time)
{
	volatile __shared__ uint8_t s[SMEM_SZ];
	const uint32_t half_smem_sz = SMEM_SZ >> 1;
	const bool block_leader = (threadIdx.x == 0);

	if (block_leader)
		printf("Activate B[%d]\n", blockIdx.x);

	if (!time) {
		while(1) {
			for (int i = 0; i < (half_smem_sz / gridDim.x); i += gridDim.x) {
				if (threadIdx.x + i < half_smem_sz)
					s[threadIdx.x + i + half_smem_sz] = s[threadIdx.x + i];
			}
			//clock_block(2^63);
			sleep_ms(1000);
			__syncthreads();
			if (block_leader)
				printf("I'm B[%d]\n", blockIdx.x);
		}
	} else {
		//clock_block(2^63);
		sleep_ms(1000 * time);
		__syncthreads();
	}
}

void usage()
{
	printf("Usage: busy_loop [OPTION]...\n");
	printf("\n");
	printf("  -d, GPU device ID, default = 0\n");
	printf("  -n, GPU kernel block number, default = %d\n", BLOCK_NR);
	printf("  -s, GPU kernel block size, default = %d\n", BLOCK_SZ);
	printf("  -t, Loop time in second, default = %d, set 0 for endless\n", LOOP_SEC);
	printf("  -h, Disaply usage\n");
}

int main(int argc, char *argv[])
{
	int dev_id = 0;
	int block_nr = BLOCK_NR;
	int block_sz = BLOCK_SZ;
	int time = LOOP_SEC;

	int o;
	const char *optstring = "d:n:s:t:h";
	while ((o = getopt(argc, argv, optstring)) != -1) {
		switch (o) {
			case 'd':
				dev_id = std::stoi(optarg);
				break;
			case 'n':
				block_nr = std::stoi(optarg);
				break;
			case 's':
				block_sz = std::stoi(optarg);
				break;
			case 't':
				time = std::stoi(optarg);
				break;
			case 'h':
				usage();
				return 0;
			default:
				printf("Unknown args %s\n", optarg);
				usage();
				return -1;
		}
	}

	cudaError_t rc;

	rc = cudaSetDevice(dev_id);
	if (rc != cudaSuccess) {
		printf("Failed to set GPU device %d, %s\n", dev_id, cudaGetErrorString(rc));
		return -1;
	}

	if (!time) {
		printf("Endless mode\n");
	} else {
		printf("Loop for %d second\n", time);
	}

	auto start = std::chrono::system_clock::now();
	busy_loop<<<block_nr, block_sz>>>(time);
	rc = cudaGetLastError();
	if (rc != cudaSuccess) {
		printf("Failed to launch kernel on device %d, %s\n", dev_id, cudaGetErrorString(rc));
		return -1;
	}

	rc = cudaDeviceSynchronize();
	if (rc != cudaSuccess) {
		printf("Failed to sync device %d, %s\n", dev_id, cudaGetErrorString(rc));
		return -1;
	}

	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "elapsed time: " << elapsed_seconds.count() << std::endl;

	return 0;
}
