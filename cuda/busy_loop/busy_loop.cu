#include <cuda.h>
#include <chrono>
#include <unistd.h>
#include <thread>
#include <pthread.h>
#include <signal.h>

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
void busy_loop(int time, volatile bool *dev_should_stop)
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
			if (*dev_should_stop)
				break;
		}
	} else {
		//clock_block(2^63);
		for (int i = 0; i < time; ++i) {
			if (*dev_should_stop)
				break;
			sleep_ms(1000);
		}
	}
}

bool *host_should_stop;
int signal_thread_ready = 0;
int run_signal_thread = 0;
static void* signal_handler_thread(void *arg) {
	sigset_t *set = (sigset_t*)arg;
	int ret, signal;

	while (!signal_thread_ready)
		std::this_thread::sleep_for(std::chrono::seconds(1));

	while (run_signal_thread) {
		ret = sigwait(set, &signal);
		if (ret != 0) {
			printf("Sigwait error %d\n", ret);
			continue;
		}

		printf("Got signal %d/%s\n", signal, strsignal(signal));

		switch (signal) {
			case SIGINT:
			case SIGABRT:
			case SIGSEGV:
			case SIGTERM:
				*host_should_stop = true;
				run_signal_thread = 0;
				break;
			default:
				printf("Caught unhandled signal: %d\n", signal);
		}
	}
	return NULL;
}

static int start_signal_thread(pthread_t *sig_thread, sigset_t *set)
{
	int ret;

	sigemptyset(set);
	sigaddset(set, SIGINT);
	sigaddset(set, SIGABRT);
	sigaddset(set, SIGSEGV);
	sigaddset(set, SIGTERM);

	ret = pthread_sigmask(SIG_BLOCK, set, NULL);
	if (ret != 0) {
		printf("Unable to setup signal set: %d\n", ret);
		return ret;
	}

	run_signal_thread = 1;
	__sync_synchronize();
	ret = pthread_create(sig_thread, NULL, &signal_handler_thread, (void*)set);
	if (ret != 0) {
		printf("Unable to create signal thread: %d\n", ret);
		return ret;
	}
	return 0;
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

	pthread_t sig_thread;
	sigset_t sig_set;

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

	if (start_signal_thread(&sig_thread, &sig_set)) {
		printf("Failed to start signal thread\n");
		return -1;
	}

	cudaError_t rc;

	rc = cudaSetDevice(dev_id);
	if (rc != cudaSuccess) {
		printf("Failed to set GPU device %d, %s\n", dev_id, cudaGetErrorString(rc));
		return -1;
	}

	rc = cudaMallocManaged((void**)&host_should_stop, sizeof(*host_should_stop));
	if (rc != cudaSuccess) {
		printf("Failed to allocate UMA, %s\n", cudaGetErrorString(rc));
		return -1;
	}

	rc = cudaMemAdvise(host_should_stop, sizeof(*host_should_stop), cudaMemAdviseSetPreferredLocation, dev_id);
	if (rc != cudaSuccess) {
		printf("Failed to MemAdvise cudaMemAdviseSetPreferredLocation, %s\n", cudaGetErrorString(rc));
		return -1;
	}

	rc = cudaMemAdvise(host_should_stop, sizeof(*host_should_stop), cudaMemAdviseSetReadMostly, dev_id);
	if (rc != cudaSuccess) {
		printf("Failed to MemAdvise cudaMemAdviseSetReadMostly, %s\n", cudaGetErrorString(rc));
		return -1;
	}
	*host_should_stop = false;
	signal_thread_ready = 1;

	if (!time) {
		printf("Endless mode\n");
	} else {
		printf("Loop for %d second\n", time);
	}

	auto start = std::chrono::system_clock::now();
	busy_loop<<<block_nr, block_sz>>>(time, host_should_stop);
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
	printf("elapsed time: %f\n", elapsed_seconds.count());

	signal_thread_ready = 1;
	run_signal_thread = 0;
	pthread_kill(sig_thread, SIGTERM);
	pthread_join(sig_thread, NULL);

	return 0;
}
