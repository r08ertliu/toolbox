#include <cuda.h>
#include <nvml.h>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <vector>

__device__
void clock_block(clock_t clock_count)
{
	clock_t start_clock = clock();
	clock_t clock_offset = 0;
	while (clock_offset < clock_count)
		clock_offset = clock() - start_clock;
}

// CUDA kernel to add elements of two arrays
__global__
void add(int n, float *x, float *y, int *dev_int)
{
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		printf("[add kernel]: dev_int = %d\n", *dev_int);
	}

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = 0; i < 30000000; i++) {
		clock_block(2^63);
	}
	for (int i = index; i < n; i += stride)
		y[i] = x[i] + y[i];
}

__global__
void breaker(int *dev_int)
{
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		printf("[breaker kernel]: dev_int = %d\n", *dev_int);
	}

	for (int i = 0; i < 30000000; i++) {
		clock_block(2^63);
	}

	int *bad_ptr = (int*)0xDEADBEEF;
	*bad_ptr = 0;
}

static std::string get_uuid_str(const unsigned char *uuid_ptr) {
	std::stringstream uuid_ss;
	uuid_ss << "GPU-" << std::setfill('0') << std::hex
		<< std::setw(2) << (unsigned)uuid_ptr[0]
		<< std::setw(2) << (unsigned)uuid_ptr[1]
		<< std::setw(2) << (unsigned)uuid_ptr[2]
		<< std::setw(2) << (unsigned)uuid_ptr[3] << "-"
		<< std::setw(2) << (unsigned)uuid_ptr[4]
		<< std::setw(2) << (unsigned)uuid_ptr[5] << "-"
		<< std::setw(2) << (unsigned)uuid_ptr[6]
		<< std::setw(2) << (unsigned)uuid_ptr[7] << "-"
		<< std::setw(2) << (unsigned)uuid_ptr[8]
		<< std::setw(2) << (unsigned)uuid_ptr[9] << "-"
		<< std::setw(2) << (unsigned)uuid_ptr[10]
		<< std::setw(2) << (unsigned)uuid_ptr[11]
		<< std::setw(2) << (unsigned)uuid_ptr[12]
		<< std::setw(2) << (unsigned)uuid_ptr[13]
		<< std::setw(2) << (unsigned)uuid_ptr[14]
		<< std::setw(2) << (unsigned)uuid_ptr[15];
	return uuid_ss.str();
}

#define DEV_NUM 2
int host_int = 123;
int prepareGPU(int dev_id)
{
	cudaError_t rc;

	nvmlInit();

	rc = cudaSetDevice(dev_id);
	if (rc != cudaSuccess) {
		printf("[%d]: Failed to set device %d, %s, rc = %d\n", getpid(), dev_id, cudaGetErrorString(rc), rc);
		return -1;
	}

	rc = cudaDeviceReset();
	if (rc != cudaSuccess) {
		printf("[%d]: Failed to reset device %d, %s, rc = %d\n", getpid(), dev_id, cudaGetErrorString(rc), rc);
		return -1;
	}

	rc = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (rc != cudaSuccess) {
		printf("[%d]: Failed to set device %d flag, %s, rc = %d\n", getpid(), dev_id, cudaGetErrorString(rc), rc);
		return -1;
	}

	CUresult cu_rc;
	char const * cu_err_msg = nullptr;
	CUdevice cu_dev;
	CUcontext cu_ctx;
        cu_rc = cuDeviceGet(&cu_dev, dev_id);
        if (cu_rc != CUDA_SUCCESS) {
                printf("Failed to get CUdevice, %s\n",
                              cuGetErrorString(cu_rc, &cu_err_msg) == CUDA_ERROR_INVALID_VALUE ?
                              "unknown error" : cu_err_msg);
                return -1;
        }

        cu_rc = cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev);
        if (cu_rc != CUDA_SUCCESS) {
                printf("Failed to retain primary context, %s\n",
                              cuGetErrorString(cu_rc, &cu_err_msg) == CUDA_ERROR_INVALID_VALUE ?
                              "unknown error" : cu_err_msg);
                return -1;
        }

	nvmlReturn_t nvmlrc;
	nvmlDevice_t nvmlDev;
	cudaDeviceProp devprop;
	std::string uuid_str;
	rc = cudaGetDeviceProperties(&devprop, dev_id);
	if (rc != cudaSuccess) {
		printf("Failed to get CUDA device properties, %s\n", cudaGetErrorString(rc));
		return -1;
	}

	uuid_str = get_uuid_str((unsigned char *)&devprop.uuid);
	nvmlrc = nvmlDeviceGetHandleByUUID(uuid_str.c_str(), &nvmlDev);
	if (nvmlrc != NVML_SUCCESS) {
		printf("Failed to get NVML Device %s, %s\n", uuid_str.c_str(), nvmlErrorString(nvmlrc));
		return -1;
	}

	nvmlrc = nvmlDeviceSetComputeMode(nvmlDev, NVML_COMPUTEMODE_EXCLUSIVE_PROCESS);
	if (nvmlrc != NVML_SUCCESS) {
		printf("Failed to set Exclusive Process Compute Mode, %s\n", nvmlErrorString(nvmlrc));
		return -1;
	}

	nvmlrc = nvmlDeviceSetPersistenceMode(nvmlDev, NVML_FEATURE_ENABLED);
	if (nvmlrc != NVML_SUCCESS) {
		printf("Failed to set Persistence Mode, %s\n", nvmlErrorString(nvmlrc));
		return -1;
	}
	nvmlShutdown();

	rc = cudaHostRegister(&host_int, sizeof(host_int), cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (rc != cudaSuccess) {
                printf("Failed to register host_int %p, %s\n", &host_int, cudaGetErrorString(rc));
                return -1;
        }

	return 0;
}

void releaseGPU(int dev_id, cudaStream_t &s)
{
	cudaDeviceSynchronize();
	cudaStreamDestroy(s);
	cudaHostUnregister(&host_int);

	CUresult cu_rc;
	CUdevice cu_dev;
	cu_rc = cuDeviceGet(&cu_dev, dev_id);
	if (cu_rc == CUDA_SUCCESS)
		cuDevicePrimaryCtxRelease(cu_dev);
	printf("[%d]: Release device %d\n", getpid(), dev_id);
}

const bool break_me[DEV_NUM] = {true, false};
const int use_dev[DEV_NUM] = {0,1};
const int relaunch_dev = 0;
int main(void)
{
	int N = 1<<12;
	float *x, *y;
	// Launch kernel on 1M elements on the GPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	std::vector<pid_t> child_pids;
	pid_t pid = fork();
	if (pid > 0) {
		printf("[%d]: Fork child1 %d\n", getpid(), pid);
		child_pids.push_back(pid);
	}

	if (child_pids.size() > 0) {
		pid = fork();
		if (pid > 0) {
			printf("[%d]: Fork child2 %d\n", getpid(), pid);
			child_pids.push_back(pid);
		}
	}

	if (child_pids.size() == 2) {
		printf("[%d]: I'm parent, Child1 pid = %d, Child2 pid = %d\n", getpid(), child_pids[0], child_pids[1]);
	} else if (child_pids.size() == 0) {
		int dev_id = use_dev[child_pids.size()];
		printf("[%d]: Child1 is using device %d\n", getpid(), dev_id);

		int ret = 0;
		int array_idx = dev_id % DEV_NUM;
		cudaError_t rc1;
		cudaStream_t stm;

		if (prepareGPU(dev_id)) {
			ret = 1;
			goto release1;
		}

		void *dev_int1;
		rc1 = cudaHostGetDevicePointer(&dev_int1, &host_int, 0);
		if (rc1 != cudaSuccess) {
			printf("Failed to get device pointer of %p, %s\n", &host_int, cudaGetErrorString(rc1));
			ret = 1;
			goto release1;
		}

		if (!break_me[array_idx]) {
			// Allocate Unified Memory -- accesstmble from CPU or GPU
			cudaMallocManaged(&x, N*sizeof(float));
			cudaMallocManaged(&y, N*sizeof(float));
		}

		rc1 = cudaStreamCreate(&stm);
		if (rc1 != cudaSuccess) {
			printf("[%d]: Failed to create stream, %s, rc1 = %d\n", getpid(), cudaGetErrorString(rc1), rc1);
			ret = 1;
			goto release1;
		}

		printf("[%d]: Child1 kernel launch\n", getpid());
		if (break_me[array_idx])
			breaker<<< 1, 32, 0, stm>>>((int*)dev_int1);
		else
			add<<<numBlocks, blockSize, 0, stm>>>(N, x, y, (int*)dev_int1);

		rc1 = cudaStreamSynchronize(stm);
		if (rc1 != cudaSuccess) {
			printf("[%d]: Device %d, cudaStreamSynchronize return %d, %s\n", getpid(), dev_id, rc1, cudaGetErrorString(rc1));
			ret = 2;
			goto release1;
		}
		printf("[%d]: Child1 kernel end\n", getpid());
release1:
		if (!break_me[array_idx]) {
			cudaFree(x);
			cudaFree(y);
		}
		releaseGPU(dev_id, stm);
		exit(ret);
	} else if (child_pids.size() == 1) {
		int dev_id = use_dev[child_pids.size()];
		printf("[%d]: Child2 is using device %d\n", getpid(), dev_id);

		int ret = 0;
		int array_idx = dev_id % DEV_NUM;
		cudaError_t rc2;
		cudaStream_t stm;

		if (prepareGPU(dev_id)) {
			ret = 1;
			goto release2;
		}

		void *dev_int2;
		rc2 = cudaHostGetDevicePointer(&dev_int2, &host_int, 0);
		if (rc2 != cudaSuccess) {
			printf("[%d]: Failed to get device pointer of %p, %s\n", getpid(), &host_int, cudaGetErrorString(rc2));
			ret = 1;
			goto release2;
		}

		if (!break_me[array_idx]) {
			// Allocate Unified Memory -- accesstmble from CPU or GPU
			cudaMallocManaged(&x, N*sizeof(float));
			cudaMallocManaged(&y, N*sizeof(float));
		}

		rc2 = cudaStreamCreate(&stm);
		if (rc2 != cudaSuccess) {
			printf("[%d]: Failed to create stream, %s, rc2 = %d\n", getpid(), cudaGetErrorString(rc2), rc2);
			ret = 1;
			goto release2;
		}

		printf("[%d]: Child2 kernel launch\n", getpid());
		if (break_me[array_idx])
			breaker<<< 1, 32, 0, stm>>>((int*)dev_int2);
		else
			add<<<numBlocks, blockSize, 0, stm>>>(N, x, y, (int*)dev_int2);

		rc2 = cudaStreamSynchronize(stm);
		if (rc2 != cudaSuccess) {
			printf("[%d]: Device %d, cudaStreamSynchronize return %d, %s\n", getpid(), dev_id, rc2, cudaGetErrorString(rc2));
			ret = 2;
			goto release2;
		}
		printf("[%d]: Child2 kernel end\n", getpid());

release2:
		if (!break_me[array_idx]) {
			cudaFree(x);
			cudaFree(y);
		}
		releaseGPU(dev_id, stm);
		exit(ret);
	} else {
		printf("Error\n");
	}

	printf("[%d]: Parent wait Child1 and Child2\n", getpid());

	bool relaunch = false;
	while (1) {
		int wstatus = 0;
		int exit_status = 0;
		int rc = wait(&wstatus);

		exit_status = WEXITSTATUS(wstatus);
		if (exit_status) {
			printf("[%d]: Child[%d] return error %d\n", getpid(), rc, exit_status);
			relaunch = true;
		}

		if (rc > 0)
			continue;
		break;
	}

	if (!relaunch) {
		printf("[%d]: No error\n", getpid());
		return 0;
	}

	printf("[%d]: Relaunch kernel on device %d with new process\n", getpid(), relaunch_dev);
	pid = fork();
	if (pid > 0)
		printf("[%d]: Fork child3 %d\n", getpid(), pid);

	if (pid == 0) {
		printf("[%d]: I'm child3\n", getpid());

		cudaError_t rc3;
		cudaStream_t stm;

		if (prepareGPU(relaunch_dev))
			goto release3;

		void *dev_int3;
		rc3 = cudaHostGetDevicePointer(&dev_int3, &host_int, 0);
		if (rc3 != cudaSuccess) {
			printf("Failed to get device pointer of %p, %s\n", &host_int, cudaGetErrorString(rc3));
			goto release3;
		}

		// Allocate Unified Memory -- accesstmble from CPU or GPU
		cudaMallocManaged(&x, N*sizeof(float));
		cudaMallocManaged(&y, N*sizeof(float));

		rc3 = cudaStreamCreate(&stm);
		if (rc3 != cudaSuccess) {
			printf("[%d]: Failed to create stream, %s, rc3 = %d\n", getpid(), cudaGetErrorString(rc3), rc3);
			goto release3;
		}

		printf("[%d]: Child3 kernel launch\n", getpid());
		//breaker<<< 1, 32, 0, stm>>>();
		//breaker<<< 1, 32, 0, stm>>>((int*)dev_int3);
		//add<<<numBlocks, blockSize, 0, stm>>>(N, x, y);
		add<<<numBlocks, blockSize, 0, stm>>>(N, x, y, (int*)dev_int3);
		rc3 = cudaStreamSynchronize(stm);
		if (rc3 != cudaSuccess) {
			printf("[%d]: Device %d, cudaStreamSynchronize return %d, %s\n", getpid(), relaunch_dev, rc3, cudaGetErrorString(rc3));
			goto release3;
		}
		printf("[%d]: Child3 kernel end\n", getpid());
release3:
		cudaFree(x);
		cudaFree(y);
		releaseGPU(relaunch_dev, stm);
		exit(0);
	}

	while (wait(NULL) > 0);

	return 0;
}
