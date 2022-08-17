#include <cuda.h>
#include <nvml.h>
#include <iostream>
#include <unistd.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <vector>
#include <cstdlib>

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
                printf("[%d] [Dev%d]: Failed to get CUdevice, %s\n",
				getpid(), dev_id,
				cuGetErrorString(cu_rc, &cu_err_msg) == CUDA_ERROR_INVALID_VALUE ?
				"unknown error" : cu_err_msg);
                return -1;
        }

        cu_rc = cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev);
        if (cu_rc != CUDA_SUCCESS) {
		printf("[%d] [Dev%d]: Failed to retain primary context, %s\n",
				getpid(), dev_id,
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
		printf("[%d] [Dev%d]: Failed to get CUDA device properties, %s\n", getpid(), dev_id, cudaGetErrorString(rc));
		return -1;
	}

	uuid_str = get_uuid_str((unsigned char *)&devprop.uuid);
	nvmlrc = nvmlDeviceGetHandleByUUID(uuid_str.c_str(), &nvmlDev);
	if (nvmlrc != NVML_SUCCESS) {
		printf("[%d] [Dev%d]: Failed to get NVML Device %s, %s\n", getpid(), dev_id, uuid_str.c_str(), nvmlErrorString(nvmlrc));
		return -1;
	}

	nvmlrc = nvmlDeviceSetComputeMode(nvmlDev, NVML_COMPUTEMODE_EXCLUSIVE_PROCESS);
	if (nvmlrc != NVML_SUCCESS) {
		printf("[%d] [Dev%d]: Failed to set Exclusive Process Compute Mode, %s\n", getpid(), dev_id, nvmlErrorString(nvmlrc));
		return -1;
	}

	nvmlrc = nvmlDeviceSetPersistenceMode(nvmlDev, NVML_FEATURE_ENABLED);
	if (nvmlrc != NVML_SUCCESS) {
		printf("[%d] [Dev%d]: Failed to set Persistence Mode, %s\n", getpid(), dev_id, nvmlErrorString(nvmlrc));
		return -1;
	}
	nvmlShutdown();

	rc = cudaHostRegister(&host_int, sizeof(host_int), cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (rc != cudaSuccess) {
                printf("[%d] [Dev%d]: Failed to register host_int %p, %s\n", getpid(), dev_id, &host_int, cudaGetErrorString(rc));
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
	printf("[%d] [Dev%d]: Released\n", getpid(), dev_id);
}

int main(int argc, char *argv[])
{
	int N = 1<<12;
	float *x, *y;
	// Launch kernel on 1M elements on the GPU
	int blockSize = 256;
	int numBlocks = (N + blockSize - 1) / blockSize;

	char *end;
	int dev_id = 0;
	int break_me = 0;

	if (argc >= 2)
		dev_id = strtol(argv[argc-2], &end, 10);

	if (argc >= 3)
		break_me = strtol(argv[argc-1], &end, 10);

	printf("[%d]: Using device %d, break_me = %d\n", getpid(), dev_id, break_me);

	int ret = 0;
	cudaError_t rc;
	cudaStream_t stm;

	if (prepareGPU(dev_id)) {
		ret = 1;
		goto release1;
	}

	void *dev_int;
	rc = cudaHostGetDevicePointer(&dev_int, &host_int, 0);
	if (rc != cudaSuccess) {
		printf("[%d] [Dev%d]: Failed to get device pointer of %p, %s\n", getpid(), dev_id, &host_int, cudaGetErrorString(rc));
		ret = 1;
		goto release1;
	}

	if (!break_me) {
		// Allocate Unified Memory -- accesstmble from CPU or GPU
		cudaMallocManaged(&x, N*sizeof(float));
		cudaMallocManaged(&y, N*sizeof(float));
	}

	rc = cudaStreamCreate(&stm);
	if (rc != cudaSuccess) {
		printf("[%d] [Dev%d]: Failed to create stream, %s, rc = %d\n", getpid(), dev_id, cudaGetErrorString(rc), rc);
		ret = 1;
		goto release1;
	}

	printf("[%d] [Dev%d]: Kernel launch\n", getpid(), dev_id);
	if (break_me)
		breaker<<< 1, 32, 0, stm>>>((int*)dev_int);
	else
		add<<<numBlocks, blockSize, 0, stm>>>(N, x, y, (int*)dev_int);

	rc = cudaStreamSynchronize(stm);
	if (rc != cudaSuccess) {
		printf("[%d] [Dev%d]: cudaStreamSynchronize return %d, %s\n", getpid(), dev_id, rc, cudaGetErrorString(rc));
		ret = 2;
		goto release1;
	}
	printf("[%d] [Dev%d]: Kernel end\n", getpid(), dev_id);
release1:
	if (!break_me) {
		cudaFree(x);
		cudaFree(y);
	}
	releaseGPU(dev_id, stm);

	return ret;
}
