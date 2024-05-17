#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sched.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>

#include <linux/nvgpu.h>
#include "hist.h"
#include <common/include/support.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define BLOCK_SIZE 512

__global__ void histogram_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {
	__dummy_kernel_prologue(50);

	for (unsigned int i = 0; i < 10; i++) {
		unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
		int stride = blockDim.x * gridDim.x;
		for (unsigned int j = tid; j < num_elements; j += stride) {
			// by default, the randomly generated value should be in range (0, 4095)
			int position = input[j];
			if (position >= 0 && position <= num_bins - 1) {
				atomicAdd(&(bins[position]), 1);
			}
		}
	}
	__dummy_kernel_epilogue();
}

Hist::Hist(unsigned int n_elements, unsigned int n_bins, int fd_,  bool sync_mode_, bool ioctl_enabled_, bool suspension_) {
	num_elements = n_elements;
	num_bins = n_bins;
	fd = fd_;
	sync_mode = sync_mode_;

	ioctl_enabled = ioctl_enabled_;
	if (suspension_ == true) {
		event_flags |= cudaEventBlockingSync;
    }else {
        event_flags = cudaEventDisableTiming;
        // event_flags = cudaEventDefault;
    }
}

void Hist::taskInit() {
	cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&ctx, 0, device);

	if (event_flags != 0) {
        checkCudaErrors(cudaEventCreateWithFlags(&start, event_flags));
        checkCudaErrors(cudaEventCreateWithFlags(&stop, event_flags));
    }else {
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
    }
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

	checkCudaErrors(cudaMallocHost((void **)&in_h, num_elements * sizeof(unsigned int)));
	checkCudaErrors(cudaMallocHost((void **)&bins_h, num_bins * sizeof(unsigned int)));
	for (unsigned int i = 0; i < num_elements; i++) {
		in_h[i] = rand() % num_bins;
	}

	checkCudaErrors(cudaMalloc((void**)&in_d, num_elements * sizeof(unsigned int)));
    checkCudaErrors(cudaMalloc((void**)&bins_d, num_bins * sizeof(unsigned int)));
}

void Hist::taskCallback(int insId, int nIter) {
	const unsigned int GRID_SIZE = (int)ceil((float(num_elements)) / BLOCK_SIZE);
	dim3 DimGrid = dim3(GRID_SIZE);
	dim3 DimBlock = dim3(BLOCK_SIZE);

	gcapsGpuSegBegin(fd, getpid(), sync_mode, ioctl_enabled);
	checkCudaErrors(cudaMemcpyAsync(in_d, in_h, num_elements * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
	checkCudaErrors(cudaMemsetAsync(bins_d, 0, num_bins * sizeof(unsigned int), stream));
	
	histogram_kernel <<<DimGrid, DimBlock, 0, stream>>> (in_d, bins_d, num_elements, num_bins);

	__event_record_start(insId, nIter);
	histogram_kernel <<<DimGrid, DimBlock, 0, stream>>> (in_d, bins_d, num_elements, num_bins);
	__event_record_stop(insId, nIter);
	
	checkCudaErrors(cudaMemcpyAsync(bins_h, bins_d, num_bins * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	gcapsGpuSegEnd(fd, getpid(), sync_mode, stream, ioctl_enabled);
}

void Hist::taskFinish() {
	checkCudaErrors(cudaFreeHost(in_h));
	checkCudaErrors(cudaFreeHost(bins_h));
	checkCudaErrors(cudaFree(in_d));
	checkCudaErrors(cudaFree(bins_d));
	cuCtxDestroy(ctx);
}

void Hist::recordPriority(int priority) {
	this->prio = priority;
}