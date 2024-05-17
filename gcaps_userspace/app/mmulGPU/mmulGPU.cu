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
#include "mmulGPU.h"
#include <common/include/support.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#define _GNU_SOURCE

#define BLOCK_SIZE 16


/**( * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int wA, int wB) {
    __dummy_kernel_prologue(50);

    for (unsigned int i = 0; i < 1; i++) {
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        int aBegin = wA * BLOCK_SIZE * by;
        int aEnd   = aBegin + wA - 1;
        int aStep  = BLOCK_SIZE;
        int bBegin = BLOCK_SIZE * bx;
        int bStep  = BLOCK_SIZE * wB;

        float Csub = 0;
        for (int a = aBegin, b = bBegin; a <= aEnd; a += aStep, b += bStep) {
            __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
            __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
            As[ty][tx] = A[a + wA * ty + tx];
            Bs[ty][tx] = B[b + wB * ty + tx];

            __syncthreads();

    #pragma unroll

            for (int k = 0; k < BLOCK_SIZE; ++k) {
                Csub += As[ty][k] * Bs[k][tx];
            }
            __syncthreads();
        }
        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
        C[c + wB * ty + tx] = Csub;
    }
    __dummy_kernel_epilogue();
}


MatrixMulGPU::MatrixMulGPU(int m, int n, int k, int fd_, bool sync_mode_, bool ioctl_enabled_, bool suspension_) {
	dimsA = dim3(5 * 2 * block_size, 5 * 2 * block_size, 1);
    dimsB = dim3(5 * 4 * block_size, 5 * 2 * block_size, 1);
	dimsA.x = k;
    dimsA.y = n;
    dimsB.x = m;
    dimsB.y = k;
	// printf("MatrixMulGPU:x MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
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

void ConstantInit(float *data, int size, float val) {
    for (int i = 0; i < size; ++i) {
        data[i] = val;
    }
}

MatrixMulGPU::~MatrixMulGPU() {}

void MatrixMulGPU::taskInit() {
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&ctx, 0, device);

	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int size_B = dimsB.x * dimsB.y;
	mem_size_A = sizeof(float) * size_A;
	mem_size_B = sizeof(float) * size_B;
    checkCudaErrors(cudaMallocHost((void **)&h_A, mem_size_A));
    checkCudaErrors(cudaMallocHost((void **)&h_B, mem_size_B));

	dimsC = dim3(dimsB.x, dimsA.y, 1);
	mem_size_C = sizeof(float) * dimsC.x * dimsC.y;
    checkCudaErrors(cudaMallocHost((void **)&h_C, mem_size_C));

    ConstantInit(h_A, size_A, 1.0f);
    ConstantInit(h_B, size_B, 0.01f);

	threads = dim3(block_size, block_size);
    grid = dim3(dimsB.x / threads.x, dimsA.y / threads.y);

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C));

	if (event_flags != 0) {
        checkCudaErrors(cudaEventCreateWithFlags(&start, event_flags));
        checkCudaErrors(cudaEventCreateWithFlags(&stop, event_flags));
    }else {
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
    }

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}


void MatrixMulGPU::taskCallback(int insId, int nIter) {
    int pid = getpid();

    gcapsGpuSegBegin(fd, pid, sync_mode, ioctl_enabled); // add mem operations
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream));
   
    // regular kernel for main.cc; warm up kernel for tsg ctx experiment
    MatrixMulCUDA<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x);

    __event_record_start(insId, nIter);

    MatrixMulCUDA<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, dimsA.x, dimsB.x); // will not trigger 
    __event_record_stop(insId, nIter);

	checkCudaErrors(cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream));    
    
    gcapsGpuSegEnd(fd, pid, sync_mode, stream, ioctl_enabled);
}

void MatrixMulGPU::taskFinish() {
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    cuCtxDestroy(ctx);
}

void MatrixMulGPU::recordPriority(int priority) {
	this->prio = priority;
}