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
#include "mmulCPU.h"
#include <common/include/support.h>

#define _GNU_SOURCE

MatrixMulCPU::MatrixMulCPU(int m, int n, int k) {
	dimsA.x = k;
    dimsA.y = n;
    dimsB.x = m;
    dimsB.y = k;
	// printf("MatrixMulCPU: MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
}

void MatrixMulCPU::taskInit() {
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int size_B = dimsB.x * dimsB.y;
	mem_size_A = sizeof(float) * size_A;
	mem_size_B = sizeof(float) * size_B;
    h_A = reinterpret_cast<float *>(malloc(mem_size_A));
    h_B = reinterpret_cast<float *>(malloc(mem_size_B));

    for (unsigned int i=0; i < size_A; i++) { h_A[i] = (rand()%100)/100.00; }
    for (unsigned int i=0; i < size_B; i++) { h_B[i] = (rand()%100)/100.00; }

	dimsC = dim3(dimsB.x, dimsA.y, 1);
	mem_size_C = sizeof(float) * dimsC.x * dimsC.y;
	h_C = reinterpret_cast<float *>(malloc(mem_size_C));
}

void MatrixMulCPU::taskCallback(int insId, int nIter) {
	const float valB = 0.01f;
    int m = dimsB.x;
    int n = dimsA.y;
    int k = dimsA.x;

    for(int row = 0; row < m; ++row) {
        for(int col = 0; col < n; ++col) {
            float sum = 0;
            for(unsigned int i = 0; i < k; ++i) {
                sum += h_A[row*k + i] * h_B[i*n + col];
            }
            h_C[row*n + col] = sum;
        }
      }
}

void MatrixMulCPU::taskFinish() {
	free(h_A);
    free(h_B);
    free(h_C);
}

void MatrixMulCPU::recordPriority(int priority) {
	this->prio = priority;
}