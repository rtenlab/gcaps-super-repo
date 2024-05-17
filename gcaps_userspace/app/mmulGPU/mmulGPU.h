#ifndef __MMULGPU_H__
#define __MMULGPU_H__

#include <cuda.h>
#include "../base_task.h"

class MatrixMulGPU : public BaseTask {
public:
	MatrixMulGPU(int m, int n, int k, int fd, bool sync_mode, bool ioctl_enabled, bool suspension);
	~MatrixMulGPU();

	void taskInit();
	void taskCallback(int insId, int nIter) override;
	void taskFinish();

	void recordPriority(int priority);

private:
	cudaStream_t stream;
	cudaEvent_t start, stop;
	int block_size = 16;
	dim3 dimsA, dimsB, dimsC;
	dim3 threads, grid;
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	float *h_A, *h_B, *h_C;
	float *d_A, *d_B, *d_C;

	int fd;
	bool sync_mode;
	bool ioctl_enabled;
	int event_flags = 0;

	int prio;

        CUcontext ctx;
	CUdevice device;
};
#endif // !__MATR__MMULGPU_H__IXMUL__