#ifndef __MMULCPU_H__
#define __MMULCPU_H__

#include "../base_task.h"


class MatrixMulCPU : public BaseTask {
public:
	MatrixMulCPU(int m, int n, int k);
	~MatrixMulCPU();

	void taskInit();
	void taskCallback(int insId, int nIter) override;
	void taskFinish();

	void recordPriority(int priority);

private:
	dim3 dimsA, dimsB, dimsC;
	unsigned int mem_size_A, mem_size_B, mem_size_C;
	float *h_A, *h_B, *h_C;

	int fd;

	int prio;
};
#endif // !__MMULCPU_H__