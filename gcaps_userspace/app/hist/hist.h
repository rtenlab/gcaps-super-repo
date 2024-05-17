#ifndef __HIST_H__
#define __HIST_H__

#include <cuda.h>
#include "../base_task.h"

class Hist : public BaseTask {
public:
	Hist(unsigned int n_elements, unsigned int n_bins, int fd, bool sync_mode, bool ioctl_enabled, bool suspension);
	~Hist();

	void taskInit();
	void taskCallback(int insId, int nIter) override;
	void taskFinish();

	void recordPriority(int priority);

private:
	cudaStream_t stream;
	cudaEvent_t start, stop;
	
	unsigned int* bins_h; // store results
    unsigned int *in_d;
    unsigned int* bins_d;
	unsigned int num_elements, num_bins;
	unsigned int *in_h;

	int fd;
	bool ioctl_enabled;
	bool sync_mode;
	int event_flags = 0;

	int prio;

        CUcontext ctx;
	CUdevice device;
};


#endif // !__HIST_H__