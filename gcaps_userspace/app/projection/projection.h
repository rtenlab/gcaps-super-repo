#ifndef __PROJECTION__
#define __PROJECTION__

#include <cuda.h>
#include "../base_task.h"

class Projection : public BaseTask {
public:
	Projection(int num_points, float focal_length, int fd, bool sync_mode, bool ioctl_enabled, bool suspension);
	~Projection();

	void taskInit();
	void taskCallback(int insId, int nIter) override;
	void taskFinish();
	void recordPriority(int priority);
private:
	cudaStream_t stream;
	cudaEvent_t start, stop;

	float *h_input, *h_output;
	float *d_input, *d_output;

	int num_points;
	float focal_length;

	int fd;
	bool sync_mode;
	bool ioctl_enabled;
	int event_flags = 0;

	int prio;

        CUcontext ctx;
	CUdevice device;
};

#endif // !__PROJECTION__