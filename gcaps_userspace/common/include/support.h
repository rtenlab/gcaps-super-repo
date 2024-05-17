#ifndef __SUPPORT_H__
#define __SUPPORT_H__

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cuda.h>

#include "config.h"

extern int task_identity;

struct mytimer_t {
	std::vector<float> time_list;
	std::chrono::time_point<std::chrono::system_clock> start_time;
	std::chrono::time_point<std::chrono::system_clock> stop_time;
	float elapsed_time;

	void init() {
		time_list.reserve(2000);
	}

	// we record a task's release time
	void record_start(std::chrono::time_point<std::chrono::system_clock> input) {
		start_time = input;
	}

	void record_stop() {
		stop_time = std::chrono::system_clock::now();
		elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();
		time_list.push_back(elapsed_time);
	}

	/* max, min, average */
	void print_result() {
		int skip_num = 0;	// skip the the number of kernels as warm-up kernels
		float max_val = *std::max_element(time_list.begin() + skip_num, time_list.end());
		float min_val = *std::min_element(time_list.begin() + skip_num, time_list.end());
		float mean_val = std::accumulate(time_list.begin() + skip_num, time_list.end(), 0.0) / time_list.size();

		float perc95;
		double index = 0.95 * (time_list.size() - 1);
		size_t lower = static_cast<size_t>(index);
		size_t upper = lower + 1;
		perc95 = (upper >= time_list.size()) ? time_list[lower] : time_list[lower] + (index - lower) * (time_list[upper] - time_list[lower]);

		printf("%f, %f, %f, %f\n", min_val/1000, mean_val/1000, perc95/1000, max_val/1000);

		// save the full timelist to csv/txt
		auto now = std::chrono::system_clock::now();
		std::time_t now_c = std::chrono::system_clock::to_time_t(now);
		struct tm* parts = std::localtime(&now_c);

		std::ostringstream oss;
		// Format the date
		oss << std::setw(2) << std::setfill('0') << (parts->tm_mon + 1);  // Month (0-11, so add 1)
		oss << std::setw(2) << std::setfill('0') << parts->tm_mday;       // Day of the month
		oss << (1900 + parts->tm_year);                                   // Year since 1900, so add 1900
		std::ofstream outFile("timelog/task" + std::to_string(task_identity) + "-" + oss.str() + ".txt");
		if (!outFile) {
			std::cerr << "Error opening file for writing!" << std::endl;
		}
		for (float value : time_list) {
			outFile << value << std::endl;
		}
		outFile.close();
	}
};

#define cudaSafeCall(x) do { \
	cudaError_t _cuda_err; \
    if ((_cuda_err = (cudaError_t)(x)) != cudaSuccess) { \
        printf("line %s CUDA error %d! %s\n", #x, _cuda_err, cudaGetErrorString(_cuda_err)); \
        return -1; \
    } \
} while (0);

#define CUSafeCall(x) do { \
	CUresult _cu_result; \
    if ((_cu_result = (CUresult)(x)) != CUDA_SUCCESS) { \
        printf("line %s CUDA error %d!\n", #x, _cu_result); \
        return -1; \
    } \
} while (0);


extern mytimer_t mytimer;
extern mytimer_t overhead_timer;

#define gcapsGpuSegBegin(_fd, _pid, _sync_mode, ioctl_enabled_) do { \
	if (ioctl_enabled == true) { \
		cudaEventRecord(start, stream); \
		struct nvgpu_gpu_runlist_update_rt_prio_args _ioctl_args; \
		_ioctl_args.pid = _pid; \
		_ioctl_args.add_req = true; \
		_ioctl_args.sync_mode = _sync_mode; \
		/* overhead_timer.record_start(); \ */ \
		int _err = ioctl(_fd, NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO, &_ioctl_args); \
		/* overhead_timer.record_stop(); \ */ \
		assert(_err >= 0); \
	} else { \
		cudaEventRecord(start, stream); \
	} \
} while(0);

#define gcapsGpuSegEnd(_fd, _pid, _sync_mode, stream, ioctl_enabled_) do { \
	if (ioctl_enabled == true) { \
		cudaEventRecord(stop, stream); \
		cudaEventSynchronize(stop); \
		struct nvgpu_gpu_runlist_update_rt_prio_args _ioctl_args; \
		_ioctl_args.pid = _pid; \
		_ioctl_args.add_req = false; \
		_ioctl_args.sync_mode = _sync_mode; \
		/* overhead_timer.record_start(); \ */\
		int _err = ioctl(_fd, NVGPU_GPU_IOCTL_RUNLIST_UPDATE_RT_PRIO, &_ioctl_args); \
		/* overhead_timer.record_stop(); \ */\
		assert(_err >= 0); \
	} else { \
		cudaEventRecord(stop, stream); \
		cudaEventSynchronize(stop); \
	} \
} while(0);

/* 
 * The macros used for tsg ctx sw measurements
 */
/* 
 * The dummy loop to extend kernel duration
 */

#ifdef CONFIG_TSG_SW
#define __dummy_kernel_prologue(_num) \
	 for (volatile int dummy = 0; dummy < _num; ++dummy) { 

#define __dummy_kernel_epilogue() \
	}

#define __event_record_start(insId, nIter) \
	cudaStreamSynchronize(stream); \
	cudaEvent_t event_kernel_start, event_kernel_stop; \
	cudaEventCreate(&event_kernel_start); \
	cudaEventCreate(&event_kernel_stop); \
	cudaEventRecord(event_kernel_start, stream); \
	for (int __iter = 0; __iter < nIter; __iter++) { 


#define __event_record_stop(insId, nIter) \
	} \
	float msecTotal = 0;	\
	cudaEventRecord(event_kernel_stop, stream);	\
	cudaEventSynchronize(event_kernel_stop);	\
	cudaEventElapsedTime(&msecTotal, event_kernel_start, event_kernel_stop); \
	if (insId == 0 && nIter > 0) { std::cout << msecTotal / nIter << std::endl; }


#else
#define __dummy_kernel_prologue(_num) \
	for (volatile int dummy = 0; dummy < 1; ++dummy) { \

#define __dummy_kernel_epilogue() \
	} 
	
#define __event_record_start(insId, nIter) \
	for (int __iter = 0; __iter < 0; __iter++) { 

#define __event_record_stop(insId, nIter) \
	}
#endif

#endif // __SUPPORT_H__