
#ifndef __BASE_TASK_H__
#define __BASE_TASK_H__

#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

class BaseTask {
	public:
		BaseTask() {};
		~BaseTask() {};

		virtual void taskInit() { 
			std::cout << "taskInit() from base." << std::endl; 
		};
		virtual void taskCallback(int insId, int nIter) { 
			std::cout << "taskInitDevice() from base." << std::endl; 
		};
		virtual void taskFinish() {
			std::cout << "taskFinish() from base." << std::endl;
		};

		virtual void recordPriority(int priority) {};
};

#endif // __BASE_TASK_H__