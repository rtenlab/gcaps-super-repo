#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/nvgpu.h>
#include <stdio.h>
#include <cuda.h>
#include <assert.h>
#include <string>
#include <sstream>
#include <chrono>
#include <sys/wait.h>
#include <thread>
#include <fstream>

// CUDA runtime
#include <cuda_runtime.h>

#include "common/include/support.h"
#include "app/base_task.h"

#include "app/mmulGPU/mmulGPU.h"
#include "app/mmulCPU/mmulCPU.h"
#include "app/hist/hist.h"
#include "app/dxtc/dxtc.h"
#include "app/projection/projection.h"


mytimer_t mytimer;
int task_identity;
std::chrono::time_point<std::chrono::system_clock> init_start_time;

mytimer_t overhead_timer;

struct Task {
	int id;
	int workload_id;
	int period; // ms
	int cpu_id;
	int prio;
	float util; // 0 -100
};

struct Cpu {
	int id;
	float util;
};

bool util_compare(Cpu c1, Cpu c2) {
	return c1.util < c2.util;
}

bool RM_prio_compare(Task t1, Task t2) {
	if (t1.prio == 0 && t2.prio != 0) {
        return false; // task1 should be placed after task2
    } else if (t1.prio != 0 && t2.prio == 0) {
        return true; // task1 should be placed before task2
    } else {
        return t1.period < t2.period; // sort based on period
    }
}

std::vector<int> read_csv_column(std::string fn, int cid) {
	std::vector<int> ret;
	std::fstream fin;
	fin.open(fn, std::ios::in);
	std::string line;

	std::string word;
	int row = 0;
	while (std::getline(fin, line)) {
		std::stringstream s(line);
		// std::cout << line << std::endl;
		row++;
		if (row == 1) 
			continue; // first row is title

		int index = 0;
		while (std::getline(s, word, ',')) {
			if (index == cid) {
				ret.push_back(std::stof(word));
			}
			index++;
		}
	}

	assert(!ret.empty());
	return ret;
}

void loadTaskset(std::vector<Task> &tasks, std::string filename) {
	if (filename.empty()) {
		printf("filename is empty! Skipped!\n");
		return;
	}

	std::vector<int> taskid_list = read_csv_column(filename, 0);
	std::vector<int> workload_list = read_csv_column(filename, 1);
	std::vector<int> period_list = read_csv_column(filename, 2);
	std::vector<int> wcet_list = read_csv_column(filename, 3);

	std::vector<float> util_list;
	for (int i = 0; i < 8; i++) {
		util_list.push_back(wcet_list[i] * 100 / float(period_list[i]));
	}
	
	for (int i = 0; i < taskid_list.size(); i++) {
		Task task;
		task.id = taskid_list[i];
		if (task.id == 6) { // the last task as BE tasks
			task.prio = 0;
		}else {
			task.prio = 1;
		}
		task.workload_id = workload_list[i];
		task.period = period_list[i];
		task.cpu_id = 0;
		task.util = util_list[i];
		tasks.push_back(task);
	}

	int ncpu = 2;
	std::vector<Cpu> cpus;
	for (int i = 0; i < ncpu; i++) {
		Cpu cpu;
		cpu.id = i + 1; // kthread pinned to 3
		cpu.util = 0;
		cpus.push_back(cpu);
	}
	for (int i = 0; i < tasks.size(); i++) {
		if (tasks[i].id == 6) {
			tasks[i].cpu_id = 5;
		}else {
			// sort with WFD, low to high util
			sort(cpus.begin(), cpus.end(), util_compare);
			cpus[0].util += tasks[i].util;
			tasks[i].cpu_id = cpus[0].id;
		}
		printf("task %d (util=%.2f) is assigned to cpu %d (util=%.2f)\n", tasks[i].id, tasks[i].util, tasks[i].cpu_id, cpus[0].util);
	}

	int base_prio = 70; 
	sort(tasks.begin(), tasks.end(), RM_prio_compare);
	for (int i = 0; i < tasks.size(); i++) {
		if (tasks[i].prio == 0) { continue; }
		tasks[i].prio = base_prio - i;
		std::cout << "task id: " << tasks[i].id << ", workload id: " << tasks[i].workload_id << " prio: " << tasks[i].prio << std::endl;
	}
}


void prog_task(Task task, int duration, int fd, bool sync_mode, int prioritized, bool ioctl_enabled, bool suspension) {
	const int sleep_time1 = 2500;
	const int sleep_time2 = 3000;

	int wid = task.workload_id;
	int period = task.period;
	int cpuid = task.cpu_id;
	int priority = task.prio;

	// we run it for <duration> secs
	int nIter = duration * 1000 / period;
	// int nIter = 30;

	task_identity = task.id;

	BaseTask *workload;

	if (wid == 0) { workload = new MatrixMulCPU(128, 128, 160); } 
	else if (wid == 1) { workload = new MatrixMulCPU(128, 128, 512); } 
	else if (wid == 2) { workload = new MatrixMulGPU(1024, 1024, 1024, fd, sync_mode, ioctl_enabled, suspension); }
	else if (wid == 3) { workload = new MatrixMulGPU(2048, 2048, 1024, fd, sync_mode, ioctl_enabled, suspension); }
	else if (wid == 4) { workload = new Hist(1600000, 4096, fd, sync_mode, ioctl_enabled, suspension); }
	else if (wid == 5) { workload = new Dxtc("app/dxtc/data/lena-orig.ppm", fd, sync_mode, ioctl_enabled, suspension); }
	else if (wid == 8) { workload = new Projection(50000, 500, fd, sync_mode, ioctl_enabled, suspension); }

	workload->taskInit();
	printf("Task %d taskInit done.\n", task.id);

	mytimer.init();

	std::this_thread::sleep_until(init_start_time + std::chrono::milliseconds(sleep_time1));

	cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpuid, &set);
    sched_setaffinity(gettid(), sizeof(cpu_set_t), &set);

	if (task.prio > 0 && prioritized == 1) {
		struct sched_param params;
		params.sched_priority = priority;
		int ret = sched_setscheduler(0, SCHED_FIFO, &params); // real-time
		if (ret != 0) {
			perror("sched_setscheduler");
			exit(EXIT_FAILURE);
		}
	}else if (task.prio == 0 && prioritized == 1) {
		struct sched_param params;
		params.sched_priority = priority;
		int ret = sched_setscheduler(0, SCHED_OTHER, &params); // default round-robin
		if (ret != 0) {
			perror("sched_setscheduler");
			exit(EXIT_FAILURE);
		}
	}
	workload->recordPriority(task.prio);
	

	// another barrier
	std::this_thread::sleep_until(init_start_time + std::chrono::milliseconds(sleep_time2));
	printf("Task %d sleeping until time2 done.\n", task.id);

	auto start_time = std::chrono::system_clock::now();

	auto next_time = init_start_time + std::chrono::milliseconds(sleep_time2);

	bool running = false;
	while (nIter > 0) {
		std::this_thread::sleep_until(next_time);
		while (running == true);
		running = true;
		mytimer.record_start(next_time); 
		// printf("[%d:%d:%d] iteration # %d start.\n", getpid(), task.id, task.prio, nIter);
		workload->taskCallback(0, 0);
		mytimer.record_stop();
		running = false;
		// printf("[%d:%d:%d] iteration # %d finished, elapsed: %f.\n", getpid(), task.id, task.prio, nIter, mytimer.elapsed_time);
		auto curr_time = std::chrono::system_clock::now();
		next_time += std::chrono::milliseconds(period);
		if (curr_time > next_time) {
			printf("Task %d job %d missed deadline!\n", task.id, nIter);
		}
		nIter--;
	}

	std::chrono::time_point<std::chrono::system_clock> exec_stop_time = std::chrono::system_clock::now();
	std::this_thread::sleep_until(exec_stop_time + std::chrono::milliseconds(1000));

	workload->taskFinish();
}

int main(int argc, char **argv) {
	int opt;
	std::string filename;
	int duration = 0;
	int ioctl_enabled = 0;
	int suspension = 0;
	int sync_mode = 0;
	while ((opt = getopt(argc, argv, "f:d:p:i:s:r:b:")) != EOF) {
		switch (opt) {
			case 'f':
				filename.assign(optarg);
				break;
			case 'd':
				duration = atoi(optarg);
				break;
			case 'i':
				ioctl_enabled = atoi(optarg);
				break;
			case 's':
				suspension = atoi(optarg);
				break;
			case 'b':
				sync_mode = atoi(optarg);
				break;
			case '?':
				std::cerr << "Invalid option: " << opt << std::endl;
			default:
				std::cout << std::endl;
				abort();
		}
	}

	std::cout << "Program configurations:\n"
		<< "taskset: " << filename << "\n"
		<< "duration: " << duration << "\n"
		<< "ioctl enabled: " << bool(ioctl_enabled) << "\n"
		<< "suspension: " << bool(suspension) << "\n"
		<< "sync mode: " << bool(sync_mode) << "\n";
	std::cout << "---------------------------------------" << std::endl;

	if (bool(sync_mode) == 1 && bool(ioctl_enabled) == 1) {
		std::cerr << "IOCTL mode and Synchronization mode should not be set at the same time! Exiting..." << std::endl;
		exit(1);
	}

	std::vector<Task> tasks;
	loadTaskset(tasks, filename);

	int fd = open("/dev/nvgpu/igpu0/ctrl", O_RDWR);
	assert(fd >= 0);

	init_start_time = std::chrono::system_clock::now();
	std::vector<int> childPIDs;
	childPIDs.reserve(10);
	
	int idx = 1;
	for (auto task : tasks) {
		int next_pid;
		if (idx != 0) {
			next_pid = childPIDs[idx-1];
		}

		int pid = fork();
		if (pid == 0) {
			prog_task(task, duration, fd, bool(sync_mode), 1, bool(ioctl_enabled), bool(suspension));
			
			if (task.id != 1) {
				int status = 0;
				waitpid(next_pid, &status, 0);
			}
			printf("[%d:%d], ", getpid(), task.id); mytimer.print_result();
			// overhead_timer.print_result();
			return 0;
		}else {
			childPIDs.push_back(pid); // Store the PID of the child process
		}
		idx++;
	}

    int status = 0;
    while ((wait(&status)) > 0);

	return 0;
}