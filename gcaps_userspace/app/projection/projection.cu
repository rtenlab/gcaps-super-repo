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
#include <random>
#include <vector>

#include <linux/nvgpu.h>
#include "projection.h"
#include <common/include/support.h>
#include <helper_cuda.h>
#include <helper_functions.h>

__global__ void projectPoints(float *d_input, float *d_output, int num_points, float f) {
	__dummy_kernel_prologue(50);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_points) {
        float x = d_input[3 * idx];
        float y = d_input[3 * idx + 1];
        float z = d_input[3 * idx + 2];

		for (int i = 0; i < 1000; i++) {
            x = sinf(x) + cosf(y);
            y = cosf(y) + sinf(x);
            z = tanf(z) + cosf(x);
        }

        // Optionally, you can add more compute-intensive operations here
        // like trigonometric functions to simulate a perspective projection:
        float scale = f / (f + z);
        d_output[2 * idx] = x * scale;
        d_output[2 * idx + 1] = y * scale;
    }
	__dummy_kernel_epilogue();
}

// dummy workload to check prime number
bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;

    if (n % 2 == 0 || n % 3 == 0) return false;

    int i = 5;
    while (i * i <= n) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
        i += 6;
    }
    return true;
}

// Function to compute prime numbers up to a limit
std::vector<int> computePrimes(int limit) {
    std::vector<int> primes;
    for (int i = 2; i <= limit; i++) {
        if (isPrime(i)) {
            primes.push_back(i);
        }
    }
    return primes;
}

Projection::Projection(int num_points_, float focal_length_, int fd_, bool sync_mode_, bool ioctl_enabled_, bool suspension_) {
	num_points = num_points_;
	focal_length = focal_length_;

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

void Projection::taskInit() {
	cuInit(0);
	cuDeviceGet(&device, 0);
	cuCtxCreate(&ctx, 0, device);

	checkCudaErrors(cudaMallocHost((void **)&h_input, sizeof(float) * 3 * num_points));
	checkCudaErrors(cudaMallocHost((void **)&h_output, sizeof(float) * 2 * num_points));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-100.0, 100.0); // Random points in range [-100, 100] for x, y, and z

	for (int i = 0; i < 3 * num_points; i++) {
		h_input[i] = dis(gen);
	}

	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_input), sizeof(float) * 3 * num_points));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_output), sizeof(float) * 2 * num_points));

	if (event_flags != 0) {
        checkCudaErrors(cudaEventCreateWithFlags(&start, event_flags));
        checkCudaErrors(cudaEventCreateWithFlags(&stop, event_flags));
    }else {
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
    }

	checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
}

void Projection::taskCallback(int insId, int nIter) {
	std::vector<int> primesBefore = computePrimes(num_points);  // Adjust the limit as needed

	// Compute the centroid of the input points
	float inputCentroidX = 0.0f;
	float inputCentroidY = 0.0f;
	float inputCentroidZ = 0.0f;

	for (int i = 0; i < num_points; i++) {
		inputCentroidX += h_input[3 * i];
		inputCentroidY += h_input[3 * i + 1];
		inputCentroidZ += h_input[3 * i + 2];
	}

	inputCentroidX /= num_points;
	inputCentroidY /= num_points;
	inputCentroidZ /= num_points;

	// *********************************************
	int pid = getpid();
	int threadsPerBlock = 256;
	int blocksPerGrid = (num_points + threadsPerBlock - 1) / threadsPerBlock;

	gcapsGpuSegBegin(fd, pid, sync_mode, ioctl_enabled);

	checkCudaErrors(cudaMemcpyAsync(d_input, h_input, sizeof(float) * 3 * num_points, cudaMemcpyHostToDevice, stream));

	projectPoints<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, num_points, focal_length);

	__event_record_start(insId, nIter);
	projectPoints<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, num_points, focal_length);
	__event_record_stop(insId, nIter);

	checkCudaErrors(cudaMemcpyAsync(h_output, d_output, sizeof(float) * 2 * num_points, cudaMemcpyDeviceToHost, stream));

	gcapsGpuSegEnd(fd, pid, sync_mode, stream, ioctl_enabled);
	// *********************************************

	std::vector<int> primesAfter = computePrimes(num_points * 2);  // Adjust the limit as needed

	// Compute the centroid of the projected points
	float projectedCentroidX = 0.0f;
	float projectedCentroidY = 0.0f;

	for (int i = 0; i < num_points; i++) {
		projectedCentroidX += h_output[2 * i];
		projectedCentroidY += h_output[2 * i + 1];
	}

	projectedCentroidX /= num_points;
	projectedCentroidY /= num_points;

	// Compute the vector difference between the two centroids
	float diffX = projectedCentroidX - inputCentroidX;
	float diffY = projectedCentroidY - inputCentroidY;
	float diffZ = -inputCentroidZ;  // Since the Z-coordinate of the projected centroid is essentially 0

	// Compute the average of the projected points
	float avgX = 0.0f;
	float avgY = 0.0f;

	for (int i = 0; i < num_points; i++) {
		avgX += h_output[2 * i];
		avgY += h_output[2 * i + 1];
	}

	avgX /= num_points;
	avgY /= num_points;

	// Compute the distance of each point from the average
	float *distances = new float[num_points];

	for (int i = 0; i < num_points; i++) {
		float dx = h_output[2 * i] - avgX;
		float dy = h_output[2 * i + 1] - avgY;
		distances[i] = sqrtf(dx * dx + dy * dy);
	}

	// Optionally, you can compute the average distance or other statistics on the distances array
	float avgDistance = 0.0f;
	for (int i = 0; i < num_points; i++) {
		avgDistance += distances[i];
	}
	avgDistance /= num_points;
}

void Projection::taskFinish() {
	cudaFree(d_input);
	cudaFree(d_output);
	cudaFreeHost(h_output);
	cudaFreeHost(h_input);

	cuCtxDestroy(ctx);
}

Projection::~Projection() {}

void Projection::recordPriority(int priority) {
	this->prio = priority;
}