#include <sys/types.h>
#include <sys/fcntl.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include <signal.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <linux/nvgpu.h>
#include <sched.h>
#include <chrono>

#include "dxtc.h"
#include <common/include/support.h>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#include <helper_functions.h>
#include <helper_cuda.h>

#include <helper_math.h>
#include <float.h> // for FLT_MAX

#include "CudaMath.h"
#include "dds.h"
#include "permutations.h"

#define NUM_THREADS 64        // Number of threads per block.

namespace cg = cooperative_groups;

template <class T>
__device__ inline void swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

//__constant__ float3 kColorMetric = { 0.2126f, 0.7152f, 0.0722f };
__constant__ float3 kColorMetric = { 1.0f, 1.0f, 1.0f };

////////////////////////////////////////////////////////////////////////////////
// Sort colors
////////////////////////////////////////////////////////////////////////////////
__device__ void sortColors(const float *values, int *ranks, cg::thread_group tile)
{
    const int tid = threadIdx.x;

    int rank = 0;

#pragma unroll

    for (int i = 0; i < 16; i++)
    {
        rank += (values[i] < values[tid]);
    }

    ranks[tid] = rank;

    cg::sync(tile);

    // Resolve elements with the same index.
    for (int i = 0; i < 15; i++)
    {
        if (tid > i && ranks[tid] == ranks[i])
        {
            ++ranks[tid];
        }
        cg::sync(tile);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Load color block to shared mem
////////////////////////////////////////////////////////////////////////////////
__device__ void loadColorBlock(const uint *image, float3 colors[16], float3 sums[16], int xrefs[16], int blockOffset, cg::thread_block cta)
{
    const int bid = blockIdx.x + blockOffset;
    const int idx = threadIdx.x;

    __shared__ float dps[16];

    float3 tmp;

    cg::thread_group tile = cg::tiled_partition(cta, 16);

    if (idx < 16)
    {
        // Read color and copy to shared mem.
        uint c = image[(bid) * 16 + idx];

        colors[idx].x = ((c >> 0) & 0xFF) * (1.0f / 255.0f);
        colors[idx].y = ((c >> 8) & 0xFF) * (1.0f / 255.0f);
        colors[idx].z = ((c >> 16) & 0xFF) * (1.0f / 255.0f);

        cg::sync(tile);
        // Sort colors along the best fit line.
        colorSums(colors, sums, tile);

        cg::sync(tile);

        float3 axis = bestFitLine(colors, sums[0], tile);

        cg::sync(tile);

        dps[idx] = dot(colors[idx], axis);

        cg::sync(tile);

        sortColors(dps, xrefs, tile);

        cg::sync(tile);

        tmp = colors[idx];

        cg::sync(tile);

        colors[xrefs[idx]] = tmp;
    }
}
////////////////////////////////////////////////////////////////////////////////
// Round color to RGB565 and expand
////////////////////////////////////////////////////////////////////////////////
inline __device__ float3 roundAndExpand(float3 v, ushort *w)
{
    v.x = rintf(__saturatef(v.x) * 31.0f);
    v.y = rintf(__saturatef(v.y) * 63.0f);
    v.z = rintf(__saturatef(v.z) * 31.0f);

    *w = ((ushort)v.x << 11) | ((ushort)v.y << 5) | (ushort)v.z;
    v.x *= 0.03227752766457f; // approximate integer bit expansion.
    v.y *= 0.01583151765563f;
    v.z *= 0.03227752766457f;
    return v;
}


__constant__ float alphaTable4[4] = { 9.0f, 0.0f, 6.0f, 3.0f };
__constant__ float alphaTable3[4] = { 4.0f, 0.0f, 2.0f, 2.0f };
__constant__ const int prods4[4] = { 0x090000,0x000900,0x040102,0x010402 };
__constant__ const int prods3[4] = { 0x040000,0x000400,0x040101,0x010401 };

#define USE_TABLES 1

////////////////////////////////////////////////////////////////////////////////
// Evaluate permutations
////////////////////////////////////////////////////////////////////////////////
static __device__ float evalPermutation4(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
    // Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable4[bits & 3] * colors[i];
        akku += prods4[bits & 3];
    }

    float alpha2_sum = float(akku >> 16);
    float beta2_sum = float((akku >> 8) & 0xff);
    float alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum = (9.0f * color_sum) - alphax_sum;
#else
    float alpha2_sum = 0.0f;
    float beta2_sum = 0.0f;
    float alphabeta_sum = 0.0f;
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        float beta = (bits & 1);

        if (bits & 2)
        {
            beta = (1 + beta) * (1.0f / 3.0f);
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    // alpha2, beta2, alphabeta and factor could be precomputed for each permutation, but it's faster to recompute them.
    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.111111111111f) * dot(e, kColorMetric);
}

static __device__ float evalPermutation3(const float3 *colors, uint permutation, ushort *start, ushort *end, float3 color_sum)
{
    // Compute endpoints using least squares.
#if USE_TABLES
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    int akku = 0;

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        alphax_sum += alphaTable3[bits & 3] * colors[i];
        akku += prods3[bits & 3];
    }

    float alpha2_sum = float(akku >> 16);
    float beta2_sum = float((akku >> 8) & 0xff);
    float alphabeta_sum = float((akku >> 0) & 0xff);
    float3 betax_sum = (4.0f * color_sum) - alphax_sum;
#else
    float alpha2_sum = 0.0f;
    float beta2_sum = 0.0f;
    float alphabeta_sum = 0.0f;
    float3 alphax_sum = make_float3(0.0f, 0.0f, 0.0f);

    // Compute alpha & beta for this permutation.
    for (int i = 0; i < 16; i++)
    {
        const uint bits = permutation >> (2*i);

        float beta = (bits & 1);

        if (bits & 2)
        {
            beta = 0.5f;
        }

        float alpha = 1.0f - beta;

        alpha2_sum += alpha * alpha;
        beta2_sum += beta * beta;
        alphabeta_sum += alpha * beta;
        alphax_sum += alpha * colors[i];
    }

    float3 betax_sum = color_sum - alphax_sum;
#endif

    const float factor = 1.0f / (alpha2_sum * beta2_sum - alphabeta_sum * alphabeta_sum);

    float3 a = (alphax_sum * beta2_sum - betax_sum * alphabeta_sum) * factor;
    float3 b = (betax_sum * alpha2_sum - alphax_sum * alphabeta_sum) * factor;

    // Round a, b to the closest 5-6-5 color and expand...
    a = roundAndExpand(a, start);
    b = roundAndExpand(b, end);

    // compute the error
    float3 e = a * a * alpha2_sum + b * b * beta2_sum + 2.0f * (a * b * alphabeta_sum - a * alphax_sum - b * betax_sum);

    return (0.25f) * dot(e, kColorMetric);
}

__device__ void evalAllPermutations(const float3 *colors, const uint *permutations, ushort &bestStart, ushort &bestEnd, uint &bestPermutation, float *errors, float3 color_sum, cg::thread_block cta)
{
    const int idx = threadIdx.x;

    float bestError = FLT_MAX;

    __shared__ uint s_permutations[160];

    for (int i = 0; i < 16; i++)
    {
        int pidx = idx + NUM_THREADS * i;

        if (pidx >= 992)
        {
            break;
        }

        ushort start, end;
        uint permutation = permutations[pidx];

        if (pidx < 160)
        {
            s_permutations[pidx] = permutation;
        }

        float error = evalPermutation4(colors, permutation, &start, &end, color_sum);

        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;
        }
    }

    if (bestStart < bestEnd)
    {
        swap(bestEnd, bestStart);
        bestPermutation ^= 0x55555555;    // Flip indices.
    }

    cg::sync(cta); // Sync here to ensure s_permutations is valid going forward

    for (int i = 0; i < 3; i++)
    {
        int pidx = idx + NUM_THREADS * i;

        if (pidx >= 160)
        {
            break;
        }

        ushort start, end;
        uint permutation = s_permutations[pidx];
        float error = evalPermutation3(colors, permutation, &start, &end, color_sum);

        if (error < bestError)
        {
            bestError = error;
            bestPermutation = permutation;
            bestStart = start;
            bestEnd = end;

            if (bestStart > bestEnd)
            {
                swap(bestEnd, bestStart);
                bestPermutation ^= (~bestPermutation >> 1) & 0x55555555;    // Flip indices.
            }
        }
    }

    errors[idx] = bestError;
}

////////////////////////////////////////////////////////////////////////////////
// Find index with minimum error
////////////////////////////////////////////////////////////////////////////////
__device__ int findMinError(float *errors, cg::thread_block cta)
{
    const int idx = threadIdx.x;
    __shared__ int indices[NUM_THREADS];
    indices[idx] = idx;

    cg::sync(cta);

    for (int d = NUM_THREADS/2; d > 0; d >>= 1)
    {
        float err0 = errors[idx];
        float err1 = (idx + d) < NUM_THREADS ? errors[idx + d] : FLT_MAX;
        int index1 = (idx + d) < NUM_THREADS ? indices[idx + d] : 0;

        cg::sync(cta);

        if (err1 < err0)
        {
            errors[idx] = err1;
            indices[idx] = index1;
        }

        cg::sync(cta);
    }

    return indices[0];
}

////////////////////////////////////////////////////////////////////////////////
// Save DXT block
////////////////////////////////////////////////////////////////////////////////
__device__ void saveBlockDXT1(ushort start, ushort end, uint permutation, int xrefs[16], uint2 *result, int blockOffset)
{
    const int bid = blockIdx.x + blockOffset;

    if (start == end)
    {
        permutation = 0;
    }

    // Reorder permutation.
    uint indices = 0;

    for (int i = 0; i < 16; i++)
    {
        int ref = xrefs[i];
        indices |= ((permutation >> (2 * ref)) & 3) << (2 * i);
    }

    // Write endpoints.
    result[bid].x = (end << 16) | start;

    // Write palette indices.
    result[bid].y = indices;
}

__global__ void compress(const uint *permutations, const uint *image, uint2 *result, int blockOffset)
{
    __dummy_kernel_prologue(50);
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();

    const int idx = threadIdx.x;

    __shared__ float3 colors[16];
    __shared__ float3 sums[16];
    __shared__ int xrefs[16];

    loadColorBlock(image, colors, sums, xrefs, blockOffset, cta);

    cg::sync(cta);

    ushort bestStart, bestEnd;
    uint bestPermutation;

    __shared__ float errors[NUM_THREADS];

    evalAllPermutations(colors, permutations, bestStart, bestEnd, bestPermutation, errors, sums[0], cta);

    // Use a parallel reduction to find minimum error.
    const int minIdx = findMinError(errors, cta);

    cg::sync(cta);

    // Only write the result of the winner thread.
    if (idx == minIdx) {
        saveBlockDXT1(bestStart, bestEnd, bestPermutation, xrefs, result, blockOffset);
    }

    __dummy_kernel_epilogue();
}

#define ERROR_THRESHOLD 0.02f

// #define __debugsync()

/* 
    Take the input arguments for this class
*/
Dxtc::Dxtc(std::string fn, int fd_, bool sync_mode_, bool ioctl_enabled_, bool suspension_) {
    input_image = fn;
    const char *image_path = input_image.c_str();
    if (!sdkLoadPPM4ub(image_path, &data, &w, &h))
        fprintf(stderr, "Error, unable to open source image file <%s>\n", image_path);

    memSize = w * h * 4;
    compressedSize = (w / 4) * (h / 4) * 8;

    fd = fd_;
    sync_mode = sync_mode_;
    ioctl_enabled = ioctl_enabled_;
    if (suspension_ == true) {
		event_flags |= cudaEventBlockingSync;
    }else {
        event_flags = cudaEventDisableTiming;
        // event_flags = cudaEventDefault;
    }
};

/*  */
Dxtc::~Dxtc() {};

/* 
    Init: 
    1, Malloc (host, device)
    2. definition of variables if necessary
*/
void Dxtc::taskInit() {
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&ctx, 0, device);

	if (event_flags != 0) {
        checkCudaErrors(cudaEventCreateWithFlags(&start, event_flags));
        checkCudaErrors(cudaEventCreateWithFlags(&stop, event_flags));
    }else {
        checkCudaErrors(cudaEventCreate(&start));
        checkCudaErrors(cudaEventCreate(&stop));
    }
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));


    // Allocate input image.
    // block_image = (unsigned int *)malloc(memSize);
    checkCudaErrors(cudaMallocHost((void **)&block_image, memSize));
    // Convert linear image to block linear.
    for (int by = 0; by < h/4; by++) {
        for (int bx = 0; bx < w/4; bx++) {
            for (int i = 0; i < 16; i++) {
                const int x = i & 3;
                const int y = i / 4;
                block_image[(by * w/4 + bx) * 16 + i] = ((unsigned int *)data)[(by * 4 + y) * 4 * (w/4) + bx * 4 + x];
            }
        }
    }

    computePermutations(permutations);

    // h_result = (unsigned int*)malloc(compressedSize * sizeof(unsigned int));
    checkCudaErrors(cudaMallocHost((void **)&h_result, compressedSize * sizeof(unsigned int)));

    checkCudaErrors(cudaMalloc((void **)&d_data, memSize));
    checkCudaErrors(cudaMalloc((void **)&d_result, compressedSize));
    checkCudaErrors(cudaMalloc((void **)&d_permutations, 1024 * sizeof(unsigned int)));
};

void Dxtc::taskCallback(int insId, int nIter) {
    unsigned int blocks = ((w + 3) / 4) * ((h + 3) / 4);
    int blocksPerLaunch = min(blocks, 768 * 6);

    gcapsGpuSegBegin(fd, getpid(), sync_mode, ioctl_enabled);
    checkCudaErrors(cudaMemcpyAsync(d_permutations, permutations, 1024 * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_data, block_image, memSize, cudaMemcpyHostToDevice, stream));

    for (int j = 0; j < (int)blocks; j += blocksPerLaunch) {
        compress <<<min(blocksPerLaunch, blocks-j), NUM_THREADS, 0, stream>>> (d_permutations, d_data, (uint2 *)d_result, j);
    }

    __event_record_start(insId, nIter);
    for (int j = 0; j < (int)blocks; j += blocksPerLaunch) {
        compress <<<min(blocksPerLaunch, blocks-j), NUM_THREADS, 0, stream>>> (d_permutations, d_data, (uint2 *)d_result, j);
    }
    __event_record_stop(insId, nIter);

    checkCudaErrors(cudaMemcpyAsync(h_result, d_result, compressedSize, cudaMemcpyDeviceToHost, stream));
    gcapsGpuSegEnd(fd, getpid(), sync_mode, stream, ioctl_enabled);
}

void Dxtc::taskFinish() {
    checkCudaErrors(cudaFree(d_permutations));
    checkCudaErrors(cudaFree(d_data));
    checkCudaErrors(cudaFree(d_result));
    checkCudaErrors(cudaFreeHost(h_result));
    checkCudaErrors(cudaFreeHost(block_image));
    free(data);
    cuCtxDestroy(ctx);
}

void Dxtc::recordPriority(int priority) {
	this->prio = priority;
}