#ifndef __DXTC_H__
#define __DXTC_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>

#include "../base_task.h"


/* 
    For each task instance, we assume the matrix size doesn't change
*/
class Dxtc : public BaseTask {
    public:
        Dxtc(std::string, int fd, bool sync_mode, bool ioctl_enabled, bool suspension);
        ~Dxtc();
        /* functions */
        // init, run once at the system power on
        void taskInit();
        // finish, run once at the system power off
        void taskFinish();
        void taskCallback(int insId, int nIter) override;

        void recordPriority(int priority);
        /* variables  */
        cudaStream_t stream;
        cudaEvent_t start, stop;
        // dim3 DimGrid, DimBlock;
        std::string input_image;
        char *reference_image_path;
        unsigned char *data;
        unsigned int w, h;
        unsigned int memSize, compressedSize;
        unsigned int permutations[1024];
        unsigned int *d_data;
        unsigned int *d_result;
        unsigned int *d_permutations;
        unsigned int *h_result;
        unsigned int *block_image;

        int fd;
        bool sync_mode;
        bool ioctl_enabled;
        int event_flags = 0;

        int prio;

        CUcontext ctx;
        CUdevice device;
};

#endif /* __DXTC_H__ */
