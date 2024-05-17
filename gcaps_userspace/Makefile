##########################################################
# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)
ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le armv7l))
    ifneq ($(TARGET_ARCH),$(HOST_ARCH))
        ifneq (,$(filter $(TARGET_ARCH),x86_64 aarch64 ppc64le))
            TARGET_SIZE := 64
        else ifneq (,$(filter $(TARGET_ARCH),armv7l))
            TARGET_SIZE := 32
        endif
    else
        TARGET_SIZE := $(shell getconf LONG_BIT)
    endif
else
    $(error ERROR - unsupported value $(TARGET_ARCH) for TARGET_ARCH!)
endif
ifneq ($(TARGET_ARCH),$(HOST_ARCH))
    ifeq (,$(filter $(HOST_ARCH)-$(TARGET_ARCH),aarch64-armv7l x86_64-armv7l x86_64-aarch64 x86_64-ppc64le))
        $(error ERROR - cross compiling from $(HOST_ARCH) to $(TARGET_ARCH) is not supported!)
    endif
endif

PROJECT_DIR := $(shell pwd)
APP_DIR := $(PROJECT_DIR)/app

CUDA_PATH ?= /usr/local/cuda-11.4

NVCC          := $(CUDA_PATH)/bin/nvcc
NVCCFLAGS 	:= -O3 -I/usr/local/cuda/include -lrt -lcuda -lcufft
LDFLAGS     := -lcudart -L/usr/local/cuda/lib64

INCLUDES  	:= -I/usr/local/include
INCLUDES 	+= -I/home/yidi/kernel-r3521/nvgpu/include/uapi # change this directory to the one that contains the copied /linux/nvgpu-ctrl.h file.
INCLUDES	+= -I$(PROJECT_DIR)
INCLUDES	+= -I$(PROJECT_DIR)/common/cuda_inc
INCLUDES  	+= -I$(CUDA_PATH)/samples/common/inc
LIBRARIES :=

SMS ?= 52 61 72
ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
endif
ifeq ($(GENCODE_FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

APP_BUILD_DIR = $(APP_DIR)/build
$(shell mkdir -p $(APP_BUILD_DIR))
$(MODEL_BUILD_DIR):
	mkdir -p $(APP_BUILD_DIR)
# $(info    APP_BUILD_DIR: $(APP_BUILD_DIR))
APP_LIST = mmulCPU/mmulCPU.cu mmulGPU/mmulGPU.cu dxtc/dxtc.cu hist/hist.cu projection/projection.cu
APP_SRC = $(patsubst %,$(APP_DIR)/%,$(APP_LIST))
APP_OBJ_LIST = mmulCPU.o mmulGPU.o dxtc.o hist.o projection.o

APP_OBJ = $(patsubst %,$(APP_BUILD_DIR)/%,$(APP_OBJ_LIST))

# app
$(APP_BUILD_DIR)/mmulCPU.o: $(APP_DIR)/mmulCPU/mmulCPU.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/mmulGPU.o: $(APP_DIR)/mmulGPU/mmulGPU.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/dxtc.o: $(APP_DIR)/dxtc/dxtc.cu
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/hist.o: $(APP_DIR)/hist/hist.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

$(APP_BUILD_DIR)/projection.o: $(APP_DIR)/projection/projection.cu 
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

all: main tsg_ctxsw
# Build
BUILD_DIR = build
$(shell mkdir -p $(BUILD_DIR))
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

main: $(BUILD_DIR)/main.o  $(APP_OBJ)
	$(NVCC) $(INCLUDES) $(LDFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
$(BUILD_DIR)/main.o: main.cc
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

tsg_ctxsw: $(BUILD_DIR)/tsg_ctxsw.o  $(APP_OBJ)
	$(NVCC) $(INCLUDES) $(LDFLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)
$(BUILD_DIR)/tsg_ctxsw.o: tsg_ctxsw.cc
	$(NVCC) $(INCLUDES) $(NVCCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

.PHONY: clean
clean:
	rm -rf $(APP_BUILD_DIR) $(BUILD_DIR) build main tsg_ctxsw
