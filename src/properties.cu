#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime.h>

void list_env_properties();

int main(int argc, char **argv){
    list_env_properties();	
}

void list_env_properties(){
    int deviceCount, device;
    struct cudaDeviceProp properties;
    cudaError_t cudaResultCode = cudaGetDeviceCount(&deviceCount);
    if (cudaResultCode != cudaSuccess)
        deviceCount = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < deviceCount; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) /* 9999 means emulation only */
        if (device==0) {
            printf("name:%s\n", properties.name);
            printf("memory:%ld\n", properties.totalGlobalMem);
            printf("warpsize:%d\n", properties.warpSize);
            printf("max threads per block:%d\n", properties.maxThreadsPerBlock);
            printf("clock rate:%d\n", properties.clockRate);
            printf("multiProcessorCount %d\n",properties.multiProcessorCount);
            printf("maxThreadsPerMultiProcessor %d\n",properties.maxThreadsPerMultiProcessor);
        }
    }
}
