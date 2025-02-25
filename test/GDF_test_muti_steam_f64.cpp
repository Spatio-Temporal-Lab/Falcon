
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuSZp.h>
#include <stdint.h>
#include <iostream>

#define NUM_STREAMS 3 // 使用3个CUDA Stream实现流水线
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void warmup(cudaStream_t stream, double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, double errorBound,
            double* oriData, size_t chunkSize)
{
    size_t processedEle = 0;
    while (processedEle < nbEle)
    {
        size_t chunkEle = (nbEle - processedEle) > chunkSize ? chunkSize : (nbEle - processedEle);

        // 将数据分块拷贝到GPU
        cudaMemcpy(d_oriData, oriData + processedEle, chunkEle * sizeof(double), cudaMemcpyHostToDevice);

        // 运行warmup压缩
        for (int i = 0; i < 3; i++)
        {
            GDFC_compress_plain_f64(d_oriData, d_cmpBytes, chunkEle, &processedEle, stream);
        }

        processedEle += chunkEle;
    }
}

class MemoryPool
{
public:
    MemoryPool(size_t poolSize, size_t chunkSize) : poolSize(poolSize), chunkSize(chunkSize)
    {
        cudaMalloc(&pool, poolSize); // 为整个内存池分配显存
        freeList = (void**)malloc(poolSize / chunkSize * sizeof(void*)); // 用于管理空闲块
        freeBlockCount = poolSize / chunkSize;

        // 初始化空闲块列表
        for (size_t i = 0; i < freeBlockCount; ++i)
        {
            freeList[i] = (void*)((char*)pool + i * chunkSize);
        }
    }

    // 从内存池中获取内存
    void* allocate()
    {
        if (freeBlockCount == 0)
        {
            std::cerr << "No available memory blocks in the pool!" << std::endl;
            return nullptr;
        }
        void* block = freeList[--freeBlockCount]; // 获取一个空闲块
        return block;
    }

    // 归还内存块
    void deallocate(void* block)
    {
        if (freeBlockCount >= poolSize / chunkSize)
        {
            std::cerr << "Memory pool is full!" << std::endl;
            return;
        }
        freeList[freeBlockCount++] = block; // 将块放回空闲列表
    }

    ~MemoryPool()
    {
        cudaFree(pool); // 释放内存池
        free(freeList); // 释放空闲列表
    }

private:
    void* pool{}; // 内存池的起始地址
    void** freeList; // 空闲块链表
    size_t poolSize; // 内存池的大小
    size_t chunkSize; // 每个内存块的大小
    size_t freeBlockCount; // 当前空闲块的数量
};

int main()
{
    printf("Generating test data...\n\n");
    // For measuring the end-to-end throughput.
    TimingGPU timer_GPU;

    // Input data preparation on CPU.
    double* oriData = NULL;
    double* decData = NULL;
    unsigned char* cmpBytes = NULL;
    size_t nbEle = 1024 * 1024 * 512/2; // 8 GB fp64 data.
    size_t cmpSize = 0;
    size_t chunkSize = 1024;

    cudaHostAlloc((void**)&oriData, nbEle * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&decData, nbEle * sizeof(double), cudaHostAllocDefault);
    cudaHostAlloc((void**)&cmpBytes, nbEle * sizeof(double), cudaHostAllocDefault);

    // Initialize oriData.
    printf("Generating test data...\n\n");
    double startValue = -20.0;
    double step = 0.1;
    double endValue = 20.0;
    size_t idx = 0;
    double value = startValue;
    while (idx < nbEle)
    {
        oriData[idx++] = value;
        value += step;
        if (value > endValue)
        {
            value = startValue;
        }
    }

    // Get value range for relative error testing
    double max_val = oriData[0];
    double min_val = oriData[0];
    for (size_t i = 0; i < nbEle; i++)
    {
        if (oriData[i] > max_val)
            max_val = oriData[i];
        else if (oriData[i] < min_val)
            min_val = oriData[i];
    }
    double errorBound = (max_val - min_val) * 1E-2;

    // Compression and decompression in chunks using pipeline with multiple streams.
    printf("=================================================\n");
    printf("=========Testing cuSZp on REL 1E-2==============\n");
    printf("=================================================\n");

    // GPU memory allocation.
    double* d_oriData[NUM_STREAMS];
    double* d_decData[NUM_STREAMS];
    unsigned char* d_cmpBytes[NUM_STREAMS];
    cudaStream_t streams[NUM_STREAMS];
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "freeMem" << freeMem << " " << totalMem << std::endl;

    size_t poolSize = freeMem * 0.4;
    poolSize = (poolSize + 1024*2*sizeof(double)-1) & ~(1024*2*sizeof(double)-1);  // 向上对齐到 256 字节
    chunkSize = poolSize * 0.5 / sizeof(double);

    MemoryPool ori_data_pool(poolSize, chunkSize * sizeof(double)); // 原始数据显存池
    MemoryPool cmp_bytes_pool(poolSize, chunkSize * sizeof(double)); // 压缩结果显存池
    std::cout << "oriData: " << nbEle << std::endl;
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    size_t processedEle = 0;
    double timeElapsed = 0;

    timer_GPU.StartCounter();

    while (processedEle < nbEle)
    {
        for (int i = 0; i < NUM_STREAMS; ++i)
        {
            size_t chunkEle = (nbEle - processedEle) > chunkSize ? chunkSize : (nbEle - processedEle);
            if(chunkEle==0)
            {
                break;
            }
            std::cout << "Processing chunk: " << processedEle << " to " << processedEle + chunkEle << std::endl;

            // 异步将数据从 CPU 拷贝到 GPU
            d_oriData[i] = (double*)ori_data_pool.allocate(); // 从内存池分配显存
            d_cmpBytes[i] = (unsigned char*)cmp_bytes_pool.allocate(); // 从内存池分配显存

            cudaError_t err = cudaMemcpyAsync(d_oriData[i], oriData + processedEle, chunkEle * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA malloc failed for d_oriData[i]: " << cudaGetErrorString(err) << std::endl;
                return 0;
            }


            // 执行压缩
            size_t chunkCmpSize = 0;
            GDFC_compress_plain_f64(d_oriData[i], d_cmpBytes[i], chunkEle, &chunkCmpSize, streams[i]);

            // 将压缩结果从 GPU 拷贝回 CPU
            err = cudaMemcpyAsync(cmpBytes + cmpSize, d_cmpBytes[i], chunkCmpSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[i]);

            if (err != cudaSuccess)
            {
                std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
                return 0;
            }
            processedEle += chunkEle;
            cmpSize += chunkCmpSize;
            // 归还内存块
            ori_data_pool.deallocate(d_oriData[i]);
            cmp_bytes_pool.deallocate(d_cmpBytes[i]);
        }
    }

    // 等待所有流完成操作
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }

    timeElapsed = timer_GPU.GetCounter();
    printf("cuSZp compression Size: %lu\n", cmpSize);
    printf("cuSZp compression origin Size: %lu\n", (nbEle * sizeof(double)));

    printf("cuSZp finished!\n");
    printf("cuSZp compression   end-to-end speed: %f GB/s\n", (nbEle * sizeof(double) / 1024.0 / 1024.0) / timeElapsed);
    printf("cuSZp compression ratio: %f\n", static_cast<double>(nbEle) * sizeof(double) / cmpSize);

    // 清理内存
    cudaFreeHost(oriData);
    cudaFreeHost(decData);
    cudaFreeHost(cmpBytes);
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        cudaFree(d_oriData[i]);
        cudaFree(d_decData[i]);
        cudaFree(d_cmpBytes[i]);
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
