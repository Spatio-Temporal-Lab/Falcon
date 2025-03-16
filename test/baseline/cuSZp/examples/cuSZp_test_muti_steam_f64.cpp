// #include <stdio.h>
// #include <stdlib.h>
// #include <unistd.h>
// #include <math.h>
// #include <cuda_runtime.h>
// #include <cuSZp.h>
// #include <iostream>

// // 定义常量，NUM_STREAMS 为 CUDA 流的数量
// #define NUM_STREAMS 3 // 使用3个CUDA Stream实现流水线
// // CUDA错误检查宏
// #define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
// {
//     if (code != cudaSuccess)
//     {
//         fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort) exit(code);
//     }
// }

// // 函数：在指定流上进行warmup操作。此操作将数据传输到GPU并进行压缩。
// void warmup(cudaStream_t stream, double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, double errorBound,
//             double* oriData, size_t chunkSize)
// {
//     size_t processedEle = 0;
//     // 当处理的数据小于总数据量时，继续处理
//     while (processedEle < nbEle)
//     {
//         // 计算每次处理的块大小
//         size_t chunkEle = (nbEle - processedEle) > chunkSize ? chunkSize : (nbEle - processedEle);

//         // 将数据从主机拷贝到设备
//         cudaMemcpy(d_oriData, oriData + processedEle, chunkEle * sizeof(double), cudaMemcpyHostToDevice);

//         // 在GPU上执行压缩操作
//         for (int i = 0; i < 3; i++)
//         {
//             // cuSZp_compress_plain_f32(d_oriData, d_cmpBytes, chunkEle, &processedEle, errorBound, stream);
//             cuSZp_compress_plain_f64(d_oriData, d_cmpBytes, chunkEle, &processedEle, errorBound, stream);
//         }

//         processedEle += chunkEle;
//     }
// }

// // 内存池类，用于管理内存分配和回收
// class MemoryPool
// {
// public:
//     MemoryPool(size_t poolSize, size_t chunkSize) : poolSize(poolSize), chunkSize(chunkSize)
//     {
//         cudaMalloc(&pool, poolSize); 
//         freeList = (void**)malloc(poolSize / chunkSize * sizeof(void*)); 
//         freeBlockCount = poolSize / chunkSize;

//         // 初始化空闲块列表
//         for (size_t i = 0; i < freeBlockCount; ++i)
//         {
//             freeList[i] = (void*)((char*)pool + i * chunkSize);
//         }

//         // 调试信息：打印内存池初始化情况
//         std::cout << "Memory pool initialized. Pool size: " << poolSize << " bytes, Free blocks: " << freeBlockCount << std::endl;
//     }

//     // 从内存池中获取一个块
//     void* allocate()
//     {
//         if (freeBlockCount == 0)
//         {
//             std::cerr << "No available memory blocks in the pool!" << std::endl;
//             return nullptr;
//         }
//         void* block = freeList[--freeBlockCount]; // 获取一个空闲块

//         // 调试信息：每次分配内存时输出信息
//         std::cout << "Allocated block: " << block << ", Free blocks left: " << freeBlockCount << std::endl;

//         return block;
//     }

//     // 归还内存块
//     void deallocate(void* block)
//     {
//         if (freeBlockCount >= poolSize / chunkSize)
//         {
//             std::cerr << "Memory pool is full!" << std::endl;
//             return;
//         }
//         freeList[freeBlockCount++] = block; // 将块放回空闲列表

//         // 调试信息：每次归还内存时输出信息
//         std::cout << "Deallocated block: " << block << ", Free blocks now: " << freeBlockCount << std::endl;
//     }

//     // 析构函数，释放内存池
//     ~MemoryPool()
//     {
//         cudaFree(pool); // 释放内存池
//         free(freeList); // 释放空闲块列表
//     }

// private:
//     void* pool{}; // 内存池的起始地址
//     void** freeList; // 空闲块链表
//     size_t poolSize; // 内存池的大小
//     size_t chunkSize; // 每个内存块的大小
//     size_t freeBlockCount; // 当前空闲块的数量
// };

// // 错误检查的函数，返回不匹配的元素数量
// int checkError(double* oriData, double* decData, size_t nbEle, double errorBound, int maxErrors = 10)
// {
//     int not_bound = 0;
//     // 计算相对误差，避免使用硬编码的阈值
//     for (size_t i = 0; i < nbEle; i++)
//     {
//         double absDiff = fabs(oriData[i] - decData[i]);
//         double relError = absDiff / fabs(oriData[i]);
//         if (relError > errorBound)  // 使用相对误差进行比较
//         {
//             not_bound++;
//             // 可以选择在这里打印出错的元素索引和值
//             std::cout << "Mismatch at index " << i << ": "
//                       << "Original value = " << oriData[i] << ", "
//                       << "Decompressed value = " << decData[i] << ", "
//                       << "Absolute error = " << absDiff << ", "
//                       << "Relative error = " << relError << std::endl;

//             // 如果错误达到最大限制，停止检查
//             if (not_bound >= maxErrors)
//             {
//                 std::cout << "Maximum error threshold reached. Stopping error check." << std::endl;
//                 break;
//             }
//         }
//     }
//     return not_bound;
// }
// // 主函数，执行整个压缩和解压流程
// int main()
// {
//     printf("Generating test data...\n\n");
//     // 用于计时的对象
//     TimingGPU timer_GPU;

//     // 输入数据的准备
//     double* oriData = NULL;
//     double* decData = NULL;
//     unsigned char* cmpBytes = NULL;
//     size_t nbEle = 1024 * 1024 * 512; // 8 GB fp32 数据
//     size_t cmpSize = 0;
//     size_t chunkSize = 1024;  // 数据块大小

//     // 在主机上分配内存
//     cudaHostAlloc((void**)&oriData, nbEle * sizeof(double), cudaHostAllocDefault);
//     cudaHostAlloc((void**)&decData, nbEle * sizeof(double), cudaHostAllocDefault);
//     cudaHostAlloc((void**)&cmpBytes, nbEle * sizeof(double), cudaHostAllocDefault);

//     // 初始化数据
//     printf("Generating test data...\n\n");
//     double startValue = -20.0f;
//     double step = 0.1f;
//     double endValue = 20.0f;
//     size_t idx = 0;
//     double value = startValue;
//     while (idx < nbEle)
//     {
//         oriData[idx++] = value;
//         value += step;
//         if (value > endValue)
//         {
//             value = startValue;
//         }
//     }

//     // 计算数据的最大值和最小值，来设置误差范围
//     double max_val = oriData[0];
//     double min_val = oriData[0];
//     for (size_t i = 0; i < nbEle; i++)
//     {
//         if (oriData[i] > max_val)
//             max_val = oriData[i];
//         else if (oriData[i] < min_val)
//             min_val = oriData[i];
//     }
//     double errorBound = (max_val - min_val) * 1E-2f;  // 设置误差范围

//     // 打印误差范围调试信息
//     std::cout << "Error bound: " << errorBound << std::endl;

//     // 初始化内存池，准备GPU内存
//     size_t freeMem, totalMem;
//     cudaMemGetInfo(&freeMem, &totalMem);
//     std::cout << "freeMem "<< freeMem <<" "<<totalMem<< std::endl;
//     size_t poolSize = freeMem * 0.4;  // 内存池大小
//     poolSize = (poolSize + 1024*2*sizeof(double)-1) & ~(1024*2*sizeof(double)-1);  // 向上对齐到 256 字节
//     chunkSize = poolSize * 0.4 / sizeof(double);
//     std::cout << "poolSize "<<poolSize<< std::endl;
//     std::cout << "chunkSize "<<chunkSize<< std::endl;

//     // 初始化内存池，管理内存分配
//     MemoryPool ori_data_pool(poolSize, chunkSize * sizeof(double)); 
//     MemoryPool cmp_bytes_pool(poolSize, poolSize * 0.5); 

//     // 为每个流分配内存
//     double* d_oriData[NUM_STREAMS];
//     double* d_decData[NUM_STREAMS];
//     unsigned char* d_cmpBytes[NUM_STREAMS];
//     cudaStream_t streams[NUM_STREAMS];
//     for (int i = 0; i < NUM_STREAMS; i++)
//     {
//         cudaStreamCreate(&streams[i]); // 创建CUDA流
//     }

//     size_t processedEle = 0;//已经压缩的数据数量
//     float timeElapsed = 0;
//     timer_GPU.StartCounter();  // 开始计时

//     // 压缩流程
//     while (processedEle < nbEle)
//     {
//         for (int i = 0; i < NUM_STREAMS; ++i)
//         {
//             size_t chunkEle = (nbEle - processedEle) > chunkSize ? chunkSize : (nbEle - processedEle);//当前块的压缩数据数量

//             // 调试信息：打印每次处理的数据块大小
//             std::cout << "Processing chunk " << processedEle << " to " << processedEle + chunkEle << " elements." << std::endl;

//             d_oriData[i] = (double*)ori_data_pool.allocate(); // 从内存池分配内存
//             d_cmpBytes[i] = (unsigned char*)cmp_bytes_pool.allocate(); 
//             std::cout << "Device memory for oriData[" << i << "]: " << d_oriData[i] << std::endl;
//             std::cout << "Device memory for cmpBytes[" << i << "]: " << d_cmpBytes[i] << std::endl;
            
//             // 异步将数据从主机拷贝到设备
//             cudaError_t err = cudaMemcpyAsync(d_oriData[i], oriData + processedEle, chunkEle * sizeof(double), cudaMemcpyHostToDevice, streams[i]);
//             // cudaError_t err = cudaMemcpy(d_oriData[i], oriData + processedEle, chunkEle * sizeof(double), cudaMemcpyHostToDevice);
//             if (err != cudaSuccess)
//             {
//                 std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
//                 return 0;
//             }

//             // 等待流中的所有操作完成
//             cudaStreamSynchronize(streams[i]);

//             // 执行压缩操作
//             size_t chunkCmpSize = 0;//本次压缩后的数据大小

//             std::cout << "begin comp chunk " << i << std::endl;
//             cuSZp_compress_plain_f64(d_oriData[i], d_cmpBytes[i], chunkEle, &chunkCmpSize, errorBound, streams[i]);
//             // GDFC_compress_plain_f32(float* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream)
//             // GDFC_compress_plain_f32(d_oriData[i], d_cmpBytes[i], chunkEle, &chunkCmpSize, streams[i]);
            
//             std::cout << "end comp chunk " << i << std::endl;

//             // 异步将结果从设备拷贝回主机
//             err = cudaMemcpyAsync(cmpBytes + cmpSize, d_cmpBytes[i], chunkCmpSize * sizeof(unsigned char), cudaMemcpyDeviceToHost, streams[i]);
//             if (err != cudaSuccess)
//             {
//                 std::cerr << "CUDA memcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
//                 return 0;
//             }
//             //更新
//             cmpSize += chunkCmpSize* sizeof(unsigned char);//当前压缩数据的总大小（偏移）
//             processedEle += chunkEle;

//             // 归还内存块
//             ori_data_pool.deallocate(d_oriData[i]);
//             cmp_bytes_pool.deallocate(d_cmpBytes[i]);
//         }
//     }

//     // 等待所有流完成操作
//     for (int i = 0; i < NUM_STREAMS; i++)
//     {
//         cudaStreamSynchronize(streams[i]);
//     }

//     // 打印压缩性能
//     timeElapsed = timer_GPU.GetCounter();
//     std::cout << "Compression finished!" << std::endl;
//     std::cout << "Compression ratio: " << static_cast<double>(nbEle) * sizeof(double) / cmpSize << std::endl;
//     std::cout << "End-to-end speed: " << (nbEle * sizeof(double) / 1024.0 / 1024.0) / timeElapsed << " GB/s" << std::endl;

//     // 错误检查：解压后的数据与原数据对比
//     // 解压后检查错误
//     int not_bound = checkError(oriData, decData, nbEle, errorBound);  // 使用新的错误检查函数

//     // 根据错误检查的结果打印信息
//     if (not_bound == 0)
//         printf("\033[0;32mPass error check!\033[0m\n");
//     else
//         printf("\033[0;31mFail error check! Exceeding data count: %d\033[0m\n", not_bound);


//     // 清理内存
//     cudaFreeHost(oriData);
//     cudaFreeHost(decData);
//     cudaFreeHost(cmpBytes);
//     for (int i = 0; i < NUM_STREAMS; i++)
//     {
//         cudaFree(d_oriData[i]);
//         cudaFree(d_decData[i]);
//         cudaFree(d_cmpBytes[i]);
//         cudaStreamDestroy(streams[i]);
//     }

//     return 0;
// }
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

    printf("cuSZp compression Size: %lu\n", cmpSize);
    printf("cuSZp compression origin Size: %lu\n", (nbEle * sizeof(double)));
    timeElapsed = timer_GPU.GetCounter();
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
