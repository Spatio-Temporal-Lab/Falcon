//共享内存优化
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cstdlib> // for rand()
#include <ctime>   // for time()


//@
//
// CUDA核函数，计算差值
__global__ void forCompressKernel(int* data, int* compressedData, int referenceValue, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局线程索引
    if (idx < size) {
        compressedData[idx] = data[idx] - referenceValue; // 计算差值并存储
    }
}


//@
//
// GPU上的压缩函数
void forCompressGPU(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    int size = data.size(); // 数据大小
    referenceValue = data[0]; // 选择第一个值作为参考值

    int* d_data, * d_compressedData; // 定义设备指针(device)
    cudaMalloc(&d_data, size * sizeof(int)); // 在设备上分配内存
    cudaMalloc(&d_compressedData, size * sizeof(int)); // 为压缩数据分配内存

    cudaMemcpy(d_data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice); // 将数据从主机复制到设备

    int blockSize = 256; // 每个块的线程数
    int numBlocks = (size + blockSize - 1) / blockSize; // 计算所需的块数
    forCompressKernel<<<numBlocks, blockSize>>>(d_data, d_compressedData, referenceValue, size); // 启动核函数

    cudaMemcpy(compressedData.data(), d_compressedData, size * sizeof(int), cudaMemcpyDeviceToHost); // 将压缩结果从设备复制回主机

    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_compressedData);
}
// CPU上的压缩函数
void forCompressCPU(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    referenceValue = data[0]; // 选择第一个值作为参考值
    for (size_t i = 0; i < data.size(); ++i) {
        compressedData[i] = data[i] - referenceValue; // 计算差值
    }
}

//@-x
//
//gpu共享内存优化
__global__ void forCompressKernel2(int* data, int* compressedData, int referenceValue, int size) {
    extern __shared__ int sharedData[]; // 声明共享内存
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将数据加载到共享内存
    if (idx < size) {
        sharedData[threadIdx.x] = data[idx];
    }
    __syncthreads(); // 确保所有线程加载完毕

    // 计算差值
    if (idx < size) {
        compressedData[idx] = sharedData[threadIdx.x] - referenceValue;
    }
}

void forCompressGPU2(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    int size = data.size();
    referenceValue = data[0];

    int* d_data, * d_compressedData;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_compressedData, size * sizeof(int));

    cudaMemcpy(d_data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256; // 每个块的线程数
    int numBlocks = (size + blockSize - 1) / blockSize;
    forCompressKernel2<<<numBlocks, blockSize, blockSize * sizeof(int)>>>(d_data, d_compressedData, referenceValue, size); // 使用共享内存

    cudaMemcpy(compressedData.data(), d_compressedData, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_compressedData);
}

//@-y
//
//GPU常量内存
__constant__ int d_referenceValue; // 常量内存，用于存储参考值

__global__ void forCompressKernel3(int* data, int* compressedData, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局线程索引
    if (idx < size) {
        compressedData[idx] = data[idx] - d_referenceValue; // 计算差值
    }
}

void forCompressGPU3(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    int size = data.size();
    referenceValue = data[0];

    // 将参考值复制到常量内存
    cudaMemcpyToSymbol(d_referenceValue, &referenceValue, sizeof(int));

    int* d_data, * d_compressedData;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_compressedData, size * sizeof(int));

    cudaMemcpy(d_data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256; // 每个块的线程数
    int numBlocks = (size + blockSize - 1) / blockSize;
    forCompressKernel3<<<numBlocks, blockSize>>>(d_data, d_compressedData, size); 

    cudaMemcpy(compressedData.data(), d_compressedData, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudaFree(d_compressedData);
}

//@-z
//
//流+异步复制
__global__ void forCompressKernel4(int* data, int* compressedData, int referenceValue, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        compressedData[idx] = data[idx] - referenceValue;
    }
}

void forCompressGPU4(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    int size = data.size();
    referenceValue = data[0];

    int* d_data, * d_compressedData;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_compressedData, size * sizeof(int));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 异步复制数据到设备
    cudaMemcpyAsync(d_data, data.data(), size * sizeof(int), cudaMemcpyHostToDevice, stream);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    // 在流中启动核函数
    forCompressKernel4<<<numBlocks, blockSize, 0, stream>>>(d_data, d_compressedData, referenceValue, size);

    // 异步复制结果回主机
    cudaMemcpyAsync(compressedData.data(), d_compressedData, size * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // 等待流中的所有操作完成
    cudaStreamSynchronize(stream);

    // 释放设备内存和流
    cudaFree(d_data);
    cudaFree(d_compressedData);
    cudaStreamDestroy(stream);
}


int main() {
    const size_t dataSize = 100000; // 设置数据集数量，边界值10万，2亿
    std::vector<int> originalData(dataSize); // 创建原始数据向量
    std::srand(static_cast<unsigned>(std::time(0))); // 设置随机数种子

    // 随机生成数据
    for (size_t i = 0; i < dataSize; ++i) {
        originalData[i] = std::rand() % dataSize; // 生成范围在0到dataSize之间的随机数
    }

    int referenceValue; // 存储参考值
    std::vector<int> compressedData(originalData.size()); // 创建压缩数据向量

    // CPU压缩
    auto startCPU = std::chrono::high_resolution_clock::now(); // 记录开始时间
    forCompressCPU(originalData, compressedData, referenceValue); // 调用CPU压缩
    auto endCPU = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> cpuDuration = endCPU - startCPU; // 计算CPU压缩所需时间

    // 输出CPU压缩结果
    std::cout << "CPU Reference Value: " << referenceValue << std::endl;
    std::cout << "CPU Compression Time: " << cpuDuration.count() << " seconds" << std::endl;

    // GPU压缩
    compressedData.assign(originalData.size(), 0); // 清空压缩数据向量
    auto startGPU = std::chrono::high_resolution_clock::now(); // 记录开始时间
    forCompressGPU(originalData, compressedData, referenceValue); // 调用GPU压缩
    auto endGPU = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> gpuDuration = endGPU - startGPU; // 计算GPU压缩所需时间

    std::cout << "GPU Reference Value: " << referenceValue << std::endl;
    std::cout << "GPU Compression Time: " << gpuDuration.count() << " seconds" << std::endl;

    // GPU压缩优化-x
    compressedData.assign(originalData.size(), 0); // 清空压缩数据向量
    auto startGPU2 = std::chrono::high_resolution_clock::now(); // 记录开始时间
    forCompressGPU2(originalData, compressedData, referenceValue); // 调用GPU压缩
    auto endGPU2 = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> gpuDuration2 = endGPU2 - startGPU2; // 计算GPU压缩所需时间

    // 输出GPU压缩结果
    std::cout << "GPU-x Reference Value: " << referenceValue << std::endl;
    std::cout << "GPU-x Compression Time: " << gpuDuration2.count() << " seconds" << std::endl;

    // GPU压缩优化-y
    compressedData.assign(originalData.size(), 0); // 清空压缩数据向量
    auto startGPU3 = std::chrono::high_resolution_clock::now(); // 记录开始时间
    forCompressGPU3(originalData, compressedData, referenceValue); // 调用GPU压缩
    auto endGPU3 = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> gpuDuration3 = endGPU3 - startGPU3; // 计算GPU压缩所需时间

    // 输出GPU压缩结果
    std::cout << "GPU-y Reference Value: " << referenceValue << std::endl;
    std::cout << "GPU-y Compression Time: " << gpuDuration3.count() << " seconds" << std::endl;

    // GPU压缩优化-z
    compressedData.assign(originalData.size(), 0); // 清空压缩数据向量
    auto startGPU4 = std::chrono::high_resolution_clock::now(); // 记录开始时间
    forCompressGPU4(originalData, compressedData, referenceValue); // 调用GPU压缩
    auto endGPU4 = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> gpuDuration4 = endGPU4 - startGPU4; // 计算GPU压缩所需时间

    // 
    std::cout << "GPU-z Reference Value: " << referenceValue << std::endl;
    std::cout << "GPU-z Compression Time: " << gpuDuration4.count() << " seconds" << std::endl;



    return 0; // 返回0表示程序成功结束
}
