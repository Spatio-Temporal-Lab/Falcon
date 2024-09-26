#include <iostream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cstdlib> // for rand()
#include <ctime>   // for time()

// CUDA核函数，计算差值
__global__ void forCompressKernel(int* data, int* compressedData, int referenceValue, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // 计算全局线程索引
    if (idx < size) {
        compressedData[idx] = data[idx] - referenceValue; // 计算差值并存储
    }
}

// CPU上的压缩函数
void forCompressCPU(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    referenceValue = data[0]; // 选择第一个值作为参考值
    for (size_t i = 0; i < data.size(); ++i) {
        compressedData[i] = data[i] - referenceValue; // 计算差值
    }
}

// GPU上的压缩函数
void forCompressGPU(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    int size = data.size(); // 数据大小
    referenceValue = data[0]; // 选择第一个值作为参考值

    int* d_data, * d_compressedData; // 定义设备指针
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

int main() {
    const size_t dataSize = 200000000; // 数据集大小设置为2亿
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

    // 输出GPU压缩结果
    std::cout << "GPU Reference Value: " << referenceValue << std::endl;
    std::cout << "GPU Compression Time: " << gpuDuration.count() << " seconds" << std::endl;

    return 0; // 返回0表示程序成功结束
}
