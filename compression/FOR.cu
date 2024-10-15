#include <iostream>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <cstdlib> // for rand()
#include <ctime>   // for time()

// CUDA核函数，计算差值
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

void forCompressCPU(const std::vector<int>& data, std::vector<int>& compressedData, int& referenceValue) {
    referenceValue = data[0]; // 选择第一个值作为参考值
    for (size_t i = 0; i < data.size(); ++i) {
        compressedData[i] = data[i] - referenceValue; // 计算差值
    }
}

int main() {
    const size_t dataSize = 100000; // 数据集大小设置为2亿
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

    // GPU压缩优化-x
    compressedData.assign(originalData.size(), 0); // 清空压缩数据向量
    auto startGPU = std::chrono::high_resolution_clock::now(); // 记录开始时间
    forCompressGPU2(originalData, compressedData, referenceValue); // 调用GPU压缩
    auto endGPU = std::chrono::high_resolution_clock::now(); // 记录结束时间
    std::chrono::duration<double> gpuDuration2 = endGPU - startGPU; // 计算GPU压缩所需时间

    // 输出GPU压缩结果
    std::cout << "GPU-x Reference Value: " << referenceValue << std::endl;
    std::cout << "GPU-x Compression Time: " << gpuDuration2.count() << " seconds" << std::endl;

    return 0; // 返回0表示程序成功结束
}
