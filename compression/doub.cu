#include <iostream>
#include <vector>
#include <limits>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <chrono>  // For measuring time
#include <algorithm>
// 定义最大值常量
__device__ const double DBL_MAX = 1.7976931348623157e+308;

// 自定义原子最小值函数
__device__ void atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        if (val < __longlong_as_double(old)) {
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
        }
    } while (assumed != old);
}

// CUDA内核：框架参考算法
__global__ void frame_of_reference_kernel(double* d_data, double min_value, double* d_compressed_data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_compressed_data[idx] = d_data[idx] - min_value;
    }
}

// 计算最小值的CUDA内核
__global__ void find_min_kernel(double* d_data, double* d_min_value, int size) {
    extern __shared__ double sdata[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // 将数据复制到共享内存
    if (idx < size) {
        sdata[tid] = d_data[idx];
    } else {
        sdata[tid] = DBL_MAX; // 使用定义的常量
    }

    __syncthreads();

    // 归约操作，计算最小值
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmin(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    // 将最小值写入全局内存
    if (tid == 0) {
        atomicMinDouble(d_min_value, sdata[0]); // 使用自定义的原子最小值函数
    }
}

void frame_of_reference_cuda(const std::vector<double>& data, std::vector<double>& compressed_data) {
    int size = data.size();
    double* d_data;
    double* d_compressed_data;
    double* d_min_value;

    // 分配设备内存
    cudaMalloc(&d_data, size * sizeof(double));
    cudaMalloc(&d_compressed_data, size * sizeof(double));
    cudaMalloc(&d_min_value, sizeof(double));

    // 复制数据到设备
    cudaMemcpy(d_data, data.data(), size * sizeof(double), cudaMemcpyHostToDevice);
    
    // 初始化最小值
    double initial_min = DBL_MAX; // 使用定义的常量
    cudaMemcpy(d_min_value, &initial_min, sizeof(double), cudaMemcpyHostToDevice);

    // 计算最小值
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    find_min_kernel<<<numBlocks, blockSize, blockSize * sizeof(double)>>>(d_data, d_min_value, size);

    // 获取最小值
    double min_value;
    cudaMemcpy(&min_value, d_min_value, sizeof(double), cudaMemcpyDeviceToHost);

    // 在GPU上执行框架参考转换
    frame_of_reference_kernel<<<numBlocks, blockSize>>>(d_data, min_value, d_compressed_data, size);

    // 复制结果回主机
    compressed_data.resize(size);
    cudaMemcpy(compressed_data.data(), d_compressed_data, size * sizeof(double), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_data);
    cudaFree(d_compressed_data);
    cudaFree(d_min_value);
}

// CPU版本的框架参考算法
void frame_of_reference_cpu(const std::vector<double>& data, std::vector<double>& compressed_data) {
    double min_value = *std::min_element(data.begin(), data.end());
    compressed_data.resize(data.size());

    for (size_t i = 0; i < data.size(); ++i) {
        compressed_data[i] = data[i] - min_value;
    }
}

int main() {



    const int DATA_SIZE = 100000; // 可以根据需要调整大小
    std::vector<double> input_data(DATA_SIZE);
    std::vector<double> compressed_data;

    // 随机生成数据
    srand(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < DATA_SIZE; ++i) {
        input_data[i] = static_cast<double>(rand() % 100000)/static_cast<double>(rand() % 100000); // 生成0到999的随机数
    }

    // 测量CPU版本运行时间
    auto start_cpu = std::chrono::high_resolution_clock::now();
    frame_of_reference_cpu(input_data, compressed_data);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    // 测量CUDA版本运行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始事件
    cudaEventRecord(start);
    auto start_cuda = std::chrono::high_resolution_clock::now();
    frame_of_reference_cuda(input_data, compressed_data);
    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cuda_duration = end_cuda - start_cuda;

    // 记录结束事件
    cudaEventRecord(stop);

    // 等待内核完成
    cudaDeviceSynchronize();

    // 计算时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 输出运行时间
    std::cout << "CPU运行时间: " << cpu_duration.count() << "秒" << std::endl;
    std::cout << "CUDA运行时间: " << cuda_duration.count() << "秒" << std::endl;

    // 输出转换后的数据的一部分
    std::cout << "Compressed Data (first 10 values): ";
    for (int i = 0; i < 10; ++i) {
        std::cout << compressed_data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
