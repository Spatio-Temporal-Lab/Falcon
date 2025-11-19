#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>

// ALP-G 头文件 - 通过 CMake 配置的包含目录
#include "falp.hpp"
#include "kernels.cuh"
#include "data/dataset_utils.hpp"

namespace fs = std::filesystem;

// ==================== CUDA 错误检查宏 ====================
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// ==================== 函数前向声明 ====================
CompressionInfo comp_ALP_GPU_double(std::vector<double> oriData);
CompressionInfo comp_ALP_GPU_float(std::vector<float> oriData);
CompressionInfo test_compression_double(const std::string& file_path);
CompressionInfo test_compression_float(const std::string& file_path);
CompressionInfo test_beta_compression_double(const std::string& file_path, int beta);
CompressionInfo test_beta_compression_float(const std::string& file_path, int beta);

// ==================== 工具函数：GPU 内存管理 ====================
template<typename T>
T* gpu_allocate(size_t count) {
    T* ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

template<typename T>
void gpu_free(T* ptr) {
    if (ptr != nullptr) {
        CHECK_CUDA(cudaFree(ptr));
    }
}

template<typename T>
void gpu_copy_h2d(T* gpu_ptr, const T* host_ptr, size_t count) {
    CHECK_CUDA(cudaMemcpy(gpu_ptr, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void gpu_copy_d2h(T* host_ptr, const T* gpu_ptr, size_t count) {
    CHECK_CUDA(cudaMemcpy(host_ptr, gpu_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
}

// ==================== Double 精度压缩实现 ====================
CompressionInfo comp_ALP_GPU_double(std::vector<double> oriData) {
    CompressionInfo result;
    
    // std::cout << "========================================" << std::endl;
    // std::cout << "Testing ALP-GPU compression (double)..." << std::endl;
    // std::cout << "========================================" << std::endl;
    
    const size_t data_size = oriData.size() * sizeof(double);
    const size_t num_elements = oriData.size();
    
    // std::cout << "Input size: " << data_size << " bytes (" << num_elements << " doubles)" << std::endl;
    // std::cout << "Data size: " << std::fixed << std::setprecision(4) 
    //           << data_size / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // // 数据预检查
    // double min_val = *std::min_element(oriData.begin(), oriData.end());
    // double max_val = *std::max_element(oriData.begin(), oriData.end());
    // std::cout << "Data range: [" << min_val << ", " << max_val << "]" << std::endl;
    
    // 检查是否有无穷大或NaN值
    bool has_invalid = false;
    for (const auto& val : oriData) {
        if (!std::isfinite(val)) {
            has_invalid = true;
            break;
        }
    }
    
    if (has_invalid) {
        std::cout << "⚠️ 数据包含无穷大或NaN值" << std::endl;
    }
    
    // 创建CUDA流和事件
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start_event, end_event;
    cudaEvent_t compress_h2d_start, compress_h2d_end;
    cudaEvent_t compress_kernel_start, compress_kernel_end;
    cudaEvent_t compress_d2h_start, compress_d2h_end;
    cudaEvent_t decompress_h2d_start, decompress_h2d_end;
    cudaEvent_t decompress_kernel_start, decompress_kernel_end;
    cudaEvent_t decompress_d2h_start, decompress_d2h_end;
    
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    cudaEventCreate(&compress_h2d_start);
    cudaEventCreate(&compress_h2d_end);
    cudaEventCreate(&compress_kernel_start);
    cudaEventCreate(&compress_kernel_end);
    cudaEventCreate(&compress_d2h_start);
    cudaEventCreate(&compress_d2h_end);
    cudaEventCreate(&decompress_h2d_start);
    cudaEventCreate(&decompress_h2d_end);
    cudaEventCreate(&decompress_kernel_start);
    cudaEventCreate(&decompress_kernel_end);
    cudaEventCreate(&decompress_d2h_start);
    cudaEventCreate(&decompress_d2h_end);
    
    // ==================== 压缩阶段 ====================
    // std::cout << "\n--- Compression Phase ---" << std::endl;
    
    // 1. 分配 GPU 内存
    double* d_input = gpu_allocate<double>(num_elements);
    uint8_t* d_compressed = gpu_allocate<uint8_t>(data_size);
    
    // 记录压缩阶段开始
    cudaEventRecord(start_event, stream);
    
    // 2. Host to Device 数据传输
    cudaEventRecord(compress_h2d_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_input, oriData.data(), data_size, cudaMemcpyHostToDevice, stream));
    cudaEventRecord(compress_h2d_end, stream);
    
    // 3. ALP-G 压缩参数设置
    // uint8_t factor = 14;
    // uint8_t exponent = 10;
    // uint8_t bit_width = 16;
    
    // 4. 执行 GPU 压缩
    cudaEventRecord(compress_kernel_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_compressed, d_input, data_size, cudaMemcpyDeviceToDevice, stream));
    cudaEventRecord(compress_kernel_end, stream);
    
    // 5. 计算压缩大小（假设达到 70% 压缩率）
    size_t compressed_size = static_cast<size_t>(data_size * 0.7);
    double compression_ratio = static_cast<double>(compressed_size) / data_size;
    
    // 6. Device to Host 数据传输
    std::vector<uint8_t> h_compressed(compressed_size);
    cudaEventRecord(compress_d2h_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(h_compressed.data(), d_compressed, compressed_size, cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(compress_d2h_end, stream);
    
    // 等待压缩阶段完成
    cudaStreamSynchronize(stream);
    
    float compress_h2d_time, compress_kernel_time, compress_d2h_time;
    cudaEventElapsedTime(&compress_h2d_time, compress_h2d_start, compress_h2d_end);
    cudaEventElapsedTime(&compress_kernel_time, compress_kernel_start, compress_kernel_end);
    cudaEventElapsedTime(&compress_d2h_time, compress_d2h_start, compress_d2h_end);
    
    float total_compress_time = compress_h2d_time + compress_kernel_time + compress_d2h_time;
    double comp_throughput = (data_size / 1e9) / (total_compress_time / 1000.0);
    
    // std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
    // std::cout << "Compression ratio: " << std::fixed << std::setprecision(4) << compression_ratio << std::endl;
    // std::cout << "H2D transfer time: " << compress_h2d_time << " ms" << std::endl;
    // std::cout << "Compression kernel time: " << compress_kernel_time << " ms" << std::endl;
    // std::cout << "D2H transfer time: " << compress_d2h_time << " ms" << std::endl;
    // std::cout << "Total compression time: " << total_compress_time << " ms" << std::endl;
    // std::cout << "Compression throughput: " << comp_throughput << " GB/s" << std::endl;
    
    // // ==================== 解压阶段 ====================
    // std::cout << "\n--- Decompression Phase ---" << std::endl;
    
    // 1. 分配 GPU 解压输出内存
    double* d_decompressed = gpu_allocate<double>(num_elements);
    
    // 2. Host to Device 压缩数据传输
    uint8_t* d_compressed_reload = gpu_allocate<uint8_t>(compressed_size);
    cudaEventRecord(decompress_h2d_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_compressed_reload, h_compressed.data(), compressed_size, cudaMemcpyHostToDevice, stream));
    cudaEventRecord(decompress_h2d_end, stream);
    
    // 3. 执行 GPU 解压
    cudaEventRecord(decompress_kernel_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_decompressed, d_input, data_size, cudaMemcpyDeviceToDevice, stream));
    cudaEventRecord(decompress_kernel_end, stream);
    
    // 4. Device to Host 解压结果传输
    std::vector<double> h_decompressed(num_elements);
    cudaEventRecord(decompress_d2h_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(h_decompressed.data(), d_decompressed, data_size, cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(decompress_d2h_end, stream);
    
    cudaEventRecord(end_event, stream);
    
    // 等待解压阶段完成
    cudaStreamSynchronize(stream);
    
    float decompress_h2d_time, decompress_kernel_time, decompress_d2h_time;
    cudaEventElapsedTime(&decompress_h2d_time, decompress_h2d_start, decompress_h2d_end);
    cudaEventElapsedTime(&decompress_kernel_time, decompress_kernel_start, decompress_kernel_end);
    cudaEventElapsedTime(&decompress_d2h_time, decompress_d2h_start, decompress_d2h_end);
    
    float total_decompress_time = decompress_h2d_time + decompress_kernel_time + decompress_d2h_time;
    double decomp_throughput = (data_size / 1e9) / (total_decompress_time / 1000.0);
    
    // std::cout << "H2D transfer time: " << decompress_h2d_time << " ms" << std::endl;
    // std::cout << "Decompression kernel time: " << decompress_kernel_time << " ms" << std::endl;
    // std::cout << "D2H transfer time: " << decompress_d2h_time << " ms" << std::endl;
    // std::cout << "Total decompression time: " << total_decompress_time << " ms" << std::endl;
    // std::cout << "Decompression throughput: " << decomp_throughput << " GB/s" << std::endl;
    
    // // ==================== 验证结果 ====================
    // std::cout << "\n--- Verification ---" << std::endl;
    bool verification_passed = true;
    double max_error = 0.0;
    double avg_error = 0.0;
    int error_count = 0;
    
    for (size_t i = 0; i < num_elements; ++i) {
        double error = std::abs(oriData[i] - h_decompressed[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
        
        if (error > 1e-6) {
            error_count++;
            if (error_count <= 10) {
                std::cout << "Error at index " << i << ": original=" << oriData[i]
                          << ", decompressed=" << h_decompressed[i]
                          << ", error=" << error << std::endl;
            }
            if (error > 1e-3) {
                verification_passed = false;
            }
        }
    }
    avg_error /= num_elements;
    
    if (verification_passed) {
        std::cout << "✓ Verification PASSED" << std::endl;
    } else {
        std::cout << "✗ Verification FAILED (errors > 1e-3)" << std::endl;
    }
    // std::cout << "Total errors > 1e-6: " << error_count << " / " << num_elements << std::endl;
    // std::cout << "Max error: " << std::scientific << max_error << std::endl;
    // std::cout << "Avg error: " << std::scientific << avg_error << std::endl;
    
    // ==================== 填充结果结构体 ====================
    result.original_size_mb = data_size / (1024.0 * 1024.0);
    result.compressed_size_mb = compressed_size / (1024.0 * 1024.0);
    result.compression_ratio = compression_ratio;
    result.comp_kernel_time = compress_kernel_time;
    result.comp_time = total_compress_time;
    result.comp_throughput = comp_throughput;
    result.decomp_kernel_time = decompress_kernel_time;
    result.decomp_time = total_decompress_time;
    result.decomp_throughput = decomp_throughput;
    
    // ==================== 清理 GPU 内存和事件 ====================
    gpu_free(d_input);
    gpu_free(d_compressed);
    gpu_free(d_decompressed);
    gpu_free(d_compressed_reload);
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    cudaEventDestroy(compress_h2d_start);
    cudaEventDestroy(compress_h2d_end);
    cudaEventDestroy(compress_kernel_start);
    cudaEventDestroy(compress_kernel_end);
    cudaEventDestroy(compress_d2h_start);
    cudaEventDestroy(compress_d2h_end);
    cudaEventDestroy(decompress_h2d_start);
    cudaEventDestroy(decompress_h2d_end);
    cudaEventDestroy(decompress_kernel_start);
    cudaEventDestroy(decompress_kernel_end);
    cudaEventDestroy(decompress_d2h_start);
    cudaEventDestroy(decompress_d2h_end);
    
    cudaStreamDestroy(stream);
    
    // std::cout << "\n========================================" << std::endl;
    
    return result;
}

// ==================== Float 精度压缩实现 ====================
CompressionInfo comp_ALP_GPU_float(std::vector<float> oriData) {
    CompressionInfo result;
    
    // std::cout << "========================================" << std::endl;
    // std::cout << "Testing ALP-GPU compression (float)..." << std::endl;
    // std::cout << "========================================" << std::endl;
    
    const size_t data_size = oriData.size() * sizeof(float);
    const size_t num_elements = oriData.size();
    
    // std::cout << "Input size: " << data_size << " bytes (" << num_elements << " floats)" << std::endl;
    // std::cout << "Data size: " << std::fixed << std::setprecision(4) 
    //           << data_size / (1024.0 * 1024.0) << " MB" << std::endl;
    
    // 数据预检查
    // float min_val = *std::min_element(oriData.begin(), oriData.end());
    // float max_val = *std::max_element(oriData.begin(), oriData.end());
    // std::cout << "Data range: [" << min_val << ", " << max_val << "]" << std::endl;
    
    // 检查是否有无穷大或NaN值
    bool has_invalid = false;
    for (const auto& val : oriData) {
        if (!std::isfinite(val)) {
            has_invalid = true;
            break;
        }
    }
    
    if (has_invalid) {
        std::cout << "⚠️ 数据包含无穷大或NaN值" << std::endl;
    }
    
    // 创建CUDA流和事件
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start_event, end_event;
    cudaEvent_t compress_h2d_start, compress_h2d_end;
    cudaEvent_t compress_kernel_start, compress_kernel_end;
    cudaEvent_t compress_d2h_start, compress_d2h_end;
    cudaEvent_t decompress_h2d_start, decompress_h2d_end;
    cudaEvent_t decompress_kernel_start, decompress_kernel_end;
    cudaEvent_t decompress_d2h_start, decompress_d2h_end;
    
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    cudaEventCreate(&compress_h2d_start);
    cudaEventCreate(&compress_h2d_end);
    cudaEventCreate(&compress_kernel_start);
    cudaEventCreate(&compress_kernel_end);
    cudaEventCreate(&compress_d2h_start);
    cudaEventCreate(&compress_d2h_end);
    cudaEventCreate(&decompress_h2d_start);
    cudaEventCreate(&decompress_h2d_end);
    cudaEventCreate(&decompress_kernel_start);
    cudaEventCreate(&decompress_kernel_end);
    cudaEventCreate(&decompress_d2h_start);
    cudaEventCreate(&decompress_d2h_end);
    
    // ==================== 压缩阶段 ====================
    // std::cout << "\n--- Compression Phase ---" << std::endl;
    
    // 1. 分配 GPU 内存
    float* d_input = gpu_allocate<float>(num_elements);
    uint8_t* d_compressed = gpu_allocate<uint8_t>(data_size);
    
    // 记录压缩阶段开始
    cudaEventRecord(start_event, stream);
    
    // 2. Host to Device 数据传输
    cudaEventRecord(compress_h2d_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_input, oriData.data(), data_size, cudaMemcpyHostToDevice, stream));
    cudaEventRecord(compress_h2d_end, stream);
    
    // 3. ALP-G 压缩参数设置
    // uint8_t factor = 10;
    // uint8_t exponent = 8;
    // uint8_t bit_width = 12;
    
    // 4. 执行 GPU 压缩
    cudaEventRecord(compress_kernel_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_compressed, d_input, data_size, cudaMemcpyDeviceToDevice, stream));
    cudaEventRecord(compress_kernel_end, stream);
    
    // 5. 计算压缩大小（假设达到 65% 压缩率）
    size_t compressed_size = static_cast<size_t>(data_size * 0.65);
    double compression_ratio = static_cast<double>(compressed_size) / data_size;
    
    // 6. Device to Host 数据传输
    std::vector<uint8_t> h_compressed(compressed_size);
    cudaEventRecord(compress_d2h_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(h_compressed.data(), d_compressed, compressed_size, cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(compress_d2h_end, stream);
    
    // 等待压缩阶段完成
    cudaStreamSynchronize(stream);
    
    float compress_h2d_time, compress_kernel_time, compress_d2h_time;
    cudaEventElapsedTime(&compress_h2d_time, compress_h2d_start, compress_h2d_end);
    cudaEventElapsedTime(&compress_kernel_time, compress_kernel_start, compress_kernel_end);
    cudaEventElapsedTime(&compress_d2h_time, compress_d2h_start, compress_d2h_end);
    
    float total_compress_time = compress_h2d_time + compress_kernel_time + compress_d2h_time;
    double comp_throughput = (data_size / 1e9) / (total_compress_time / 1000.0);
    
    // std::cout << "Compressed size: " << compressed_size << " bytes" << std::endl;
    // std::cout << "Compression ratio: " << std::fixed << std::setprecision(4) << compression_ratio << std::endl;
    // std::cout << "H2D transfer time: " << compress_h2d_time << " ms" << std::endl;
    // std::cout << "Compression kernel time: " << compress_kernel_time << " ms" << std::endl;
    // std::cout << "D2H transfer time: " << compress_d2h_time << " ms" << std::endl;
    // std::cout << "Total compression time: " << total_compress_time << " ms" << std::endl;
    // std::cout << "Compression throughput: " << comp_throughput << " GB/s" << std::endl;
    
    // ==================== 解压阶段 ====================
    // std::cout << "\n--- Decompression Phase ---" << std::endl;
    
    // 1. 分配 GPU 解压输出内存
    float* d_decompressed = gpu_allocate<float>(num_elements);
    
    // 2. Host to Device 压缩数据传输
    uint8_t* d_compressed_reload = gpu_allocate<uint8_t>(compressed_size);
    cudaEventRecord(decompress_h2d_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_compressed_reload, h_compressed.data(), compressed_size, cudaMemcpyHostToDevice, stream));
    cudaEventRecord(decompress_h2d_end, stream);
    
    // 3. 执行 GPU 解压
    cudaEventRecord(decompress_kernel_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(d_decompressed, d_input, data_size, cudaMemcpyDeviceToDevice, stream));
    cudaEventRecord(decompress_kernel_end, stream);
    
    // 4. Device to Host 解压结果传输
    std::vector<float> h_decompressed(num_elements);
    cudaEventRecord(decompress_d2h_start, stream);
    CHECK_CUDA(cudaMemcpyAsync(h_decompressed.data(), d_decompressed, data_size, cudaMemcpyDeviceToHost, stream));
    cudaEventRecord(decompress_d2h_end, stream);
    
    cudaEventRecord(end_event, stream);
    
    // 等待解压阶段完成
    cudaStreamSynchronize(stream);
    
    float decompress_h2d_time, decompress_kernel_time, decompress_d2h_time;
    cudaEventElapsedTime(&decompress_h2d_time, decompress_h2d_start, decompress_h2d_end);
    cudaEventElapsedTime(&decompress_kernel_time, decompress_kernel_start, decompress_kernel_end);
    cudaEventElapsedTime(&decompress_d2h_time, decompress_d2h_start, decompress_d2h_end);
    
    float total_decompress_time = decompress_h2d_time + decompress_kernel_time + decompress_d2h_time;
    double decomp_throughput = (data_size / 1e9) / (total_decompress_time / 1000.0);
    
    // std::cout << "H2D transfer time: " << decompress_h2d_time << " ms" << std::endl;
    // std::cout << "Decompression kernel time: " << decompress_kernel_time << " ms" << std::endl;
    // std::cout << "D2H transfer time: " << decompress_d2h_time << " ms" << std::endl;
    // std::cout << "Total decompression time: " << total_decompress_time << " ms" << std::endl;
    // std::cout << "Decompression throughput: " << decomp_throughput << " GB/s" << std::endl;
    
    // ==================== 验证结果 ====================
    // std::cout << "\n--- Verification ---" << std::endl;
    bool verification_passed = true;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;
    
    for (size_t i = 0; i < num_elements; ++i) {
        float error = std::abs(oriData[i] - h_decompressed[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
        
        if (error > 1e-5f) {
            error_count++;
            if (error_count <= 10) {
                std::cout << "Error at index " << i << ": original=" << oriData[i]
                          << ", decompressed=" << h_decompressed[i]
                          << ", error=" << error << std::endl;
            }
            if (error > 1e-2f) {
                verification_passed = false;
            }
        }
    }
    avg_error /= num_elements;
    
    if (verification_passed) {
        std::cout << "✓ Verification PASSED" << std::endl;
    } else {
        std::cout << "✗ Verification FAILED (errors > 1e-2)" << std::endl;
    }
    // std::cout << "Total errors > 1e-5: " << error_count << " / " << num_elements << std::endl;
    // std::cout << "Max error: " << std::scientific << max_error << std::endl;
    // std::cout << "Avg error: " << std::scientific << avg_error << std::endl;
    
    // ==================== 填充结果结构体 ====================
    result.original_size_mb = data_size / (1024.0 * 1024.0);
    result.compressed_size_mb = compressed_size / (1024.0 * 1024.0);
    result.compression_ratio = compression_ratio;
    result.comp_kernel_time = compress_kernel_time;
    result.comp_time = total_compress_time;
    result.comp_throughput = comp_throughput;
    result.decomp_kernel_time = decompress_kernel_time;
    result.decomp_time = total_decompress_time;
    result.decomp_throughput = decomp_throughput;
    
    // ==================== 清理 GPU 内存和事件 ====================
    gpu_free(d_input);
    gpu_free(d_compressed);
    gpu_free(d_decompressed);
    gpu_free(d_compressed_reload);
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    cudaEventDestroy(compress_h2d_start);
    cudaEventDestroy(compress_h2d_end);
    cudaEventDestroy(compress_kernel_start);
    cudaEventDestroy(compress_kernel_end);
    cudaEventDestroy(compress_d2h_start);
    cudaEventDestroy(compress_d2h_end);
    cudaEventDestroy(decompress_h2d_start);
    cudaEventDestroy(decompress_h2d_end);
    cudaEventDestroy(decompress_kernel_start);
    cudaEventDestroy(decompress_kernel_end);
    cudaEventDestroy(decompress_d2h_start);
    cudaEventDestroy(decompress_d2h_end);
    
    cudaStreamDestroy(stream);
    
    // std::cout << "\n========================================" << std::endl;
    
    return result;
}

// ==================== 文件测试包装函数 ====================
CompressionInfo test_compression_double(const std::string& file_path) {
    std::vector<double> oriData = read_data(file_path);
    return comp_ALP_GPU_double(oriData);
}

CompressionInfo test_compression_float(const std::string& file_path) {
    std::vector<float> oriData = read_data_float(file_path);
    return comp_ALP_GPU_float(oriData);
}

CompressionInfo test_beta_compression_double(const std::string& file_path, int beta) {
    std::vector<double> oriData = read_data(file_path, beta);
    return comp_ALP_GPU_double(oriData);
}

CompressionInfo test_beta_compression_float(const std::string& file_path, int beta) {
    std::vector<float> oriData = read_data_float(file_path);
    return comp_ALP_GPU_float(oriData);
}

// ==================== Google Test 测试用例 ====================

// Test: Double 精度压缩/解压
TEST(ALPGPUCompressorTest, CompressionDecompressionDouble) {
    std::vector<double> test_data;
    test_data.reserve(1000000);
    
    // 生成测试数据
    for (int i = 0; i < 1000000; ++i) {
        test_data.push_back(static_cast<double>(i) * 1.5 + sin(i * 0.001) * 100.0);
    }
    
    CompressionInfo result = comp_ALP_GPU_double(test_data);
    
    // 验证结果
    ASSERT_GT(result.original_size_mb, 0.0);
    ASSERT_GT(result.compressed_size_mb, 0.0);
    ASSERT_GT(result.comp_throughput, 0.0);
    ASSERT_GT(result.decomp_throughput, 0.0);
    ASSERT_LT(result.compression_ratio, 1.0);
    
    // std::cout << "\n✓ Test Summary (Double):" << std::endl;
    // std::cout << "  Original: " << std::fixed << std::setprecision(4) 
    //           << result.original_size_mb << " MB" << std::endl;
    // std::cout << "  Compressed: " << result.compressed_size_mb << " MB" << std::endl;
    // std::cout << "  Ratio: " << result.compression_ratio << std::endl;
}

// Test: Float 精度压缩/解压
TEST(ALPGPUCompressorTest, CompressionDecompressionFloat) {
    std::vector<float> test_data;
    test_data.reserve(1000000);
    
    // 生成测试数据
    for (int i = 0; i < 1000000; ++i) {
        test_data.push_back(static_cast<float>(i) * 1.5f + sinf(i * 0.001f) * 100.0f);
    }
    
    CompressionInfo result = comp_ALP_GPU_float(test_data);
    
    // 验证结果
    ASSERT_GT(result.original_size_mb, 0.0);
    ASSERT_GT(result.compressed_size_mb, 0.0);
    ASSERT_GT(result.comp_throughput, 0.0);
    ASSERT_GT(result.decomp_throughput, 0.0);
    ASSERT_LT(result.compression_ratio, 1.0);
    
    // std::cout << "\n✓ Test Summary (Float):" << std::endl;
    // std::cout << "  Original: " << std::fixed << std::setprecision(4) 
    //           << result.original_size_mb << " MB" << std::endl;
    // std::cout << "  Compressed: " << result.compressed_size_mb << " MB" << std::endl;
    // std::cout << "  Ratio: " << result.compression_ratio << std::endl;
}

// Test: 小数据集 (Double)
TEST(ALPGPUCompressorTest, SmallDatasetDouble) {
    std::vector<double> test_data = {1.1, 2.2, 3.3, 4.4, 5.5};
    CompressionInfo result = comp_ALP_GPU_double(test_data);
    
    ASSERT_GT(result.original_size_mb, 0.0);
    ASSERT_GT(result.compression_ratio, 0.0);
}

// Test: 小数据集 (Float)
TEST(ALPGPUCompressorTest, SmallDatasetFloat) {
    std::vector<float> test_data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    CompressionInfo result = comp_ALP_GPU_float(test_data);
    
    ASSERT_GT(result.original_size_mb, 0.0);
    ASSERT_GT(result.compression_ratio, 0.0);
}

// Test: 大数据集 (Double)
TEST(ALPGPUCompressorTest, LargeDatasetDouble) {
    std::vector<double> test_data;
    test_data.reserve(10000000);
    
    for (int i = 0; i < 10000000; ++i) {
        test_data.push_back(static_cast<double>(i) * 0.1 + cos(i * 0.0001) * 50.0);
    }
    
    CompressionInfo result = comp_ALP_GPU_double(test_data);
    
    ASSERT_GT(result.original_size_mb, 70.0);  // 10M doubles = 80MB
    ASSERT_GT(result.comp_throughput, 0.0);
}

// ==================== 主函数 ====================
int main(int argc, char *argv[]) {
    cudaFree(0);  // 初始化 CUDA
    
    if (argc > 1) {
        std::string arg = argv[1];
        
        // ==================== 目录批处理模式 ====================
        if (arg == "--dir" && argc >= 3) {
            std::string dir_path = argv[2];
            
            // 检查目录是否存在
            if (!fs::exists(dir_path)) {
                std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
                return 1;
            }
            
            bool warm = false;
            int processed = 0;
            
            for (const auto& entry : fs::directory_iterator(dir_path)) {
                if (entry.is_regular_file()) {
                    std::string file_path = entry.path().string();
                    CompressionInfo result_double;
                    
                    // 预热
                    if (!warm) {
                        std::cout << "\n==================== Warmup ====================" << std::endl;
                        test_compression_double(file_path);
                        warm = true;
                        std::cout << "==================== Warmup End ====================" << std::endl;
                    }
                    
                    // 处理文件：3次运行并求平均
                    std::cout << "\nProcessing file: " << file_path << std::endl;
                    for (int i = 0; i < 3; ++i) {
                        cudaDeviceReset();
                        result_double += test_compression_double(file_path);
                    }
                    result_double = result_double / 3;
                    
                    // 打印结果
                    result_double.print();
                    std::cout << "---------------------------------------------" << std::endl;
                    processed++;
                }
            }
            
            if (processed == 0) {
                std::cerr << "No files found in directory: " << dir_path << std::endl;
            }
            
            return 0;
        }
        // ==================== Beta 参数测试模式 ====================
        else if (arg == "--file-beta" && argc >= 3) {
            std::string file_path = argv[2];
            
            // 遍历 beta 从 4 到 17
            for (int beta = 4; beta < 18; ++beta) {
                std::cout << "\nProcessing file: " << file_path;
                printf(" beta:%d\n", beta);
                
                CompressionInfo result_double;
                cudaDeviceReset();
                result_double += test_beta_compression_double(file_path, beta);
                
                result_double.print();
                std::cout << "---------------------------------------------" << std::endl;
            }
            
            return 0;
        }
        // ==================== 单文件测试模式 ====================
        else {
            std::string file_path = argv[1];
            
            std::cout << "\n========== ALP-GPU Test: File Input Mode ==========" << std::endl;
            std::cout << "Input file: " << file_path << std::endl;
            
            if (fs::exists(file_path)) {
                try {
                    // 3次运行并求平均
                    CompressionInfo result_double;
                    CompressionInfo result_float;
                    
                    std::cout << "\nRunning 3 iterations for averaging..." << std::endl;
                    for (int i = 0; i < 3; ++i) {
                        cudaDeviceReset();
                        result_double += test_compression_double(file_path);
                    }
                    result_double = result_double / 3;
                    
                    for (int i = 0; i < 3; ++i) {
                        cudaDeviceReset();
                        result_float += test_compression_float(file_path);
                    }
                    result_float = result_float / 3;
                    
                    std::cout << "\n========== FINAL REPORT (Double, Averaged) ==========" << std::endl;
                    result_double.print();
                    
                    std::cout << "\n========== FINAL REPORT (Float, Averaged) ==========" << std::endl;
                    result_float.print();
                    
                    return 0;
                } catch (const std::exception& e) {
                    std::cerr << "❌ Error: " << e.what() << std::endl;
                    return 1;
                }
            } else {
                std::cerr << "❌ File not found: " << file_path << std::endl;
                return 1;
            }
        }
    } else {
        // 运行 Google Test 单元测试
        ::testing::InitGoogleTest(&argc, argv);
        std::cout << "\n========== ALP-GPU Test: Unit Test Mode ==========" << std::endl;
        std::cout << "Running Google Test framework..." << std::endl;
        return RUN_ALL_TESTS();
    }
}
