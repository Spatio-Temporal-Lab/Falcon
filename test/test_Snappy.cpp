#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <random>
#include <iomanip>
#include "data/dataset_utils.hpp"
// nvcomp headers
#include "nvcomp/snappy.h"
#include "nvcomp.h"
#include <gtest/gtest.h>
// CUDA headers
#include <cuda_runtime.h>
#include <filesystem>
namespace fs = std::filesystem;

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Performance metrics structure
struct CompressionMetrics {
    size_t input_size;
    size_t compressed_size;
    size_t decompressed_size;
    double compression_ratio;
    double compression_all_time_ms;
    double decompression_all_time_ms;
    double compression_core_time_ms;
    double decompression_core_time_ms;
    double compression_throughput_mbps;
    double decompression_throughput_mbps;
    bool data_integrity_ok;
    
    void print() const {
        std::cout << "\n=== Compression Metrics ===" << std::endl;
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Input size:           " << input_size << " bytes (" << input_size / 1024.0 / 1024.0 << " MB)" << std::endl;
        std::cout << "Compressed size:      " << compressed_size << " bytes (" << compressed_size / 1024.0 / 1024.0 << " MB)" << std::endl;
        std::cout << "Compression ratio:    " << compression_ratio << ":1" << std::endl;
        std::cout << "Compression core time:     " << compression_core_time_ms << " ms" << std::endl;
        std::cout << "Decompression core time:   " << decompression_core_time_ms << " ms" << std::endl;
        std::cout << "Compression all time:     " << compression_all_time_ms << " ms" << std::endl;
        std::cout << "Decompression all time:   " << decompression_all_time_ms << " ms" << std::endl;
        std::cout << "Compression speed:    " << compression_throughput_mbps << " MB/s" << std::endl;
        std::cout << "Decompression speed:  " << decompression_throughput_mbps << " MB/s" << std::endl;
        std::cout << "Data integrity:       " << (data_integrity_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    }
};

// Core Snappy compression/decompression function
CompressionMetrics test_snappy_compression_core(const void* input_data, size_t input_size) {
    CompressionMetrics metrics = {};
    metrics.input_size = input_size;
    
    // 创建CUDA流用于异步操作
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // 创建CUDA事件
    cudaEvent_t global_start, global_stop, start, stop;
    CHECK_CUDA(cudaEventCreate(&global_start));
    CHECK_CUDA(cudaEventCreate(&global_stop));
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Device memory pointers
    void* d_input = nullptr;
    void* d_compressed = nullptr;
    void* d_decompressed = nullptr;
    void* d_temp_compress = nullptr;
    void* d_temp_decompress = nullptr;
    
    // Batch operation arrays
    void** d_input_ptrs = nullptr;
    void** d_output_ptrs = nullptr;
    void** d_decomp_output_ptrs = nullptr;
    size_t* d_input_sizes = nullptr;
    size_t* d_output_sizes = nullptr;
    size_t* d_actual_output_sizes = nullptr;
    size_t* d_decomp_output_sizes = nullptr;
    size_t* d_actual_decomp_sizes = nullptr;
    nvcompStatus_t* d_decomp_statuses = nullptr;
    
    try {
        const size_t num_chunks = 1;
        
        // 记录全局开始时间
        CHECK_CUDA(cudaEventRecord(global_start, stream));
        
        // === SETUP PHASE ===
        // 使用cudaMallocAsync for better performance (if available)
        CHECK_CUDA(cudaMalloc(&d_input, input_size));
        CHECK_CUDA(cudaMemcpyAsync(d_input, input_data, input_size, cudaMemcpyHostToDevice, stream));
        
        // 批量分配所有需要的设备内存
        CHECK_CUDA(cudaMalloc(&d_input_ptrs, sizeof(void*) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_output_ptrs, sizeof(void*) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_decomp_output_ptrs, sizeof(void*) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_input_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_output_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_actual_output_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_decomp_output_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_actual_decomp_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_decomp_statuses, sizeof(nvcompStatus_t) * num_chunks));
        
        // 准备输入参数
        std::vector<void*> h_input_ptrs = {d_input};
        std::vector<size_t> h_input_sizes = {input_size};
        CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(), sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(), sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice, stream));

        // === COMPRESSION SETUP ===
        nvcompBatchedSnappyOpts_t opts{};
        // 设置合适的选项 - 对于nvcomp 2.2，可以尝试设置这些优化选项
        // opts.chunk_size = 1 << 16; // 64KB chunks，根据数据大小调整
        
        size_t temp_bytes = 0;
        size_t max_output_size = 0;
        
        // 获取压缩所需的临时空间和最大输出大小
        nvcompStatus_t status = nvcompBatchedSnappyCompressGetTempSize(num_chunks, input_size, opts, &temp_bytes);
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get compression temp size: " + std::to_string(status));
        }
        
        status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(input_size, opts, &max_output_size);
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get max output size: " + std::to_string(status));
        }
        
        // 分配压缩缓冲区
        CHECK_CUDA(cudaMalloc(&d_compressed, max_output_size));
        if (temp_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_temp_compress, temp_bytes));
        }
        
        // 设置输出参数
        std::vector<void*> h_output_ptrs = {d_compressed};  
        std::vector<size_t> h_output_sizes = {max_output_size};
        CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(), sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_output_sizes, h_output_sizes.data(), sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice, stream));
        
        // 确保所有内存拷贝完成
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // === COMPRESSION ===
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        status = nvcompBatchedSnappyCompressAsync(
            d_input_ptrs, d_input_sizes,
            max_output_size, num_chunks,
            d_temp_compress, temp_bytes,
            d_output_ptrs, d_actual_output_sizes,
            opts, stream);  // 使用stream而不是0
        
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));  // 等待压缩完成
        
        if (status != nvcompSuccess) {
            throw std::runtime_error("Compression failed with status: " + std::to_string(status));
        }
        
        // 获取实际压缩大小
        std::vector<size_t> h_actual_output_sizes(num_chunks);
        CHECK_CUDA(cudaMemcpyAsync(h_actual_output_sizes.data(), d_actual_output_sizes, sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        metrics.compressed_size = h_actual_output_sizes[0];
        
        // 计算压缩性能
        float compression_core_time;
        CHECK_CUDA(cudaEventElapsedTime(&compression_core_time, start, stop));
        metrics.compression_core_time_ms = compression_core_time;
        metrics.compression_ratio = (double)input_size / metrics.compressed_size;
        metrics.compression_throughput_mbps = (input_size / 1024.0 / 1024.0) / (compression_core_time / 1000.0);
        
        // === DECOMPRESSION SETUP ===
        size_t decomp_temp_bytes = 0;
        status = nvcompBatchedSnappyDecompressGetTempSize(num_chunks, max_output_size, &decomp_temp_bytes);
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get decompression temp size: " + std::to_string(status));
        }
        
        // 获取解压缩大小
        status = nvcompBatchedSnappyGetDecompressSizeAsync(d_output_ptrs, d_actual_output_sizes, d_decomp_output_sizes, num_chunks, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get decompression output size: " + std::to_string(status));
        }
        
        std::vector<size_t> h_decomp_output_sizes(num_chunks);
        CHECK_CUDA(cudaMemcpyAsync(h_decomp_output_sizes.data(), d_decomp_output_sizes, sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        size_t decomp_output_bytes = h_decomp_output_sizes[0];
        
        // 分配解压缩缓冲区
        CHECK_CUDA(cudaMalloc(&d_decompressed, decomp_output_bytes));
        if (decomp_temp_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_temp_decompress, decomp_temp_bytes));
        }
        
        std::vector<void*> h_decomp_output_ptrs = {d_decompressed};
        CHECK_CUDA(cudaMemcpyAsync(d_decomp_output_ptrs, h_decomp_output_ptrs.data(), sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // === DECOMPRESSION ===
        CHECK_CUDA(cudaEventRecord(start, stream));
        
        status = nvcompBatchedSnappyDecompressAsync(
            d_output_ptrs, d_actual_output_sizes,
            d_decomp_output_sizes, d_actual_decomp_sizes,
            num_chunks,
            d_temp_decompress, decomp_temp_bytes,
            d_decomp_output_ptrs,
            d_decomp_statuses,  
            stream);  // 使用stream
        
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));  // 等待解压缩完成
        
        if (status != nvcompSuccess) {
            throw std::runtime_error("Decompression failed with status: " + std::to_string(status));
        }
        
        // 检查解压缩状态
        std::vector<nvcompStatus_t> h_decomp_statuses(num_chunks);
        CHECK_CUDA(cudaMemcpyAsync(h_decomp_statuses.data(), d_decomp_statuses, sizeof(nvcompStatus_t) * num_chunks, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        for (size_t i = 0; i < num_chunks; ++i) {
            if (h_decomp_statuses[i] != nvcompSuccess) {
                throw std::runtime_error("Decompression failed for chunk " + std::to_string(i) + " with status: " + std::to_string(h_decomp_statuses[i]));
            }
        }

        std::vector<size_t> h_actual_decomp_sizes(num_chunks);
        CHECK_CUDA(cudaMemcpyAsync(h_actual_decomp_sizes.data(), d_actual_decomp_sizes, sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        metrics.decompressed_size = h_actual_decomp_sizes[0];
        
        // 计算解压缩性能
        float decompression_core_time;
        CHECK_CUDA(cudaEventElapsedTime(&decompression_core_time, start, stop));
        metrics.decompression_core_time_ms = decompression_core_time;
        metrics.decompression_throughput_mbps = (metrics.decompressed_size / 1024.0 / 1024.0) / (decompression_core_time / 1000.0);
        
        // 记录全局结束时间
        CHECK_CUDA(cudaEventRecord(global_stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        float total_time;
        CHECK_CUDA(cudaEventElapsedTime(&total_time, global_start, global_stop));
        metrics.compression_all_time_ms = total_time;
        metrics.decompression_all_time_ms = total_time;  // 这里应该分别记录
        
        // === DATA INTEGRITY CHECK ===
        if (metrics.decompressed_size == input_size) {
            std::vector<uint8_t> output_data(metrics.decompressed_size);
            CHECK_CUDA(cudaMemcpyAsync(output_data.data(), d_decompressed, metrics.decompressed_size, cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            metrics.data_integrity_ok = (std::memcmp(input_data, output_data.data(), input_size) == 0);
        } else {
            metrics.data_integrity_ok = false;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in compression core: " << e.what() << std::endl;
        metrics.data_integrity_ok = false;
    }
    
    // 清理设备内存
    if (d_input) cudaFree(d_input);
    if (d_compressed) cudaFree(d_compressed);
    if (d_decompressed) cudaFree(d_decompressed);
    if (d_temp_compress) cudaFree(d_temp_compress);
    if (d_temp_decompress) cudaFree(d_temp_decompress);
    if (d_input_ptrs) cudaFree(d_input_ptrs);
    if (d_output_ptrs) cudaFree(d_output_ptrs);
    if (d_decomp_output_ptrs) cudaFree(d_decomp_output_ptrs);
    if (d_input_sizes) cudaFree(d_input_sizes);
    if (d_output_sizes) cudaFree(d_output_sizes);
    if (d_actual_output_sizes) cudaFree(d_actual_output_sizes);
    if (d_decomp_output_sizes) cudaFree(d_decomp_output_sizes);
    if (d_actual_decomp_sizes) cudaFree(d_actual_decomp_sizes);
    if (d_decomp_statuses) cudaFree(d_decomp_statuses);
    
    // 清理CUDA对象
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(global_start));
    CHECK_CUDA(cudaEventDestroy(global_stop));
    CHECK_CUDA(cudaStreamDestroy(stream));
    
    return metrics;
}

// Test data generators
std::vector<uint8_t> generate_random_float_data(size_t num_floats) {
    std::vector<float> float_data(num_floats);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1000.0f, 1000.0f);
    
    for (size_t i = 0; i < num_floats; ++i) {
        float_data[i] = dis(gen);
    }
    
    std::vector<uint8_t> byte_data(num_floats * sizeof(float));
    std::memcpy(byte_data.data(), float_data.data(), byte_data.size());
    return byte_data;
}

std::vector<uint8_t> generate_patterned_float_data(size_t num_floats) {
    std::vector<float> float_data(num_floats);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    
    for (size_t i = 0; i < num_floats; ++i) {
        if (i % 100 < 30) {
            float_data[i] = 0.0f;  // 30% zeros
        } else if (i % 100 < 50) {
            float_data[i] = 1.0f;  // 20% ones
        } else if (i % 100 < 70) {
            float_data[i] = static_cast<float>(i % 10);  // 20% small integers
        } else {
            float_data[i] = dis(gen);  // 30% random
        }
    }
    
    std::vector<uint8_t> byte_data(num_floats * sizeof(float));
    std::memcpy(byte_data.data(), float_data.data(), byte_data.size());
    return byte_data;
}

std::vector<uint8_t> generate_scientific_float_data(size_t num_floats) {
    std::vector<float> float_data(num_floats);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform_dist(1e-6f, 1e6f);
    
    for (size_t i = 0; i < num_floats; ++i) {
        if (i % 3 == 0) {
            float_data[i] = normal_dist(gen);  // Normal distribution
        } else if (i % 3 == 1) {
            float_data[i] = uniform_dist(gen);  // Uniform large range
        } else {
            float_data[i] = std::sin(i * 0.01f) * 100.0f;  // Sinusoidal pattern
        }
    }
    
    std::vector<uint8_t> byte_data(num_floats * sizeof(float));
    std::memcpy(byte_data.data(), float_data.data(), byte_data.size());
    return byte_data;
}

std::vector<uint8_t> convert(const std::vector<double>& oriData) {
    std::vector<uint8_t> byte_data(oriData.size() * sizeof(double));
    std::memcpy(byte_data.data(), oriData.data(), byte_data.size());
    return byte_data;
}

void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    auto patterned_data = convert(oriData);

    auto metrics = test_snappy_compression_core(patterned_data.data(), patterned_data.size());
    metrics.print();
}

void run_float_compression_tests() {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "FLOAT DATA COMPRESSION TESTS" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    const size_t num_floats = 1024 * 1024;  // 1M floats = 4MB
    
    // Test 1: Random float data
    std::cout << "\n[TEST 1] Random Float Data (4MB)" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    auto random_data = generate_random_float_data(num_floats);
    auto metrics1 = test_snappy_compression_core(random_data.data(), random_data.size());
    metrics1.print();
    
    // Test 2: Patterned float data
    std::cout << "\n[TEST 2] Patterned Float Data (30% zeros, 20% ones, 20% small ints, 30% random)" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    auto patterned_data = generate_patterned_float_data(num_floats);
    auto metrics2 = test_snappy_compression_core(patterned_data.data(), patterned_data.size());
    metrics2.print();
    
    // Test 3: Scientific float data
    std::cout << "\n[TEST 3] Scientific Float Data (normal dist + uniform + sinusoidal)" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    auto scientific_data = generate_scientific_float_data(num_floats);
    auto metrics3 = test_snappy_compression_core(scientific_data.data(), scientific_data.size());
    metrics3.print();
    
    // Summary comparison
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "COMPARISON SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(15) << "Test Type" << std::setw(12) << "Ratio" << std::setw(12) << "Comp(MB/s)" << std::setw(12) << "Decomp(MB/s)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::setw(15) << "Random" << std::setw(12) << metrics1.compression_ratio << std::setw(12) << metrics1.compression_throughput_mbps << std::setw(12) << metrics1.decompression_throughput_mbps << std::endl;
    std::cout << std::setw(15) << "Patterned" << std::setw(12) << metrics2.compression_ratio << std::setw(12) << metrics2.compression_throughput_mbps << std::setw(12) << metrics2.decompression_throughput_mbps << std::endl;
    std::cout << std::setw(15) << "Scientific" << std::setw(12) << metrics3.compression_ratio << std::setw(12) << metrics3.compression_throughput_mbps << std::setw(12) << metrics3.decompression_throughput_mbps << std::endl;
}

// Google Test 测试用例
TEST(SnappyCompressorTest, CompressionDecompression) {
    // 初始化CUDA设备
    int device_count;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    
    if (device_count == 0) {
        GTEST_SKIP() << "No CUDA devices found!";
        return;
    }
    
    // 显示GPU信息
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    
    bool warmup=0;
    // 读取数据并测试压缩和解压
    std::string dir_path = "../test/data/temp"; 
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            if(!warmup)
            {
                warmup=1;
                std::cout << "====================warmup==========================" << std::endl;
                test_compression(file_path);
                std::cout << "====================warmup_end=========================" << std::endl;
            }
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_compression(file_path);
            std::cout << "---------------------------------------------" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}