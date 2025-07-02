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
    double compression_kernel_time_ms;
    double decompression_kernel_time_ms;
    double compression_total_time_ms;
    double decompression_total_time_ms;
    double compression_total_throughput_gbps;
    double decompression_total_throughput_gbps;
    bool data_integrity_ok;
    
    void print() const {
        std::cout << "\n=== Compression Metrics ===" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        // std::cout << "Input size:                    " << input_size << " bytes (" << input_size / 1024.0 / 1024.0 << " MB)" << std::endl;
        // std::cout << "Compressed size:               " << compressed_size << " bytes (" << compressed_size / 1024.0 / 1024.0 << " MB)" << std::endl;
        std::cout << "压缩率:             " << compression_ratio << std::endl;
        std::cout << "压缩核函数时间       " << compression_kernel_time_ms << " ms" << std::endl;
        std::cout << "解压核函数时间:     " << decompression_kernel_time_ms << " ms" << std::endl;
        std::cout << "压缩时间:        " << compression_total_time_ms << " ms" << std::endl;
        std::cout << "压缩吞吐量:  " << compression_total_throughput_gbps << " GB/s" << std::endl;
        std::cout << "解压时间:      " << decompression_total_time_ms << " ms" << std::endl;
        std::cout << "解压吞吐量:" << decompression_total_throughput_gbps << " GB/s" << std::endl;
        std::cout << "Data integrity:                " << (data_integrity_ok ? "✓ PASS" : "✗ FAIL") << std::endl;
    }
        // 加法运算符重载
    CompressionMetrics operator+(const CompressionMetrics& other) const {
        return {
            input_size + other.input_size,
            compressed_size + other.compressed_size,
            decompressed_size + other.decompressed_size,
            compression_ratio + other.compression_ratio,
            compression_kernel_time_ms + other.compression_kernel_time_ms,
            decompression_kernel_time_ms + other.decompression_kernel_time_ms,
            compression_total_time_ms + other.compression_total_time_ms,
            decompression_total_time_ms + other.decompression_total_time_ms,
            compression_total_throughput_gbps + other.compression_total_throughput_gbps,
            decompression_total_throughput_gbps + other.decompression_total_throughput_gbps,
            data_integrity_ok && other.data_integrity_ok // 只有当两者都有效时结果才有效
        };
    }
    
    // 加法赋值运算符重载
    CompressionMetrics& operator+=(const CompressionMetrics& other) {
        input_size += other.input_size;
        compressed_size += other.compressed_size;
        decompressed_size += other.decompressed_size;
        compression_ratio += other.compression_ratio;
        compression_kernel_time_ms += other.compression_kernel_time_ms;
        decompression_kernel_time_ms += other.decompression_kernel_time_ms;
        compression_total_time_ms += other.compression_total_time_ms;
        decompression_total_time_ms += other.decompression_total_time_ms;
        compression_total_throughput_gbps += other.compression_total_throughput_gbps;
        decompression_total_throughput_gbps += other.decompression_total_throughput_gbps;
        data_integrity_ok = data_integrity_ok && other.data_integrity_ok;
        return *this;
    }
    
    // 除法运算符重载 (除以整数)
    CompressionMetrics operator/(int divisor) const {
        if (divisor == 0) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        return {
            input_size / divisor,
            compressed_size / divisor,
            decompressed_size / divisor,
            compression_ratio / divisor,
            compression_kernel_time_ms / divisor,
            decompression_kernel_time_ms / divisor,
            compression_total_time_ms / divisor,
            decompression_total_time_ms / divisor,
            compression_total_throughput_gbps / divisor,
            decompression_total_throughput_gbps / divisor,
            data_integrity_ok // 布尔值保持不变
        };
    }
    
    // 除法运算符重载 (除以浮点数)
    CompressionMetrics operator/(double divisor) const {
        if (divisor == 0.0) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        return {
            static_cast<size_t>(input_size / divisor),
            static_cast<size_t>(compressed_size / divisor),
            static_cast<size_t>(decompressed_size / divisor),
            compression_ratio / divisor,
            compression_kernel_time_ms / divisor,
            decompression_kernel_time_ms / divisor,
            compression_total_time_ms / divisor,
            decompression_total_time_ms / divisor,
            compression_total_throughput_gbps / divisor,
            decompression_total_throughput_gbps / divisor,
            data_integrity_ok // 布尔值保持不变
        };
    }
    
};
CompressionInfo change(CompressionMetrics a)
{
    return CompressionInfo{
        a.input_size/1024.0/1024.0,
        a.compressed_size/1024.0/1024.0,
        a.compression_ratio,
        a.compression_kernel_time_ms,
        a.compression_total_time_ms,
        a.compression_total_throughput_gbps,
        a.decompression_kernel_time_ms,
        a.decompression_total_time_ms,
        a.decompression_total_throughput_gbps};
}
// Core Snappy compression/decompression function
CompressionMetrics test_snappy_compression_core(const void* input_data, size_t input_size) {
    CompressionMetrics metrics = {};
    metrics.input_size = input_size;
    
    // 创建CUDA流用于异步操作
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    // 创建CUDA事件
    cudaEvent_t comp_start, comp_kernel_start, comp_kernel_stop, comp_stop;
    cudaEvent_t decomp_start, decomp_kernel_start, decomp_kernel_stop, decomp_stop;
    CHECK_CUDA(cudaEventCreate(&comp_start));
    CHECK_CUDA(cudaEventCreate(&comp_kernel_start));
    CHECK_CUDA(cudaEventCreate(&comp_kernel_stop));
    CHECK_CUDA(cudaEventCreate(&comp_stop));
    CHECK_CUDA(cudaEventCreate(&decomp_start));
    CHECK_CUDA(cudaEventCreate(&decomp_kernel_start));
    CHECK_CUDA(cudaEventCreate(&decomp_kernel_stop));
    CHECK_CUDA(cudaEventCreate(&decomp_stop));

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
        // 定义最大chunk大小 (例如 64MB)
        const size_t max_chunk_size = 16 * 1024 * 1024; // 16MB
        const size_t num_chunks = (input_size + max_chunk_size - 1) / max_chunk_size;
        
        // std::cout << "Input size: " << input_size << " bytes" << std::endl;
        // std::cout << "Number of chunks: " << num_chunks << std::endl;
        // std::cout << "Max chunk size: " << max_chunk_size << " bytes" << std::endl;
        
        // ============ COMPRESSION PIPELINE ============
        // 记录压缩总时间开始
        CHECK_CUDA(cudaEventRecord(comp_start, stream));
        
        // === H2D for Compression ===
        CHECK_CUDA(cudaMalloc(&d_input, input_size));
        CHECK_CUDA(cudaMemcpyAsync(d_input, input_data, input_size, cudaMemcpyHostToDevice, stream));
        
        // 批量分配压缩所需的设备内存
        CHECK_CUDA(cudaMalloc(&d_input_ptrs, sizeof(void*) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_output_ptrs, sizeof(void*) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_input_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_output_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_actual_output_sizes, sizeof(size_t) * num_chunks));
        
        // 准备输入参数 - 分块处理
        std::vector<void*> h_input_ptrs(num_chunks);
        std::vector<size_t> h_input_sizes(num_chunks);
        
        for (size_t i = 0; i < num_chunks; ++i) {
            size_t offset = i * max_chunk_size;
            size_t chunk_size = std::min(max_chunk_size, input_size - offset);
            h_input_ptrs[i] = static_cast<char*>(d_input) + offset;
            h_input_sizes[i] = chunk_size;
        }
        
        CHECK_CUDA(cudaMemcpyAsync(d_input_ptrs, h_input_ptrs.data(), sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_input_sizes, h_input_sizes.data(), sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice, stream));

        // === 压缩设置 ===
        nvcompBatchedSnappyOpts_t opts{};
        
        size_t temp_bytes = 0;
        size_t max_output_size = 0;
        
        // 获取压缩所需的临时空间和最大输出大小 - 使用最大chunk大小
        nvcompStatus_t status = nvcompBatchedSnappyCompressGetTempSize(num_chunks, max_chunk_size, opts, &temp_bytes);
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get compression temp size: " + std::to_string(status));
        }
        
        status = nvcompBatchedSnappyCompressGetMaxOutputChunkSize(max_chunk_size, opts, &max_output_size);
        if (status != nvcompSuccess) {
            throw std::runtime_error("Failed to get max output size: " + std::to_string(status));
        }
        
        // std::cout << "Max output size per chunk: " << max_output_size << " bytes" << std::endl;
        // std::cout << "Temp bytes needed: " << temp_bytes << " bytes" << std::endl;
        
        // 分配压缩缓冲区 - 为所有chunks分配空间
        size_t total_compressed_buffer_size = max_output_size * num_chunks;
        CHECK_CUDA(cudaMalloc(&d_compressed, total_compressed_buffer_size));
        if (temp_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_temp_compress, temp_bytes));
        }
        
        // 设置输出参数 - 为每个chunk分配输出空间
        std::vector<void*> h_output_ptrs(num_chunks);
        std::vector<size_t> h_output_sizes(num_chunks);
        
        for (size_t i = 0; i < num_chunks; ++i) {
            h_output_ptrs[i] = static_cast<char*>(d_compressed) + i * max_output_size;
            h_output_sizes[i] = max_output_size;
        }
        
        CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_output_ptrs.data(), sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_output_sizes, h_output_sizes.data(), sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice, stream));
        
        // 确保所有内存拷贝完成
        // CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // === COMPRESSION KERNEL ===
        CHECK_CUDA(cudaEventRecord(comp_kernel_start, stream));
        
        status = nvcompBatchedSnappyCompressAsync(
            d_input_ptrs, d_input_sizes,
            max_output_size, num_chunks,
            d_temp_compress, temp_bytes,
            d_output_ptrs, d_actual_output_sizes,
            opts, stream);
        
        CHECK_CUDA(cudaEventRecord(comp_kernel_stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        if (status != nvcompSuccess) {
            throw std::runtime_error("Compression failed with status: " + std::to_string(status));
        }
        
        // === D2H for Compression ===
        // 获取实际压缩大小
        std::vector<size_t> h_actual_output_sizes(num_chunks);
        CHECK_CUDA(cudaMemcpyAsync(h_actual_output_sizes.data(), d_actual_output_sizes, sizeof(size_t) * num_chunks, cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // 计算总压缩大小
        metrics.compressed_size = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            metrics.compressed_size += h_actual_output_sizes[i];
            // std::cout << "Chunk " << i << " compressed size: " << h_actual_output_sizes[i] << " bytes" << std::endl;
        }
        
        // 将压缩数据拷贝到主机（完整的D2H）
        std::vector<uint8_t> compressed_host_data(metrics.compressed_size);
        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            CHECK_CUDA(cudaMemcpyAsync(
                compressed_host_data.data() + compressed_offset,
                h_output_ptrs[i],
                h_actual_output_sizes[i],
                cudaMemcpyDeviceToHost,
                stream));
            compressed_offset += h_actual_output_sizes[i];
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // 记录压缩总时间结束
        CHECK_CUDA(cudaEventRecord(comp_stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // ============ DECOMPRESSION PIPELINE ============
        // 记录解压缩总时间开始
        CHECK_CUDA(cudaEventRecord(decomp_start, stream));
        
        // === H2D for Decompression ===
        // 为解压缩分配新的设备内存并拷贝压缩数据
        void* d_compressed_for_decomp = nullptr;
        CHECK_CUDA(cudaMalloc(&d_compressed_for_decomp, metrics.compressed_size));
        CHECK_CUDA(cudaMemcpyAsync(d_compressed_for_decomp, compressed_host_data.data(), metrics.compressed_size, cudaMemcpyHostToDevice, stream));
        
        // 分配解压缩相关的设备内存
        CHECK_CUDA(cudaMalloc(&d_decomp_output_ptrs, sizeof(void*) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_decomp_output_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_actual_decomp_sizes, sizeof(size_t) * num_chunks));
        CHECK_CUDA(cudaMalloc(&d_decomp_statuses, sizeof(nvcompStatus_t) * num_chunks));
        
        // 重新设置压缩数据指针 - 分块指向
        std::vector<void*> h_compressed_ptrs(num_chunks);
        compressed_offset = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            h_compressed_ptrs[i] = static_cast<char*>(d_compressed_for_decomp) + compressed_offset;
            compressed_offset += h_actual_output_sizes[i];
        }
        
        CHECK_CUDA(cudaMemcpyAsync(d_output_ptrs, h_compressed_ptrs.data(), sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_actual_output_sizes, h_actual_output_sizes.data(), sizeof(size_t) * num_chunks, cudaMemcpyHostToDevice, stream));
        
        // === 解压缩设置 ===
        size_t decomp_temp_bytes = 0;
        status = nvcompBatchedSnappyDecompressGetTempSize(num_chunks, max_chunk_size, &decomp_temp_bytes);
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
        
        // 分配解压缩缓冲区
        size_t total_decomp_size = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            total_decomp_size += h_decomp_output_sizes[i];
        }
        
        CHECK_CUDA(cudaMalloc(&d_decompressed, total_decomp_size));
        if (decomp_temp_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_temp_decompress, decomp_temp_bytes));
        }
        
        // 设置解压缩输出指针
        std::vector<void*> h_decomp_output_ptrs(num_chunks);
        size_t decomp_offset = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            h_decomp_output_ptrs[i] = static_cast<char*>(d_decompressed) + decomp_offset;
            decomp_offset += h_decomp_output_sizes[i];
        }
        
        CHECK_CUDA(cudaMemcpyAsync(d_decomp_output_ptrs, h_decomp_output_ptrs.data(), sizeof(void*) * num_chunks, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // === DECOMPRESSION KERNEL ===
        CHECK_CUDA(cudaEventRecord(decomp_kernel_start, stream));
        
        status = nvcompBatchedSnappyDecompressAsync(
            d_output_ptrs, d_actual_output_sizes,
            d_decomp_output_sizes, d_actual_decomp_sizes,
            num_chunks,
            d_temp_decompress, decomp_temp_bytes,
            d_decomp_output_ptrs,
            d_decomp_statuses,  
            stream);
        
        CHECK_CUDA(cudaEventRecord(decomp_kernel_stop, stream));
        // CHECK_CUDA(cudaStreamSynchronize(stream));
        
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
        
        metrics.decompressed_size = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            metrics.decompressed_size += h_actual_decomp_sizes[i];
        }
        
        // === D2H for Decompression ===
        // 将解压缩数据拷贝到主机（完整的D2H）
        std::vector<uint8_t> decompressed_host_data(metrics.decompressed_size);
        size_t host_offset = 0;
        for (size_t i = 0; i < num_chunks; ++i) {
            CHECK_CUDA(cudaMemcpyAsync(
                decompressed_host_data.data() + host_offset,
                h_decomp_output_ptrs[i],
                h_actual_decomp_sizes[i],
                cudaMemcpyDeviceToHost,
                stream));
            host_offset += h_actual_decomp_sizes[i];
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // 记录解压缩总时间结束
        CHECK_CUDA(cudaEventRecord(decomp_stop, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
        
        // ============ 计算性能指标 ============
        // 压缩核函数时间
        float compression_kernel_time;
        CHECK_CUDA(cudaEventElapsedTime(&compression_kernel_time, comp_kernel_start, comp_kernel_stop));
        metrics.compression_kernel_time_ms = compression_kernel_time;
        
        // 解压缩核函数时间
        float decompression_kernel_time;
        CHECK_CUDA(cudaEventElapsedTime(&decompression_kernel_time, decomp_kernel_start, decomp_kernel_stop));
        metrics.decompression_kernel_time_ms = decompression_kernel_time;
        
        // 压缩总时间（H2D + KERNEL + D2H）
        float compression_total_time;
        CHECK_CUDA(cudaEventElapsedTime(&compression_total_time, comp_start, comp_stop));
        metrics.compression_total_time_ms = compression_total_time;
        
        // 解压缩总时间（H2D + KERNEL + D2H）
        float decompression_total_time;
        CHECK_CUDA(cudaEventElapsedTime(&decompression_total_time, decomp_start, decomp_stop));
        metrics.decompression_total_time_ms = decompression_total_time;
        
        // 压缩率（小于1表示压缩效果）
        metrics.compression_ratio = (double)metrics.compressed_size / input_size;
        
        // 吞吐量计算（GB/s）
        metrics.compression_total_throughput_gbps = (input_size / 1024.0 / 1024.0 / 1024.0) / (compression_total_time / 1000.0);
        metrics.decompression_total_throughput_gbps = (metrics.decompressed_size / 1024.0 / 1024.0 / 1024.0) / (decompression_total_time / 1000.0);
        
        // === 数据完整性检查 ===
        if (metrics.decompressed_size == input_size) {
            metrics.data_integrity_ok = (std::memcmp(input_data, decompressed_host_data.data(), input_size) == 0);
        } else {
            metrics.data_integrity_ok = false;
        }
        
        // 清理解压缩专用内存
        if (d_compressed_for_decomp) cudaFree(d_compressed_for_decomp);
        
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
    CHECK_CUDA(cudaEventDestroy(comp_start));
    CHECK_CUDA(cudaEventDestroy(comp_kernel_start));
    CHECK_CUDA(cudaEventDestroy(comp_kernel_stop));
    CHECK_CUDA(cudaEventDestroy(comp_stop));
    CHECK_CUDA(cudaEventDestroy(decomp_start));
    CHECK_CUDA(cudaEventDestroy(decomp_kernel_start));
    CHECK_CUDA(cudaEventDestroy(decomp_kernel_stop));
    CHECK_CUDA(cudaEventDestroy(decomp_stop));
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

void test_compression(const std::string& file_path,bool show=1) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    auto patterned_data = convert(oriData);
    auto metrics = test_snappy_compression_core(patterned_data.data(), patterned_data.size());
    for(int i=0;i<2;i++)
    {
        metrics += test_snappy_compression_core(patterned_data.data(), patterned_data.size());
    }
    metrics=metrics/3;

    if(show){
        CompressionInfo ans=change(metrics);
        ans.print();
    }
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
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "COMPARISON SUMMARY" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << std::setw(15) << "Test Type" << std::setw(12) << "Ratio" << std::setw(15) << "Comp Total(GB/s)" << std::setw(15) << "Decomp Total(GB/s)" << std::setw(15) << "Comp Kernel(ms)" << std::setw(15) << "Decomp Kernel(ms)" << std::endl;
    std::cout << std::string(95, '-') << std::endl;
    std::cout << std::setw(15) << "Random" << std::setw(12) << metrics1.compression_ratio << std::setw(15) << metrics1.compression_total_throughput_gbps << std::setw(15) << metrics1.decompression_total_throughput_gbps << std::setw(15) << metrics1.compression_kernel_time_ms << std::setw(15) << metrics1.decompression_kernel_time_ms << std::endl;
    std::cout << std::setw(15) << "Patterned" << std::setw(12) << metrics2.compression_ratio << std::setw(15) << metrics2.compression_total_throughput_gbps << std::setw(15) << metrics2.decompression_total_throughput_gbps << std::setw(15) << metrics2.compression_kernel_time_ms << std::setw(15) << metrics2.decompression_kernel_time_ms << std::endl;
    std::cout << std::setw(15) << "Scientific" << std::setw(12) << metrics3.compression_ratio << std::setw(15) << metrics3.compression_total_throughput_gbps << std::setw(15) << metrics3.decompression_total_throughput_gbps << std::setw(15) << metrics3.compression_kernel_time_ms << std::setw(15) << metrics3.decompression_kernel_time_ms << std::endl;
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
    
    bool warmup = false;
    // 读取数据并测试压缩和解压
    std::string dir_path = "../test/data/tsbs_csv"; 
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            if (!warmup) {
                warmup = true;
                std::cout << "====================WARMUP==========================" << std::endl;
                test_compression(file_path);
                std::cout << "====================WARMUP_END======================" << std::endl;
            }
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_compression(file_path);
            std::cout << "---------------------------------------------" << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
    
    cudaFree(0);
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {

        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        bool warm=0;
        int processed = 0;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                if(!warm)
                {
                    std::cout << "\n-------------------warm-------------------------- " << file_path << std::endl;
                    test_compression(file_path,0);
                    warm=1;
                    std::cout << "-------------------warm_end------------------------" << std::endl;
                }
                std::cout << "\nProcessing file: " << file_path << std::endl;
                test_compression(file_path);
                std::cout << "---------------------------------------------" << std::endl;
                processed++;
            }
        }
        
        if (processed == 0) {
            std::cerr << "No files found in directory: " << dir_path << std::endl;
        }
    }
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}