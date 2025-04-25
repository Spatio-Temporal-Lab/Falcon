#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <nvcomp/lz4.hpp>  // 引入 nvcomp LZ4 库
#include <cassert>
#include <nvcomp/lz4.h>
#include <cuda_runtime.h>
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;


void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);

    // 获取数据大小
    size_t in_bytes = oriData.size() * sizeof(double);
    
    // 将数据复制到char类型的数组中，因为LZ4压缩处理的是字节数组
    char* input_data = reinterpret_cast<char*>(oriData.data());

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 计算块大小
    const size_t chunk_size = 65536;
    const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;

    // 分配CUDA内存
    char* device_input_data;
    cudaMalloc(&device_input_data, in_bytes);
    cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream);

    size_t* host_uncompressed_bytes;
    cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (i + 1 < batch_size) {
            host_uncompressed_bytes[i] = chunk_size;
        } else {
            host_uncompressed_bytes[i] = in_bytes - (chunk_size * i);
        }
    }

    // 设置每个块的指针
    void** host_uncompressed_ptrs;
    cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_ptrs[i] = device_input_data + chunk_size * i;
    }

    size_t* device_uncompressed_bytes;
    void** device_uncompressed_ptrs;
    cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
    cudaMalloc(&device_uncompressed_ptrs, sizeof(void*) * batch_size);
    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);

    // 分配临时空间
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void* device_temp_ptr;
    cudaMalloc(&device_temp_ptr, temp_bytes);

    // 获取每个块的最大输出大小
    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    // 分配压缩后的输出空间
    void** host_compressed_ptrs;
    cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        cudaMalloc(&host_compressed_ptrs[i], max_out_bytes);
    }

    void** device_compressed_ptrs;
    cudaMalloc(&device_compressed_ptrs, sizeof(void*) * batch_size);
    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);

    size_t* device_compressed_bytes;
    cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);

    // 记录压缩开始时间
    auto start_compress = std::chrono::high_resolution_clock::now();

    // 压缩数据
    nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        chunk_size,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        device_compressed_ptrs,
        device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        stream);

    if (comp_res != nvcompSuccess) {
        std::cerr << "Compression failed!" << std::endl;
        assert(comp_res == nvcompSuccess);
    }

    // 记录压缩结束时间
    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;
    std::cout << "Compression Time: " << compress_duration.count() << " seconds." << std::endl;

    // 解压数据

    // 获取解压后的大小
    nvcompBatchedLZ4GetDecompressSizeAsync(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        batch_size,
        stream);

    // 分配解压临时缓冲区
    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes);
    void* device_decomp_temp;
    cudaMalloc(&device_decomp_temp, decomp_temp_bytes);

    nvcompStatus_t* device_statuses;
    cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * batch_size);

    size_t* device_actual_uncompressed_bytes;
    cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t) * batch_size);

    // 记录解压开始时间
    auto start_decompress = std::chrono::high_resolution_clock::now();

    // 解压数据
    nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        device_actual_uncompressed_bytes,
        batch_size,
        device_decomp_temp,
        decomp_temp_bytes,
        device_uncompressed_ptrs,
        device_statuses,
        stream);

    if (decomp_res != nvcompSuccess) {
        std::cerr << "Decompression failed!" << std::endl;
        assert(decomp_res == nvcompSuccess);
    }

    // 记录解压结束时间
    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;
    std::cout << "Decompression Time: " << decompress_duration.count() << " seconds." << std::endl;
    

    // 1. 同步流并获取实际压缩大小
    cudaStreamSynchronize(stream);
    size_t total_compressed = 0;
    size_t* host_compressed_bytes = new size_t[batch_size];
    cudaMemcpy(host_compressed_bytes, device_compressed_bytes, 
            sizeof(size_t)*batch_size, cudaMemcpyDeviceToHost);

    for (size_t i=0; i<batch_size; ++i) {
        total_compressed += host_compressed_bytes[i];
    }
    
    // 解压后添加校验代码
    char* reconstructed = new char[in_bytes];
    cudaMemcpyAsync(reconstructed, device_input_data, in_bytes, 
                cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    if (memcmp(input_data, reconstructed, in_bytes) != 0) {
        std::cerr << "Data mismatch!" << std::endl;
    }

    // 2. 计算真实压缩率
    double compression_ratio = total_compressed / static_cast<double>(in_bytes) ;
    std::cout << "Compression Ratio: " << compression_ratio << std::endl;
    
    // 清理CUDA资源
    cudaStreamSynchronize(stream);
    cudaFree(device_input_data);
    cudaFree(device_uncompressed_bytes);
    cudaFree(device_uncompressed_ptrs);
    cudaFree(device_temp_ptr);
    cudaFree(device_compressed_ptrs);
    cudaFree(device_compressed_bytes);
    cudaFree(device_decomp_temp);
    cudaFree(device_statuses);
    cudaFree(device_actual_uncompressed_bytes);
}


// Google Test 测试用例
TEST(LZ4CompressorTest, CompressionDecompression) {
    // 读取数据并测试压缩和解压
    std::string dir_path = "../test/data/big"; 
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
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
