#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <nvcomp/gdeflate.hpp>
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;


void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    size_t in_bytes = oriData.size() * sizeof(double);
    if (in_bytes == 0) {
        std::cerr << "Error: Empty file or read failure: " << file_path << std::endl;
        return;
    }

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 分配设备内存
    uint8_t* device_input_data;
    cudaMalloc(&device_input_data, in_bytes);
    cudaMemcpyAsync(device_input_data, oriData.data(), in_bytes, cudaMemcpyHostToDevice, stream);

    // 创建 GDeflate 管理器 - 使用正确的选项结构
    const size_t chunk_size = 65536;
    nvcompBatchedGdeflateOpts_t format_opts = {0}; // 默认算法
    nvcomp::GdeflateManager manager{chunk_size, format_opts, stream};

    // 配置压缩
    nvcomp::CompressionConfig comp_config = manager.configure_compression(in_bytes);
    
    // 分配压缩输出缓冲区
    uint8_t* device_compressed_data;
    cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size);

    // 记录压缩时间
    auto start_compress = std::chrono::high_resolution_clock::now();
    manager.compress(device_input_data, device_compressed_data, comp_config);
    cudaStreamSynchronize(stream);
    auto end_compress = std::chrono::high_resolution_clock::now();
    double compress_time = std::chrono::duration<double>(end_compress - start_compress).count();

    // 获取压缩后大小
    size_t comp_out_bytes = manager.get_compressed_output_size(device_compressed_data);

    // 配置解压
    nvcomp::DecompressionConfig decomp_config = manager.configure_decompression(device_compressed_data);
    
    // 分配解压输出缓冲区
    uint8_t* device_decompressed_data;
    cudaMalloc(&device_decompressed_data, decomp_config.decomp_data_size);

    // 记录解压时间
    auto start_decompress = std::chrono::high_resolution_clock::now();
    manager.decompress(device_decompressed_data, device_compressed_data, decomp_config);
    cudaStreamSynchronize(stream);
    auto end_decompress = std::chrono::high_resolution_clock::now();
    double decompress_time = std::chrono::duration<double>(end_decompress - start_decompress).count();

    // 计算性能指标
    double compress_throughput = (in_bytes / 1e6) / compress_time;
    double decompress_throughput = (in_bytes / 1e6) / decompress_time;
    double compression_ratio = static_cast<double>(in_bytes) / comp_out_bytes;

    // 验证解压数据
    std::vector<double> decompressedData(oriData.size());
    cudaMemcpyAsync(decompressedData.data(), device_decompressed_data, in_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // 数据验证（使用容差比较浮点数）
    const double tolerance = 1e-9;
    bool valid = true;
    for (size_t i = 0; i < oriData.size(); ++i) {
        if (std::abs(oriData[i] - decompressedData[i]) > tolerance) {
            valid = false;
            std::cerr << "Mismatch at position " << i 
                      << ": original=" << oriData[i] 
                      << ", decompressed=" << decompressedData[i] 
                      << std::endl;
            break;
        }
    }

    if (!valid) {
        std::cerr << "Data mismatch detected in " << file_path << "!" << std::endl;
        FAIL() << "Data validation failed for file: " << file_path;
    } else {
        std::cout << "Decompression validated successfully." << std::endl;
    }

    // 输出结果
    std::cout << "File: " << fs::path(file_path).filename() << std::endl;
    std::cout << "Original Size: " << in_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Compressed Size: " << comp_out_bytes / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Compression Ratio: " << compression_ratio << ":1" << std::endl;
    std::cout << "Compression Time: " << compress_time * 1000 << " ms (" 
              << compress_throughput << " MB/s)" << std::endl;
    std::cout << "Decompression Time: " << decompress_time * 1000 << " ms (" 
              << decompress_throughput << " MB/s)" << std::endl;

    // 清理资源
    cudaFree(device_input_data);
    cudaFree(device_compressed_data);
    cudaFree(device_decompressed_data);
    // cudaStreamDestroy(stream);
}

// Google Test 测试用例
TEST(GDeflateCompressorTest, CompressionDecompression) {
    std::string dir_path = "../test/data/big";
    if (!fs::exists(dir_path)) {
        GTEST_SKIP() << "Data directory not found: " << dir_path;
    }
    
    int processed = 0;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
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

int main(int argc, char** argv) {
    // 初始化CUDA
    cudaFree(0);
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}