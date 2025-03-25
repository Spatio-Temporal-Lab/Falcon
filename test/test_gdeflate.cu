#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <nvcomp.hpp>
#include <nvcomp/gdeflate.hpp>  // 引入 nvcomp GDeflate 库
#include <cassert>
#include <cuda_runtime.h>
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);

    // 获取数据大小
    size_t in_bytes = oriData.size() * sizeof(double);

    // 将数据复制到char类型的数组中，因为GDeflate压缩处理的是字节数组
    char* input_data = reinterpret_cast<char*>(oriData.data());

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 计算块大小
    const size_t chunk_size = 65536;

    // 分配CUDA内存
    char* device_input_data;
    cudaMalloc(&device_input_data, in_bytes);
    cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream);

    // 创建 GdeflateManager 实例
    int algo = 0;  // 默认算法
    nvcomp::GdeflateManager manager{chunk_size, algo, stream};

    // 配置压缩
    auto comp_config = manager.configure_compression(in_bytes);

    // 分配输出缓冲区
    uint8_t* device_compressed_data;
    cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size);

    // 记录压缩开始时间
    auto start_compress = std::chrono::high_resolution_clock::now();

    // 压缩数据
    manager.compress(
        reinterpret_cast<const uint8_t*>(device_input_data),
        device_compressed_data,
        comp_config);

    // 同步流
    cudaStreamSynchronize(stream);

    // 记录压缩结束时间
    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;
    double compress_time = compress_duration.count();

    // 获取压缩后的数据大小
    size_t comp_out_bytes = manager.get_compressed_output_size(device_compressed_data);

    // 计算压缩吞吐量 (MB/s)
    double compress_throughput = (in_bytes / 1e6) / compress_time;

    // 配置解压缩
    auto decomp_config = manager.configure_decompression(device_compressed_data);

    // 分配解压缩输出缓冲区
    char* device_decompressed_data;
    cudaMalloc(&device_decompressed_data, decomp_config.decomp_data_size);

    // 记录解压开始时间
    auto start_decompress = std::chrono::high_resolution_clock::now();

    // 解压数据
    manager.decompress(
        reinterpret_cast<uint8_t*>(device_decompressed_data),
        device_compressed_data,
        decomp_config);

    // 同步流
    cudaStreamSynchronize(stream);

    // 记录解压结束时间
    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;
    double decompress_time = decompress_duration.count();

    // 计算解压吞吐量 (MB/s)
    double decompress_throughput = (decomp_config.decomp_data_size / 1e6) / decompress_time;

    // 计算压缩率
    double compression_ratio = static_cast<double>(in_bytes) / comp_out_bytes;

    // 验证解压缩后的数据
    std::vector<double> res(oriData.size());
    cudaMemcpy(&res[0], device_decompressed_data, in_bytes, cudaMemcpyDeviceToHost);

    if (res != oriData) {
        std::cerr << "Data mismatch!" << std::endl;
    } else {
        std::cout << "Decompression is valid." << std::endl;
    }

    // 输出结果
    std::cout << "Compression Time: " << compress_time << " seconds." << std::endl;
    std::cout << "Decompression Time: " << decompress_time << " seconds." << std::endl;
    std::cout << "Compression Ratio: " << compression_ratio << std::endl;
    std::cout << "Compression Throughput: " << compress_throughput << " MB/s" << std::endl;
    std::cout << "Decompression Throughput: " << decompress_throughput << " MB/s" << std::endl;

    // 清理CUDA资源
    cudaFree(device_input_data);
    cudaFree(device_compressed_data);
    cudaFree(device_decompressed_data);
    cudaStreamDestroy(stream);
}

// Google Test 测试用例
TEST(GDeflateCompressorTest, CompressionDecompression) {
    // 读取数据并测试压缩和解压
    std::string dir_path = "../test/data/float"; 
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