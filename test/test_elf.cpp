#include "elf.h"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdint> // 确保 uint8_t 等类型定义
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;
// 数据生成模板
std::vector<double> generate_elf_test_data(size_t size, int pattern_type) {
    std::vector<double> data(size);
    const double amplitude = 20.0f;
    
    switch(pattern_type) {
        case 0: // 随机波动数据
            for(size_t i=0; i<size; ++i) {
                data[i] = amplitude * (static_cast<double>(rand())/RAND_MAX - 0.5f);
            }
            break;
            
        case 1: // 线性递增数据
            for(size_t i=0; i<size; ++i) {
                data[i] = 0.0001f + i*0.000001f;
            }
            break;
            
        case 2: // 周期信号+噪声
            for(size_t i=0; i<size; ++i) {
                double base = amplitude * std::sin(i * 0.1f);
                data[i] = base + 0.1f*(static_cast<double>(rand())/RAND_MAX - 0.5f);
            }
            break;
    }
    return data;
}

// 核心测试模板
CompressionInfo test_elf_compression(const std::vector<double>& original, double error_bound) {
    // 压缩测试
    uint8_t* compressed = nullptr;
    auto compress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t compressed_size = elf_encode(
        const_cast<double*>(original.data()), 
        original.size(),
        &compressed,
        error_bound
    );
    
    auto compress_end = std::chrono::high_resolution_clock::now();
    
    // 解压测试
    std::vector<double> decompressed(original.size());
    auto decompress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t dec_size = elf_decode(
        compressed,
        compressed_size,
        decompressed.data(),
        error_bound
    );
    
    auto decompress_end = std::chrono::high_resolution_clock::now();
    
    // 性能指标计算
    const double original_bytes = original.size() * sizeof(double);
    const double compress_ratio = compressed_size/original_bytes;
    const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
    const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);

    // 压缩时间转换为秒（1 秒 = 1000 毫秒）
    const double compress_sec = compress_time / 1000.0;
    const double decompress_sec = decompress_time / 1000.0;
    // 压缩吞吐量（GB/s）
    const double compression_throughput_GBs = original_GB / compress_sec;
    const double decompression_throughput_GBs = original_GB / decompress_sec;
    
    free(compressed); // 释放压缩数据内存
    return CompressionInfo{
        original_bytes/104/1024.0,
        compressed_size/1024.0/1024.0,
        compress_ratio,
        0,
        compress_time,
        compression_throughput_GBs,0,decompress_time,decompression_throughput_GBs};
}

// 测试用例组
TEST(ElfCompressorTest, SmallDataset) {
    const std::vector<double> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    test_elf_compression(data, 0.01f);
}

TEST(ElfCompressorTest, IncrementalData) {
    auto data = generate_elf_test_data(1024*256, 1); // 生成262K数据点
    test_elf_compression(data, 0.005f);
}

TEST(ElfCompressorTest, PeriodicWithNoise) {
    auto data = generate_elf_test_data(1024*1024, 2); // 生成1M数据点
    test_elf_compression(data, 0.02f);
}

// 添加本地数据测试模板
CompressionInfo test_elf_with_file(const std::string& file_path, double error_bound) {
    auto data = read_data(file_path);
    return test_elf_compression(data, error_bound);
}

// 新增测试用例组
TEST(ElfCompressorTest, LocalDataset) {
    // const std::string data_dir = "../test/data/big"; // 修改为实际路径
    const std::string data_dir = "../test/data/new_tsbs"; // 修改为实际路径

    const double error_bound = 0.001; // 根据数据特性调整
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::cout << "\n正在处理文件:  " << entry.path().string() << std::endl;
            test_elf_with_file(entry.path().string(), error_bound);
        }
    }
}
int main(int argc, char *argv[]) {
    
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {

        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        const double error_bound = 0.001; // 根据数据特性调整
        
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                CompressionInfo ans;
                std::cout << "\n正在处理文件: " << entry.path().string() << std::endl;
                for(int i=0;i<3;i++)
                {
                    ans+=test_elf_with_file(entry.path().string(), error_bound);
                }
                ans=ans/3;
                ans.print();
                std::cout << "\n---------------------------------------------" << std::endl;
            }
        }
    }
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}