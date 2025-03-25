#include "elf.h"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdint> // 确保 uint8_t 等类型定义
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
void test_elf_compression(const std::vector<double>& original, double error_bound) {
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
    const double compress_time = std::chrono::duration<double>(compress_end - compress_start).count();
    const double decompress_time = std::chrono::duration<double>(decompress_end - decompress_start).count();
    
    // 输出性能报告
    std::cout << "\n[性能报告]";
    std::cout << "\n原始尺寸: " << original_bytes << " bytes";
    std::cout << "\n压缩尺寸: " << compressed_size << " bytes";
    std::cout << "\n压缩比率: " << compress_ratio ;
    // std::cout << "\n压缩比率: " << std::fixed << std::setprecision(2) << compress_ratio << ":1";
    std::cout << "\n压缩吞吐: " << (original_bytes/1e9)/compress_time << " GB/s";
    std::cout << "\n解压吞吐: " << (original_bytes/1e9)/decompress_time << " GB/s\n";
    
    // 数据验证
    ASSERT_EQ(dec_size, static_cast<ssize_t>(original.size()));
    for(size_t i=0; i<original.size(); ++i) {
        ASSERT_NEAR(original[i], decompressed[i], error_bound) 
            << "数据不一致 @ 位置 " << i 
            << " (原始值: " << original[i] 
            << ", 解压值: " << decompressed[i] << ")";
    }
    
    free(compressed); // 释放压缩数据内存
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

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}