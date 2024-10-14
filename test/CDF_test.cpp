//
// Created by lz on 24-9-26.
//


#include "CDFCompressor.h"
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <iostream>

namespace fs = std::filesystem;

// 读取浮点数数据文件
std::vector<long> read_data(const std::string& file_path) {
    std::vector<long> data;
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return data;
    }
    float value;
    while (file >> value) {
        data.push_back(static_cast<long>(std::round(value)));  // 转换为long类型
    }
    return data;
}

// 测试压缩和解压缩
void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<long> oriData = read_data(file_path);

    // 压缩相关变量
    std::vector<unsigned char> cmpData;
    std::vector<unsigned int> cmpOffset;
    std::vector<int> flag;
    size_t nbEle = oriData.size();

    // 记录压缩时间
    auto start_compress = std::chrono::high_resolution_clock::now();
    CDFCompressor_test(oriData, cmpData, cmpOffset, flag, nbEle);
    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;

    // 打印压缩时间
    std::cout << "压缩时间: " << compress_duration.count() << " 秒" << std::endl;

    // 解压缩相关变量
    std::vector<long> decompressedData;
    int bit_rate = get_bit_num(*std::max_element(cmpOffset.begin(), cmpOffset.end()));

    // 记录解压时间
    auto start_decompress = std::chrono::high_resolution_clock::now();
    CDFDecompressor(cmpData, cmpOffset, decompressedData, bit_rate, nbEle);
    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;

    // 打印解压时间
    std::cout << "解压时间: " << decompress_duration.count() << " 秒" << std::endl;

    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(long);
    size_t compressed_size = cmpData.size() * sizeof(unsigned char);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;

    // 打印压缩率
    std::cout << "压缩率: " << compression_ratio << std::endl;

    // 验证解压结果是否与原始数据一致
    ASSERT_EQ(decompressedData, oriData) << "解压失败，数据不一致。";
}

// Google Test 测试用例
TEST(CDFCompressorTest, CompressionDecompression) {
    std::string dir_path = "Serf/test/data_set";
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