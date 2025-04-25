//
// Created by lz on 24-9-26.
//


#include "CDFCompressor.h"
#include "CDFDecompressor.h"
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <iostream>
#include "alp.hpp"
#include "data/dataset_utils.hpp"
namespace fs = std::filesystem;



// 测试压缩和解压缩
void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    //std::cout<<" ordata : "<<oriData[0]<<std::endl;
    // 压缩相关变量
    std::vector<unsigned char> cmpData;
    std::vector<unsigned int> cmpOffset;
    std::vector<int> flag;
    size_t nbEle = oriData.size();

    // 记录压缩时间
    std::cout<<"压缩开始\n";
    auto start_compress = std::chrono::high_resolution_clock::now();
    //进行压缩
    CDFCompressor CDFC;
    CDFC.compress(oriData,cmpData);
    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;
    // 打印压缩时间
    std::cout << "压缩时间: " << compress_duration.count() << " 秒" << std::endl;
    
    // 解压缩相关变量
    std::vector<double> decompressedData;
    // int bit_rate = get_bit_num(*std::max_element(cmpOffset.begin(), cmpOffset.end()));

    // 记录解压时间
    std::cout<<"解压开始\n";
    auto start_decompress = std::chrono::high_resolution_clock::now();

    //进行解压
    CDFDecompressor CDFD;
    CDFD.decompress(cmpData,decompressedData);
    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;

    // 打印解压时间
    std::cout << "解压时间: " << decompress_duration.count() << " 秒" << std::endl;

    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(double);
    size_t compressed_size = cmpData.size() * sizeof(unsigned char);
    double compression_ratio = compressed_size/ static_cast<double>(original_size);

    // 打印压缩率
    std::cout << "压缩率: " << compression_ratio << std::endl;
    ASSERT_EQ(decompressedData.size() , oriData.size()) << "解压失败，数据不一致。";
    for(int i=0;i<oriData[i];i++)
    {
        // 验证解压结果是否与原始数据一致
        // std::cout << std::fixed << std::setprecision(10)<<decompressedData[i]<<" "<<oriData[i]<<std::endl;
        ASSERT_EQ(decompressedData[i] , oriData[i]) <<i<< "解压失败，数据不一致。";

    }

}

// Google Test 测试用例
TEST(CDFCompressorTest, CompressionDecompression) {
    std::string dir_path = "../test/data/big";//有毛病还没有数据集
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_compression(file_path);
            std::cout << "---------------------------------------------" << std::endl;
        }
    }
}

std::vector<uint8_t> ConvertArrayToVector(const Array<uint8_t>& arr) {
    return std::vector<uint8_t>(arr.begin(), arr.end());
}

