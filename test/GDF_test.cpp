//
// Created by lz on 24-10-9.
//

#include "GDFCompressor.cuh"
#include "GDFDecompressor.cuh"
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

#include <algorithm>


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


// 测试压缩
void test_compression0(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    std::vector<unsigned char> cmpData1;
    std::vector<unsigned char> cmpData2;
    std::vector<unsigned int> cmpOffset;
    std::vector<int> flag;
    size_t nbEle = oriData.size();

    // 记录压缩时间
    std::cout<<"CPU压缩开始\n";
    auto start_compress = std::chrono::high_resolution_clock::now();
    //进行压缩
    CDFCompressor CDFC;
    CDFC.compress(oriData,cmpData1);

    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;
    // 打印压缩时间
    std::cout << "CPU压缩时间: " << compress_duration.count() << " 秒" << std::endl;


    // 记录压缩时间
    std::cout<<"GPU压缩开始\n";
    start_compress = std::chrono::high_resolution_clock::now();
    GDFCompressor GDFC;
    GDFC.compress(oriData,cmpData2);
    end_compress = std::chrono::high_resolution_clock::now();
    compress_duration = end_compress - start_compress;

    // 打印压缩时间
    std::cout << "GPU压缩时间: " << compress_duration.count() << " 秒" << std::endl;


    // 比较压缩后的数据
    try {
        ASSERT_EQ(cmpData1.size(), cmpData2.size()) << "压缩后数据的大小不一致。";

        for (size_t i = 0; i < cmpData1.size(); ++i) {
            if (cmpData1[i] != cmpData2[i]) {
                std::cerr << "数据不一致，位置: " << i << ", CPU 数据: " 
                          << static_cast<int>(cmpData1[i]) 
                          << ", GPU 数据: " 
                          << static_cast<int>(cmpData2[i]) << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "断言失败: " << e.what() << std::endl;
    }
    
}

// // Google Test 测试用例
// TEST(CDFCompressorTest, CompressionDecompression) {
//     std::string dir_path = "/mnt/e/start/gpu/CUDA/cuCompressor/test/data/float";//有毛病还没有数据集
//     for (const auto& entry : fs::directory_iterator(dir_path)) {
//         if (entry.is_regular_file()) {
//             std::string file_path = entry.path().string();
//             std::cout << "正在处理文件: " << file_path << std::endl;
//             test_compression0(file_path);
//             std::cout << "---------------------------------------------" << std::endl;
//         }
//     }
// }

void comp(std::vector<double> oriData)
{
    std::vector<unsigned char> cmpData1;
    std::vector<unsigned char> cmpData2;
    std::vector<unsigned int> cmpOffset;
    std::vector<int> flag;
    size_t nbEle = oriData.size();

    // 记录压缩时间
    std::cout<<"CPU压缩开始\n";
    auto start_compress = std::chrono::high_resolution_clock::now();
    //进行压缩
    CDFCompressor CDFC;
    CDFC.compress(oriData,cmpData1);

    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;
    // 打印压缩时间
    std::cout << "CPU压缩时间: " << compress_duration.count() << " 秒" << std::endl;


    // 记录压缩时间
    std::cout<<"GPU压缩开始\n";
    start_compress = std::chrono::high_resolution_clock::now();
    GDFCompressor GDFC;
    GDFC.compress(oriData,cmpData2);
    end_compress = std::chrono::high_resolution_clock::now();
    compress_duration = end_compress - start_compress;

    // 打印压缩时间
    std::cout << "GPU压缩时间: " << compress_duration.count() << " 秒" << std::endl;


    // 比较压缩后的数据


    // 打印压缩数据
    std::cout << "\nCPU压缩结果:\n";
    for (size_t i = 0; i < cmpData1.size(); ++i) {
        std::cout << std::setw(2) << std::setfill('0') << std::hex 
                  << static_cast<int>(cmpData1[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << "\n"; // 每 16 字节换行
    }
    std::cout << std::endl;

    std::cout << "\nGPU压缩结果:\n";
    for (size_t i = 0; i < cmpData2.size(); ++i) {
        std::cout << std::setw(2) << std::setfill('0') << std::hex 
                  << static_cast<int>(cmpData2[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << "\n"; // 每 16 字节换行
    }
    std::cout << std::endl;
    try {
        ASSERT_EQ(cmpData1.size(), cmpData2.size()) << "压缩后数据的大小不一致。";

        for (size_t i = 0; i < cmpData1.size(); ++i) {
            if (cmpData1[i] != cmpData2[i]) {
                std::cerr << "数据不一致，位置: " << i << ", CPU 数据: " 
                          << static_cast<int>(cmpData1[i]) 
                          << ", GPU 数据: " 
                          << static_cast<int>(cmpData2[i]) << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "断言失败: " << e.what() << std::endl;
    }
}

// 测试压缩
TEST(GDFCompressorTest0, CompressionDecompression) {
    // 读取数据
    std::vector<double> oriData = {0.1,0.2,0.3,0.4,0.5};
    comp(oriData);
    
}
TEST(GDFCompressorTest1, CompressionDecompression) {
    // 读取数据
    std::vector<double> oriData = {0.1};
    comp(oriData);
    
}

std::vector<uint8_t> ConvertArrayToVector(const Array<uint8_t>& arr) {
    return std::vector<uint8_t>(arr.begin(), arr.end());
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}


