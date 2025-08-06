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
#include <algorithm>

namespace fs = std::filesystem;

CompressionInfo comp(std::vector<double> oriData, std::vector<double> &decompressedData);


// 测试压缩和解压缩
CompressionInfo test_compression(const std::string& file_path) {
    std::vector<double> oriData = read_data(file_path);
    std::vector<double> decompressedData;

    return comp(oriData,decompressedData);
    // ASSERT_EQ(decompressedData.size() , oriData.size()) << "解压失败，数据不一致。";
    // for(int i=0;i<oriData[i];i++)
    // {
    //     // 验证解压结果是否与原始数据一致
    //     // std::cout << std::fixed << std::setprecision(10)<<decompressedData[i]<<" "<<oriData[i]<<std::endl;
    //     ASSERT_EQ(decompressedData[i] , oriData[i]) <<i<< "解压失败，数据不一致。";

    // }
}

CompressionInfo comp(std::vector<double> oriData, std::vector<double> &decompressedData)
{
    // 读取数据

    //std::cout<<" ordata : "<<oriData[0]<<std::endl;
    // 压缩相关变量
    std::vector<unsigned char> cmpData;
    std::vector<unsigned int> cmpOffset;
    std::vector<int> flag;
    size_t nbEle = oriData.size();

    // 记录压缩时间
    // std::cout<<"压缩开始\n";
    auto start_compress = std::chrono::high_resolution_clock::now();
    //进行压缩
    CDFCompressor CDFC;
    CDFC.compress(oriData,cmpData);
    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;
    // 打印压缩时间
    // std::cout << "压缩时间: " << compress_duration.count() << " 秒" << std::endl;
    
    // 解压缩相关变量

    // int bit_rate = get_bit_num(*std::max_element(cmpOffset.begin(), cmpOffset.end()));

    // 记录解压时间
    // std::cout<<"解压开始\n";
    auto start_decompress = std::chrono::high_resolution_clock::now();

    //进行解压
    CDFDecompressor CDFD;
    CDFD.decompress(cmpData,decompressedData);
    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;

    // // 打印解压时间
    // std::cout << "解压时间: " << decompress_duration.count() << " 秒" << std::endl;
    const double compress_time = std::chrono::duration<double, std::milli>(end_compress - start_compress).count();
    const double decompress_time = std::chrono::duration<double, std::milli>(end_decompress - start_decompress).count();
    const double compress_sec = compress_time / 1000.0;
    const double decompress_sec = decompress_time / 1000.0;

    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(double);
    size_t compressed_size = cmpData.size() * sizeof(unsigned char);
    const double original_GB = original_size/ (1024.0 * 1024.0 * 1024.0);
    double compression_ratio = compressed_size/ static_cast<double>(original_size);
    const double compression_throughput_GBs = original_GB / compress_sec;
    const double decompression_throughput_GBs = original_GB / decompress_sec;
    
    // 打印压缩率
    // std::cout << "压缩率: " << compression_ratio << std::endl;
    return CompressionInfo{
        original_size/1024.0/1024.0,
        compressed_size/1024.0/1024.0,
        compression_ratio,
        0,
        compress_time,
        compression_throughput_GBs,0,decompress_time,decompression_throughput_GBs};
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

int main(int argc, char *argv[]) {
    
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {

        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        // const double error_bound = 0.000; // 根据数据特性调整
        
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                CompressionInfo ans;
                std::cout << "\n正在处理文件: " << entry.path().string() << std::endl;
                for(int i=0;i<3;i++)
                {
                    ans+=test_compression(entry.path().string());
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