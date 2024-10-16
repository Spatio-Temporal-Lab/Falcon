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

namespace fs = std::filesystem;

// 读取浮点数数据文件
std::vector<double> read_data(const std::string& file_path) {
    std::cout<<file_path<<" 01\n";
    std::vector<double> data;
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return data;
    }
    double value;
    while (file >> value) {
        //data.push_back(static_cast<double>(std::round(value))); 四舍五入  
        data.push_back(static_cast<double>(value));  
    }
    //std::cout<<"read:"<<data[0]<<std::endl;
    return data;
}

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
    
    std::cout << "压缩后的数据内容: ";
    for (const auto& byte : cmpData) {
        std::cout << std::hex << static_cast<int>(byte) << " "; // 打印为十六进制
    }
    std::cout << std::dec << std::endl; // 恢复为十进制格式


    
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


    std::cout << "Decompressed Data: ";
    for (const auto& val : decompressedData) {
        std::cout << val << " ";
    }
    std::cout << std::endl;


    
    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(double);
    size_t compressed_size = cmpData.size() * sizeof(unsigned char);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;

    // 打印压缩率
    std::cout << "压缩率: " << compression_ratio << std::endl;

    // 验证解压结果是否与原始数据一致
    ASSERT_EQ(decompressedData, oriData) << "解压失败，数据不一致。";
}

// Google Test 测试用例
TEST(CDFCompressorTest, CompressionDecompression) {
    std::string dir_path = "/mnt/e/start/gpu/CUDA/cuCompressor/test/data/float";//有毛病还没有数据集
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

// TEST(StreamTest, inout)
// {
//     // 创建 OutputBitStream，并写入 8 位数据
//     OutputBitStream outStream(10);
//     outStream.Write(5, 8);  // 写入 8 位整数 5
//     outStream.Flush(); // 刷新缓冲区

//     // 从输出流获取数据并初始化 InputBitStream
//     Array<uint8_t> buffer = outStream.GetBuffer(4); // 获取缓冲区
//     std::vector<uint8_t> stdBuffer = ConvertArrayToVector(buffer); // 转换为 std::vector
//     InputBitStream inStream(stdBuffer);

//     // 从输入流读取 8 位数据
//     int maxDecimalPlaces = static_cast<int>(inStream.Read(8));

//     // 打印结果
//     std::cout << "读取的最大小数位数: " << maxDecimalPlaces << std::endl;  // 应输出 5

// }
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
