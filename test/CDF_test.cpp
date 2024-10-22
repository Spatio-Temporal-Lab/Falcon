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
    
    // std::cout << "压缩后的数据内容: ";
    // for (const auto& byte : cmpData) {
    //     std::cout << std::dec << static_cast<int>(byte) << " "; // 打印为十六进制
    // }
    std::cout << std::dec << std::endl; // 恢复为十进制格式

    // std::cout << "压缩后的数据内容（以二进制格式）: ";
    // for (const auto& byte : cmpData) {
    //     std::cout << std::bitset<8>(byte) << " "; // 打印为二进制
    // }
    // std::cout << std::dec << std::endl; // 恢复为十进制格式
    
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


    // std::cout << "Decompressed Data: ";
    // for (const auto& val : decompressedData) {
    //     std::cout << val << " ";
    // }
    // std::cout << std::endl;



    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(double)/8;
    size_t compressed_size = cmpData.size() * sizeof(unsigned char);
    double compression_ratio = static_cast<double>(original_size) / compressed_size;

    // 打印压缩率
    std::cout << "压缩率: " << compression_ratio << std::endl;
    ASSERT_EQ(decompressedData.size() , oriData.size()) << "解压失败，数据不一致。";
    for(int i=0;i<original_size;i++)
    {
        // 验证解压结果是否与原始数据一致
        ASSERT_EQ(decompressedData[i] , oriData[i]) << "解压失败，数据不一致。";

    }

}

// Google Test 测试用例
TEST(CDFCompressorTest, CompressionDecompression) {
    std::string dir_path = "/home/lz/workspace/cuCompressor/test/data/float";//有毛病还没有数据集
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

//读取测试
// TEST(input,read){
//     std::vector<unsigned char> input = {0x01, 0x04, 0xF0};  // 测试数据流
//     InputBitStream bitStream(input);
//
//     // 读取并打印每个步骤的结果
//     uint64_t first = bitStream.Read(8);  // 读取 01
//     //std::cout << std::hex << "First: " << first << std::endl; // 输出: 1
//     ASSERT_EQ(input[0], first);
//     uint64_t second = bitStream.Read(8); // 读取 04
//     //std::cout << std::hex << "Second: " << second << std::endl; // 输出: 4
//     ASSERT_EQ(input[1], second);
//     uint64_t third = bitStream.Read(5);  // 读取 F0 中的前 5 位，即 11110 -> 1E
//     //std::cout << std::hex << "Third: " << third << std::endl;  // 期望输出: 1E
//     ASSERT_EQ(0x1e, third);
//
// }

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}



//void test_read(const std::string& file_path) {
//     std::vector<double> oriData = read_data(file_path);
//     // 压缩相关变量
//     std::vector<unsigned char> cmpData;

//     //进行压缩
//     CDFCompressor CDFC;
//     CDFC.compress(oriData, cmpData);

//     std::cout << "压缩后的数据内容: ";
//     for (const auto& byte : cmpData) {
//         std::cout << std::hex << static_cast<int>(byte) << " "; // 打印为十六进制
//     }
//     std::cout << std::dec << std::endl; // 恢复为十进制格式

//     InputBitStream in(cmpData);
//     std::vector<uint8_t> reconstructedData;

//     // 读取8位
//     long firstByte = in.Read(8);
//     reconstructedData.push_back(static_cast<uint8_t>(firstByte));

//     // 逐步读取64位数据
//     size_t bytesRead = 1; // 初始已读取1字节
//     while (bytesRead < cmpData.size()/8) {
//         long encodedValue = in.Read(64);
//         // 将64位数据拆成8个字节并存储
//         for (int j = 7; j >= 0; --j) {
//             reconstructedData.push_back(static_cast<uint8_t>((encodedValue >> (j * 8)) & 0xFF));
//         }
//         bytesRead += 8; // 更新已读取字节数
//         std::cout << std::hex << std::setw(16) << encodedValue << std::endl;
//     }

//     // 检查拼接后的数据是否与原始数据一致
//     ASSERT_EQ(reconstructedData, cmpData) << "Reconstructed data does not match original data!";
//     std::cout << "All data matches!" << std::endl;
// }


// TEST(readTest, InputBitStreams) {
//     std::string dir_path = "/mnt/e/start/gpu/CUDA/cuCompressor/test/data/float";//有毛病还没有数据集
//     for (const auto& entry : fs::directory_iterator(dir_path)) {
//         if (entry.is_regular_file()) {
//             std::string file_path = entry.path().string();
//             std::cout << "正在处理文件: " << file_path << std::endl;
//             test_read(file_path);
//             std::cout << "---------------------------------------------" << std::endl;
//         }
//     }
// }


//测试输入流的
// TEST(Test1, readin) {
//     std::vector<uint8_t> data = {
//         0x10, 0x01, 0x6B, 0x71, 0x4E, 0xD8, 0x85, 0xC0, 
//         0x00, 0x00, 0x00, 0x12, 0x00, 0x9C, 0xE5, 0x40, 
//         0x00, 0x00, 0x00, 0x12, 0x00, 0x9C, 0xE5, 0x3F, 
//         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
//         0x00, 0x00, 0x00, 0x00
//     };
//     InputBitStream in(data);
//     std::vector<uint8_t> reconstructedData;

//     // 逐步读取 64 位并存储
//     for (size_t i = 0; i < data.size() / 8; ++i) {
//         long encodedValue = in.Read(64);
//         for (int j = 7; j >= 0; --j) {
//             reconstructedData.push_back(static_cast<uint8_t>((encodedValue >> (j * 8)) & 0xFF));
//         }
//         std::cout << std::hex << std::setw(16) << encodedValue << std::endl;
//     }

//     // 检查拼接后的数据是否与原始数据一致
//     ASSERT_EQ(reconstructedData, data) << "Reconstructed data does not match original data!";
//     std::cout << "All data matches!" << std::endl;
// }

// long zigzag_decode(unsigned long z) {
//     return (z >> 1) ^ (-(z & 1));
// }
// unsigned long zigzag_encode(long value)
// {
//     return (value << 1) ^ (value >> 63);
// }
// TEST(ZIGZAG,zigzag){
//     long a=15;
//     unsigned long zigzag_encoded_value = 0xe0; // 例子：十六进制值
//     long original_value = zigzag_decode(zigzag_encoded_value);
//     std::cout << "原始值: " << a << std::endl;    // 输出原始值 
//     std::cout << "修改值: " << zigzag_encode(a) << std::endl; // 输出原始值
//     std::cout << "原始值: " << zigzag_decode(zigzag_encode(a)) << std::endl; // 输出原始值
//     std::cout << "原始值: " << original_value << std::endl; // 输出原始值

// }