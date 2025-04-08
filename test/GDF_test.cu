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




void comp(std::vector<double> oriData,std::vector<double> &decompData)
{
    std::vector<unsigned char> cmpData1;//cpu
    std::vector<unsigned char> cmpData2;//gpu
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



        // 记录解压时间
    std::cout<<"解压开始\n";
    auto start_decompress = std::chrono::high_resolution_clock::now();

    // 比较压缩后的数据
    // GDFDecompressor0 GDFD;//cpu解压 已弃用
    GDFDecompressor GDFD;//gpu解压
    GDFD.decompress(cmpData2, decompData,nbEle);

    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;

    // 打印解压时间
    std::cout << "解压时间: " << decompress_duration.count() << " 秒" << std::endl;

    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(double);
    size_t compressed_size = cmpData2.size() * sizeof(unsigned char);
    double compression_ratio = compressed_size/ static_cast<double>(original_size);

    // 打印压缩率
    std::cout << "压缩率: " << compression_ratio << std::endl;

    //ASSERT_EQ(decompData.size() , oriData.size()) << "解压失败，数据不一致。";
    // for(int i=0;i<oriData.size();i++)
    // {
    //     // 验证解压结果是否与原始数据一致
    //     //std::cout << std::fixed << std::setprecision(10)<<decompData[i]<<" "<<oriData[i]<<std::endl;
    //     ASSERT_EQ(decompData[i] , oriData[i]) <<i<< "解压失败，数据不一致。";

    // }

    // 打印压缩数据
    // std::cout << "\nCPU压缩结果:\n";
    // for (size_t i = 0; i < cmpData1.size(); ++i) {
    //     std::cout << std::setw(2) << std::setfill('0') << std::hex 
    //               << static_cast<int>(cmpData1[i]) << " ";
    //     if ((i + 1) % 16 == 0) std::cout << "\n"; // 每 16 字节换行
    // }
    // std::cout << std::endl;

    // std::cout << "\nGPU压缩结果:\n";
    // for (size_t i = 0; i < cmpData2.size(); ++i) {
    //     std::cout << std::setw(2) << std::setfill('0') << std::hex 
    //               << static_cast<int>(cmpData2[i]) << " ";
    //     if ((i + 1) % 16 == 0) std::cout << "\n"; // 每 16 字节换行
    // }
    // std::cout << std::endl;
    try {
        for (size_t i = 0; i < oriData.size(); ++i) {
            ASSERT_NEAR(oriData[i], decompData[i], 1e-6) << "第 " << i << " 个值不相等。";
        }

        // ASSERT_EQ(oriData,decompData) << "压缩后数据的大小不一致。";
    } catch (const std::exception& e) {
        std::cerr << "断言失败: " << e.what() << std::endl;
    }
}


// 测试压缩和解压缩
void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    std::vector<double> decompressedData;
    comp(oriData,decompressedData);

}


// 测试压缩
TEST(GDFCompressorTest0, CompressionDecompression) {
    // 读取数据
//    {0.1,0.2,0.3,0.4,0.5,0.12,0.1,0.3,2.1};//{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8};//
    std::vector<double> oriData = {0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33};//{0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.4, 0.2, 0.4, 0.1, 0.015, 0.23};
    std::vector<double> newData;
    comp(oriData,newData);
    for(auto n:newData)
    {
        printf("%f,",n);
    }
    printf("\n");
}

TEST(GDFCompressorTest1, CompressionDecompression) {
    // 读取数据
    //std::vector<double> oriData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.8};//{0.001,0.122,0.01543,0.14,0.5,0.47};
    const size_t size = 1024*2;
    std::vector<double> oriData(size*2);

    // 初始化数组，每个值递增 0.1
    double startValue = 0.0001;
    double step = 0.00001;

    for (size_t i = 0; i < size; ++i) {
        oriData[i] = startValue + i * step;
    }
    for (size_t i = size; i < size * 2; ++i) {
        oriData[i] = std::floor((oriData[i - 1] - 0.05) * 100000000.0) / 100000000.0;
    }


    std::vector<double> newData;
    comp(oriData,newData);

}

std::vector<uint8_t> ConvertArrayToVector(const Array<uint8_t>& arr) {
    return std::vector<uint8_t>(arr.begin(), arr.end());
}
//Google Test 测试用例
TEST(CDFCompressorTest, CompressionDecompression) {
    std::string dir_path = "../test/data/temp";//有毛病还没有数据集
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


