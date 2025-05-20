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
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <iostream>
#include "alp.hpp"
#include "data/dataset_utils.hpp"
namespace fs = std::filesystem;

#include <algorithm>

// 原有的CPU/GPU压缩解压测试函数
void comp(std::vector<double> oriData, std::vector<double> &decompData)
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
    auto duration = std::chrono::duration<double, std::milli>(end_compress - start_compress);
    std::cout << "CPU压缩时间: " 
          << std::fixed << std::setprecision(3) 
          << duration.count() << " ms" << std::endl;
    // 打印压缩时间
    std::cout << "CPU压缩时间: " << compress_duration.count() << " 秒" << std::endl;

    // 记录压缩时间
    std::cout<<"GPU压缩开始\n";
    start_compress = std::chrono::high_resolution_clock::now();
    GDFCompressor GDFC;
    GDFC.compress(oriData,cmpData2);
    end_compress = std::chrono::high_resolution_clock::now();
    compress_duration = end_compress - start_compress;

    duration = std::chrono::duration<double, std::milli>(end_compress - start_compress);
    std::cout << "GPU压缩时间: " 
          << std::fixed << std::setprecision(3) 
          << duration.count() << " ms" << std::endl;
    // 打印压缩时间
    std::cout << "GPU压缩时间: " << compress_duration.count() << " 秒" << std::endl;

    // 记录解压时间
    std::cout<<"解压开始\n";
    auto start_decompress = std::chrono::high_resolution_clock::now();

    // GPU解压
    GDFDecompressor GDFD;
    GDFD.decompress(cmpData2, decompData, nbEle);

    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;

    // 打印解压时间
    std::cout << "解压时间: " << decompress_duration.count() << " 秒" << std::endl;

    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(double);
    size_t compressed_size = cmpData2.size() * sizeof(unsigned char);
    double compression_ratio = compressed_size / static_cast<double>(original_size);

    // 打印压缩率
    std::cout << "压缩率: " << compression_ratio << std::endl;

    try {
        for (size_t i = 0; i < oriData.size(); ++i) {
            ASSERT_NEAR(oriData[i], decompData[i], 1e-6) << "第 " << i << " 个值不相等。";
        }
    } catch (const std::exception& e) {
        std::cerr << "断言失败: " << e.what() << std::endl;
    }
}

// 新增: 使用CUDA流的压缩解压测试函数
void comp_stream(std::vector<double> oriData, std::vector<double> &decompData)
{
    // size_t nbEle = oriData.size();
    size_t nbEle =1024*1024*3;
    // oriData.resize(nbELe);
    size_t cmpSize = 0; // 将由压缩函数设置

    // 分配设备内存
    double* d_oriData;
    double* d_decData;
    unsigned char* d_cmpBytes;

    // 为原始数据分配内存
    cudaMalloc(&d_oriData, nbEle * sizeof(double));
    cudaMemcpy(d_oriData, oriData.data(), nbEle * sizeof(double), cudaMemcpyHostToDevice);

    // 为压缩数据分配足够大的内存 (最坏情况下与原始数据一样大)
    cudaMalloc(&d_cmpBytes, nbEle * 2 * sizeof(double));
    
    // 为解压数据分配内存
    cudaMalloc(&d_decData, nbEle * sizeof(double));

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 记录压缩时间
    std::cout<<"GPU流式压缩开始\n";
    auto start_compress = std::chrono::high_resolution_clock::now();
    
    // 调用GDFC_compress
    GDFCompressor GDFC;
    GDFC.GDFC_compress(d_oriData, d_cmpBytes, nbEle, &cmpSize, stream);
    
    // 同步流以确保压缩完成
    cudaStreamSynchronize(stream);
    
    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;
    auto duration = std::chrono::duration<double, std::milli>(end_compress - start_compress);
    
    std::cout << "GPU流式压缩时间: " 
          << std::fixed << std::setprecision(3) 
          << duration.count() << " ms" << std::endl;
    
    // 打印压缩信息
    std::cout << "压缩后大小: " << cmpSize << " 字节" << std::endl;
    double compression_ratio = static_cast<double>(cmpSize) / (nbEle * sizeof(double));
    std::cout << "压缩率: " << compression_ratio << std::endl;

    // 记录解压时间
    std::cout<<"GPU流式解压开始\n";
    auto start_decompress = std::chrono::high_resolution_clock::now();
    
    // 调用GDFC_decompress
    GDFDecompressor GDFD;
    GDFD.GDFC_decompress(d_decData, d_cmpBytes, nbEle, cmpSize, stream);
    
    // 同步流以确保解压完成
    cudaStreamSynchronize(stream);
    
    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;
    
    std::cout << "GPU流式解压时间: " 
          << std::fixed << std::setprecision(3) 
          << std::chrono::duration<double, std::milli>(decompress_duration).count() << " ms" << std::endl;

    // 将解压后的数据复制回主机
    decompData.resize(nbEle);
    cudaMemcpy(decompData.data(), d_decData, nbEle * sizeof(double), cudaMemcpyDeviceToHost);

    // 释放设备内存和流
    cudaFree(d_oriData);
    cudaFree(d_cmpBytes);
    cudaFree(d_decData);
    cudaStreamDestroy(stream);

    // 检查解压结果是否正确
    try {
        for (size_t i = 0; i < oriData.size(); ++i) {
            ASSERT_NEAR(oriData[i], decompData[i], 1e-6) << "第 " << i << " 个值不相等。";
        }
    } catch (const std::exception& e) {
        std::cerr << "断言失败: " << e.what() << std::endl;
    }
}

// 新增: 测试流式压缩解压
void test_stream_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    std::vector<double> decompressedData;
    comp_stream(oriData, decompressedData);
}

// 旧测试: 文件压缩解压缩
void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    std::vector<double> decompressedData;
    comp(oriData, decompressedData);
}

// 测试:小数据集的流式压缩和解压
TEST(GDFStreamTest, StreamCompressionSmall) {
    std::vector<double> oriData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                                  1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0};
    std::vector<double> decompressedData;
    
    comp_stream(oriData, decompressedData);
    
    ASSERT_EQ(oriData.size(), decompressedData.size()) << "解压后数据大小不一致";
    
    for (size_t i = 0; i < oriData.size(); ++i) {
        EXPECT_NEAR(oriData[i], decompressedData[i], 1e-6) << "第 " << i << " 个值不相等";
    }
}

// 测试:中等数据集的流式压缩和解压
TEST(GDFStreamTest, StreamCompressionMedium) {
    // 生成1024个样本数据
    const size_t size = 1024;
    std::vector<double> oriData(size);
    
    // 初始化数组，每个值递增 0.1
    double startValue = 0.0001;
    double step = 0.00001;
    
    for (size_t i = 0; i < size; ++i) {
        oriData[i] = startValue + i * step;
    }
    
    std::vector<double> decompressedData;
    comp_stream(oriData, decompressedData);
    
    ASSERT_EQ(oriData.size(), decompressedData.size()) << "解压后数据大小不一致";
    
    // 验证前10个和最后10个元素，避免全部比较过于冗长
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_NEAR(oriData[i], decompressedData[i], 1e-6) << "前部第 " << i << " 个值不相等";
    }
    
    for (size_t i = size - 10; i < size; ++i) {
        EXPECT_NEAR(oriData[i], decompressedData[i], 1e-6) << "后部第 " << i << " 个值不相等";
    }
}

// 测试:大数据集的流式压缩和解压
// TEST(GDFStreamTest, StreamCompressionLarge) {
//     // 生成一个包含递增和随机变化的更大的数据集
//     const size_t size = 1024 * 64; // 65536个元素
//     std::vector<double> oriData(size);
    
//     // 填充数据:一半是递增的，一半是振荡的
//     double startValue = 1.0;
    
//     for (size_t i = 0; i < size / 2; ++i) {
//         oriData[i] = startValue + i * 0.01;
//     }
    
//     for (size_t i = size / 2; i < size; ++i) {
//         // 生成振荡数据 - 随机波动在原来值的基础上
//         oriData[i] = oriData[i - size/2] + (std::sin(i * 0.1) * 0.5);
//     }
    
//     std::vector<double> decompressedData;
//     comp_stream(oriData, decompressedData);
    
//     ASSERT_EQ(oriData.size(), decompressedData.size()) << "解压后数据大小不一致";
    
//     // 随机选择100个点进行验证
//     srand(42); // 设置固定的随机种子以便结果可复现
//     int errors = 0;
//     const int check_points = 100;
    
//     for (int i = 0; i < check_points; ++i) {
//         size_t idx = rand() % size;
//         if (std::abs(oriData[idx] - decompressedData[idx]) > 1e-6) {
//             errors++;
//             if (errors <= 5) { // 只显示前5个错误
//                 std::cout << "错误: 索引 " << idx << ", 原始值 " << oriData[idx] 
//                         << ", 解压值 " << decompressedData[idx] << std::endl;
//             }
//         }
//     }
    
//     EXPECT_EQ(0, errors) << "在" << check_points << "个随机检查点中有" << errors << "个错误";
// }

// 测试:混合浮点数类型的流式压缩和解压
// TEST(GDFCompressorTest, StreamCompressionMixedFloats) {
//     // 测试各种不同范围和精度的浮点数
//     std::vector<double> oriData = {
//         // 非常小的值
//         1e-10, 2e-10, 3e-10,
//         // 小值
//         0.0001, 0.0002, 0.0003,
//         // 常规值
//         1.0, 2.0, 3.0,
//         // 大值
//         1000.0, 2000.0, 3000.0,
//         // 非常大的值
//         1e10, 2e10, 3e10,
//         // 负值
//         -1.0, -2.0, -3.0,
//         // 小数部分有多位的值
//         1.23456789, 2.34567890, 3.45678901,
//         // 特殊值
//         0.0, -0.0
//     };
    
//     std::vector<double> decompressedData;
//     comp_stream(oriData, decompressedData);
    
//     ASSERT_EQ(oriData.size(), decompressedData.size()) << "解压后数据大小不一致";
    
//     for (size_t i = 0; i < oriData.size(); ++i) {
//         double relative_error = std::abs(oriData[i]) > 1e-10 ? 
//                                std::abs((oriData[i] - decompressedData[i]) / oriData[i]) : 
//                                std::abs(oriData[i] - decompressedData[i]);
                               
//         EXPECT_LT(relative_error, 1e-6) << "第 " << i << " 个值相对误差过大: " 
//                                        << "原始值=" << oriData[i] 
//                                        << ", 解压值=" << decompressedData[i];
//     }
// }

// 原有测试用例 - 保持不变
TEST(GDFCompressorTest, testsmall) {
    // 读取数据
    std::vector<double> oriData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33};
    std::vector<double> newData;
    comp(oriData, newData);
    for(auto n : newData) {
        printf("%f,", n);
    }
    printf("\n");
}

// 原有测试用例 - 保持不变
TEST(GDFCompressorTest, testbig) {
    // 读取数据
    const size_t size = 1024*2*1024;
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
    comp(oriData, newData);
}

// 新增:测试文件目录下的所有数据文件的流式压缩解压
TEST(GDFStreamTest, DirectoryStreamCompression) {
    std::string dir_path = "../test/data/temp";
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_stream_compression(file_path);
            std::cout << "---------------------------------------------" << std::endl;
        }
    }
}

// 性能对比测试:比较原始接口和流式接口的性能
TEST(GDFPerformanceTest, CompareInterfaces) {
    // 生成一个足够大的数据集来测试性能
    const size_t size = 1024 * 64; // 100万个元素
    std::vector<double> oriData(size);
    
    // 填充数据
    for (size_t i = 0; i < size; ++i) {
        oriData[i] = i * 0.01 + i *i * 0.005;
    }
    
    // warmup
    for(int i = 0; i < 3; i++)
    {
        std::vector<double> decompData0;
        comp(oriData, decompData0);
        comp_stream(oriData, decompData0);
    }

    // 进行原始接口测试
    std::vector<double> decompData1;
    auto start1 = std::chrono::high_resolution_clock::now();
    comp(oriData, decompData1);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration<double, std::milli>(end1 - start1);
    
    // 进行流式接口测试
    std::vector<double> decompData2;
    auto start2 = std::chrono::high_resolution_clock::now();
    comp_stream(oriData, decompData2);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration<double, std::milli>(end2 - start2);
    
    std::cout << "原始接口总时间: " << duration1.count() << " ms" << std::endl;
    std::cout << "流式接口总时间: " << duration2.count() << " ms" << std::endl;
    std::cout << "性能提升: " << (duration1.count() / duration2.count()) << "倍" << std::endl;
    
    // 验证两种方法解压的结果是否一致
    ASSERT_EQ(decompData1.size(), decompData2.size()) << "两种解压方法产生的数据大小不一致";
    
    int mismatches = 0;
    for (size_t i = 0; i < decompData1.size(); ++i) {
        if (std::abs(decompData1[i] - decompData2[i]) > 1e-6) {
            mismatches++;
            if (mismatches <= 5) {
                std::cout << "不匹配: 索引 " << i << ", 原始接口 " << decompData1[i] 
                        << ", 流式接口 " << decompData2[i] << std::endl;
            }
        }
    }
    
    EXPECT_EQ(0, mismatches) << "两种接口解压结果有 " << mismatches << " 个不匹配";
}

// 保持原有的辅助函数
std::vector<uint8_t> ConvertArrayToVector(const Array<uint8_t>& arr) {
    return std::vector<uint8_t>(arr.begin(), arr.end());
}

// 原有测试用例 - 保持不变
TEST(GDFCompressorTest, testfiles) {
    std::string dir_path = "../test/data/temp";
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