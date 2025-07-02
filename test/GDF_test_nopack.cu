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

#include <cmath>
#include <iostream>
#include "alp.hpp"
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

#include <algorithm>



// 新增: 使用CUDA流的压缩解压测试函数
CompressionInfo comp_stream(std::vector<double> oriData, std::vector<double> &decompData)
{

    size_t nbEle = oriData.size();
    size_t cmpSize = 0; // 将由压缩函数设置

    // 分配设备内存
    double* d_oriData;
    double* d_decData;
    unsigned char* d_cmpBytes;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建CUDA事件
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    cudaEvent_t g_startEvent, g_stopEvent;
    cudaEventCreate(&g_startEvent);
    cudaEventCreate(&g_stopEvent);
    
    // ===== 完整的压缩流程 =====
    // std::cout << "开始完整的GPU流式压缩流程\n";
    cudaEventRecord(g_startEvent, stream);
    
    // 1. 为原始数据分配设备内存并从主机复制到设备c
    // std::cout << "1. 复制原始数据到设备...\n";
    cudaMalloc(&d_oriData, nbEle * sizeof(double));
    cudaMalloc(&d_cmpBytes, nbEle * sizeof(double));
    
    cudaMemcpyAsync(d_oriData, oriData.data(), nbEle * sizeof(double), cudaMemcpyHostToDevice, stream);

    // 3. 执行GPU压缩
    // std::cout << "2. 执行GPU压缩...\n";
    cudaEventRecord(startEvent, stream);
    GDFCompressor GDFC;
    GDFC.GDFC_compress_no_pack(d_oriData, d_cmpBytes, nbEle, &cmpSize, stream);
    
    cudaEventRecord(stopEvent, stream);
    // 同步流以确保压缩完成
    cudaStreamSynchronize(stream);


    // 4. 将压缩数据从设备复制回主机
    // std::cout << "3. 复制压缩数据回主机...\n";
    std::vector<unsigned char> h_cmpBytes(cmpSize);
    cudaMemcpyAsync(h_cmpBytes.data(), d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(g_stopEvent, stream);
    cudaStreamSynchronize(stream);
    float end_full_compress=0.0;
    float total_compress_time = 0.0;
    cudaEventElapsedTime(&end_full_compress, startEvent, stopEvent);
    cudaEventElapsedTime(&total_compress_time, g_startEvent,g_stopEvent);
    double compression_ratio = static_cast<double>(cmpSize) / (nbEle * sizeof(double));


    // 释放原始数据的设备内存（模拟实际场景中的内存管理）
    cudaFree(d_oriData);
    cudaFree(d_cmpBytes);

    size_t in_bytes = oriData.size() * sizeof(double);
    double data_size_gb = in_bytes / (1024.0 * 1024.0 * 1024.0);
    double compress_throughput = data_size_gb / (total_compress_time / 1000.0);

    CompressionInfo tmp{
        in_bytes/1024.0/1024.0,
        static_cast<double>(cmpSize) /1024.0/1024.0,
        compression_ratio,
        end_full_compress,
        total_compress_time,
        compress_throughput,
        0,
        0,
        0};
    // 释放设备内存和流
    cudaFree(d_cmpBytes);
    cudaFree(d_decData);
    cudaStreamDestroy(stream);

    return tmp;
}

// 新增: 测试流式压缩解压
CompressionInfo test_stream_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    std::vector<double> decompressedData;
    return comp_stream(oriData, decompressedData);
    // comp_stream_no_pack(oriData);

}
void warmup()
{
    // 读取数据
    std::vector<double> oriData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33};
    std::vector<double> newData;
    comp_stream(oriData, newData);
}


// 测试文件目录下的所有数据文件的流式压缩解压
TEST(GDFStreamTest, DirectoryStreamCompression) {
    warmup();
    // std::string dir_path = "../test/data/big";
    std::string dir_path = "../test/data/neon";

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_stream_compression(file_path);
            std::cout << "---------------------------------------------" << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
    
    cudaFree(0);
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {

        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        bool warm=0;
        int processed = 0;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                CompressionInfo a;
                if(!warm)
                {
                    // std::cout << "\n-------------------warm-------------------------- " << file_path << std::endl;
                    test_stream_compression(file_path);
                    warm=1;
                    // std::cout << "-------------------warm_end------------------------" << std::endl;
                }
                std::cout <<"正在处理文件: " << file_path << std::endl;
                for(int i=0;i<3;i++)
                {
                    a+=test_stream_compression(file_path);
                }
                a=a/3;
                a.print();
                std::cout << "---------------------------------------------" << std::endl;
                processed++;
            }
        }
        
        if (processed == 0) {
            std::cerr << "No files found in directory: " << dir_path << std::endl;
        }
    }
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}