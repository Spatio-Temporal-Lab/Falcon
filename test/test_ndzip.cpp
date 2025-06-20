//cpu版本   

    // #include "ndzip/ndzip.hh"
    // #include <gtest/gtest.h>
    // #include <fstream>
    // #include <vector>
    // #include <chrono>
    // #include <filesystem>
    // #include <iostream>

    // namespace fs = std::filesystem;

    // // 读取浮点数据文件
    // std::vector<float> read_data(const std::string &file_path) {
    //     std::ifstream file(file_path, std::ios::binary);
    //     if (!file.is_open()) {
    //         throw std::runtime_error("无法打开文件: " + file_path);
    //     }

    //     file.seekg(0, std::ios::end);
    //     size_t size = file.tellg() / sizeof(float);
    //     file.seekg(0, std::ios::beg);

    //     std::vector<float> data(size);
    //     file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
    //     file.close();
    //     return data;
    // }

    // // 压缩和解压测试
    // void test_compression(const std::string &file_path) {
    //     // 读取数据
    //     std::vector<float> oriData = read_data(file_path);

    //     // 压缩率相关变量
    //     size_t original_size = oriData.size() * sizeof(float);
    //     // 压缩相关变量
    //     auto compressor = ndzip::make_compressor<float>(1); // 一维压缩器
    //     size_t compressedSize = ndzip::compressed_length_bound<float>({static_cast<ndzip::index_type>(oriData.size())});
    //     std::vector<ndzip::compressed_type<float>> cmpData(compressedSize);

    //     // 开始压缩
    //     std::cout << "压缩开始\n";
    //     auto start_compress = std::chrono::high_resolution_clock::now();
    //     size_t actualCompressedSize = compressor->compress(oriData.data(), 
    //         {static_cast<ndzip::index_type>(oriData.size())}, cmpData.data());
    //     auto end_compress = std::chrono::high_resolution_clock::now();
    //     cmpData.resize(actualCompressedSize/sizeof(ndzip::compressed_type<float>));  // 调整压缩数据大小
        
    //     std::chrono::duration<double> compress_duration = end_compress - start_compress;

    //     // 输出压缩时间
    //     std::cout << "压缩时间: " << compress_duration.count() << " 秒\n";
    //     // 在压缩时间输出后添加：
    //     double compress_throughput = (original_size / (1024.0 * 1024.0)) / compress_duration.count();
    //     std::cout << "压缩吞吐量: " << compress_throughput << " MB/s\n       " << compress_throughput/1024 <<" GB/S\n";

    //     // 解压缩相关变量
    //     auto decompressor = ndzip::make_decompressor<float>(1); // 一维解压缩器
    //     std::vector<float> decompressedData(oriData.size());

    //     // 开始解压缩
    //     std::cout << "解压开始\n";
    //     auto start_decompress = std::chrono::high_resolution_clock::now();
    //     decompressor->decompress(cmpData.data(), decompressedData.data(), 
    //         {static_cast<ndzip::index_type>(oriData.size())});
    //     auto end_decompress = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;

    //     // 输出解压时间
    //     std::cout << "解压时间: " << decompress_duration.count() << " 秒\n";
    //     // 在解压时间输出后添加：
    //     double decompress_throughput = (original_size / (1024.0 * 1024.0)) / decompress_duration.count();
    //     std::cout << "解压吞吐量: " << decompress_throughput << " MB/s\n       " << decompress_throughput/1024 <<" GB/S\n";
    //     // 计算压缩率

    //     // 直接使用字节数计算压缩率
    //     // size_t compressed_size = actualCompressedSize;  // 这里已经是以字节为单位

    //     size_t compressed_size = cmpData.size()*sizeof(uint32_t) ;//* sizeof(ndzip::compressed_type<float>);
    //     double compression_ratio = static_cast<double>(compressed_size) / original_size;

    //     // 输出压缩率
    //     std::cout << "压缩率: " << compression_ratio << "\n";

    //     // 验证解压结果
    //     ASSERT_EQ(decompressedData.size(), oriData.size()) << "解压失败，数据大小不一致";
    //     for (size_t i = 0; i < oriData.size(); ++i) {
    //         if(i<5)
    //         {
    //             std::cout<<i<<" "<<decompressedData[i]<<" "<<oriData[i]<<"\n";
    //         }
    //         ASSERT_FLOAT_EQ(decompressedData[i], oriData[i]) << i << " 解压数据与原始数据不一致";
    //     }
    // }

    // // Google Test 测试用例
    // TEST(NDZipTest, CompressionDecompression) {
    //     std::string dir_path = "/mnt/e/start/gpu/CUDA/cuCompressor/test/data/float"; // 数据文件目录
    //     for (const auto &entry : fs::directory_iterator(dir_path)) {
    //         if (entry.is_regular_file()) {
    //             std::string file_path = entry.path().string();
    //             std::cout << "正在处理文件: " << file_path << "\n";
    //             test_compression(file_path);
    //             std::cout << "---------------------------------------------\n";
    //         }
    //     }
    // }
//
#include "ndzip/cuda.hh"
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <cuda_runtime.h>
#include <thread>
namespace fs = std::filesystem;

// 读取浮点数据文件
std::vector<float> read_data(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg() / sizeof(float);
    file.seekg(0, std::ios::beg);

    std::vector<float> data(size);
    file.read(reinterpret_cast<char *>(data.data()), size * sizeof(float));
    file.close();
    return data;
}

// 基于CUDA的压缩和解压测试
void test_cuda_compression(const std::string &file_path) {
    // 读取数据
    std::vector<float> oriData = read_data(file_path);

    // 数据大小
    size_t data_size = oriData.size();
    ndzip::extent data_extent{static_cast<ndzip::index_type>(data_size)};
    
    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(float);
    
    // 创建CUDA事件用于精确计时
    cudaEvent_t start_total, end_total;
    cudaEvent_t start_h2d, end_h2d;
    cudaEvent_t start_compress, end_compress;
    cudaEvent_t start_d2h_compress, end_d2h_compress;
    cudaEvent_t start_h2d_decompress, end_h2d_decompress;
    cudaEvent_t start_decompress, end_decompress;
    cudaEvent_t start_d2h_decompress, end_d2h_decompress;
    
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&end_h2d);
    cudaEventCreate(&start_compress);
    cudaEventCreate(&end_compress);
    cudaEventCreate(&start_d2h_compress);
    cudaEventCreate(&end_d2h_compress);
    cudaEventCreate(&start_h2d_decompress);
    cudaEventCreate(&end_h2d_decompress);
    cudaEventCreate(&start_decompress);
    cudaEventCreate(&end_decompress);
    cudaEventCreate(&start_d2h_decompress);
    cudaEventCreate(&end_d2h_decompress);

    // 开始总体计时
    cudaEventRecord(start_total);
    
    // ===== 完整的压缩流程 =====
    std::cout << "CUDA 压缩开始\n";
    
    // 1. 分配GPU内存并复制数据到设备
    std::cout << "1. 复制原始数据到设备...\n";
    cudaEventRecord(start_h2d);
    
    float *device_data;
    cudaMalloc(&device_data, data_size * sizeof(float));
    cudaMemcpy(device_data, oriData.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(end_h2d);

    // 2. 创建CUDA压缩器并分配压缩结果存储的GPU内存
    cudaEventRecord(start_compress);
    
    auto compressor = ndzip::make_cuda_compressor<float>(
        ndzip::compressor_requirements{data_extent});

    size_t compressed_size = ndzip::compressed_length_bound<float>(data_extent);
    ndzip::compressed_type<float> *device_compressed_data;
    cudaMalloc(&device_compressed_data, compressed_size * sizeof(ndzip::compressed_type<float>));

    ndzip::index_type *device_compressed_length;
    cudaMalloc(&device_compressed_length, sizeof(ndzip::index_type));

    // 3. 执行GPU压缩
    std::cout << "2. 执行GPU压缩...\n";
    // cudaEventRecord(start_compress);
    
    compressor->compress(device_data, data_extent, device_compressed_data, device_compressed_length);
    
    cudaDeviceSynchronize(); // 确保压缩完成
    cudaEventRecord(end_compress);

    // 4. 拷贝压缩结果到主机
    std::cout << "3. 复制压缩数据回主机...\n";
    cudaEventRecord(start_d2h_compress);
    
    ndzip::index_type host_compressed_length;
    cudaMemcpy(&host_compressed_length, device_compressed_length, sizeof(ndzip::index_type), cudaMemcpyDeviceToHost);

    std::vector<ndzip::compressed_type<float>> host_compressed_data(host_compressed_length);
    cudaMemcpy(host_compressed_data.data(), device_compressed_data,
               host_compressed_length * sizeof(ndzip::compressed_type<float>), cudaMemcpyDeviceToHost);
    
    cudaEventRecord(end_d2h_compress);

    // ===== 完整的解压流程 =====
    std::cout << "\nCUDA 解压缩开始\n";
    
    // 1. 重新分配设备内存并将压缩数据复制到设备（模拟实际场景）
    std::cout << "1. 复制压缩数据到设备...\n";
    cudaEventRecord(start_h2d_decompress);
    
    // 释放之前的设备内存（模拟实际场景中的内存管理）
    cudaFree(device_data);
    cudaFree(device_compressed_data);
    
    // 重新分配并复制压缩数据到设备
    cudaMalloc(&device_compressed_data, host_compressed_length * sizeof(ndzip::compressed_type<float>));
    cudaMemcpy(device_compressed_data, host_compressed_data.data(),
               host_compressed_length * sizeof(ndzip::compressed_type<float>), cudaMemcpyHostToDevice);
    
    cudaEventRecord(end_h2d_decompress);

    // 2. 创建CUDA解压缩器并分配解压结果内存
    cudaEventRecord(start_decompress);
    auto decompressor = ndzip::make_cuda_decompressor<float>(1);
    
    float *device_decompressed_data;
    cudaMalloc(&device_decompressed_data, data_size * sizeof(float));

    // 3. 执行GPU解压缩
    std::cout << "2. 执行GPU解压缩...\n";
    
    decompressor->decompress(device_compressed_data, device_decompressed_data, data_extent);
    
    cudaDeviceSynchronize(); // 确保解压完成
    cudaEventRecord(end_decompress);

    // 4. 拷贝解压缩结果到主机
    std::cout << "3. 复制解压数据回主机...\n";
    cudaEventRecord(start_d2h_decompress);
    
    std::vector<float> host_decompressed_data(data_size);
    cudaMemcpy(host_decompressed_data.data(), device_decompressed_data, data_size * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    cudaEventRecord(end_d2h_decompress);
    cudaEventRecord(end_total);
    
    // 同步所有事件
    cudaDeviceSynchronize();

    // 计算各个阶段的时间
    float h2d_time, compress_time, d2h_compress_time;
    float h2d_decompress_time, decompress_time, d2h_decompress_time, total_time;
    
    cudaEventElapsedTime(&h2d_time, start_h2d, end_h2d);
    cudaEventElapsedTime(&compress_time, start_compress, end_compress);
    cudaEventElapsedTime(&d2h_compress_time, start_d2h_compress, end_d2h_compress);
    cudaEventElapsedTime(&h2d_decompress_time, start_h2d_decompress, end_h2d_decompress);
    cudaEventElapsedTime(&decompress_time, start_decompress, end_decompress);
    cudaEventElapsedTime(&d2h_decompress_time, start_d2h_decompress, end_d2h_decompress);
    cudaEventElapsedTime(&total_time, start_total, end_total);

    // 输出详细的性能统计
    std::cout << "\n=== CUDA事件精确计时结果 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    // 压缩阶段时间
    float total_compress_time = h2d_time + compress_time + d2h_compress_time;
    std::cout << "压缩阶段详细时间:" << std::endl;
    std::cout << "  - 原始数据传输到设备: " << h2d_time << " ms" << std::endl;
    std::cout << "  - GPU压缩核函数: " << compress_time << " ms" << std::endl;
    std::cout << "  - 压缩数据传输到主机: " << d2h_compress_time << " ms" << std::endl;
    std::cout << "  - 压缩总时间: " << total_compress_time << " ms" << std::endl;
    
    // 解压阶段时间
    float total_decompress_time = h2d_decompress_time + decompress_time + d2h_decompress_time;
    std::cout << "\n解压阶段详细时间:" << std::endl;
    std::cout << "  - 压缩数据传输到设备: " << h2d_decompress_time << " ms" << std::endl;
    std::cout << "  - GPU解压核函数: " << decompress_time << " ms" << std::endl;
    std::cout << "  - 解压数据传输到主机: " << d2h_decompress_time << " ms" << std::endl;
    std::cout << "  - 解压总时间: " << total_decompress_time << " ms" << std::endl;
    
    // 总体统计
    std::cout << "\n总体性能统计:" << std::endl;
    std::cout << "  - 纯计算时间(核函数): " << (compress_time + decompress_time) << " ms" << std::endl;
    std::cout << "  - 数据传输时间: " << (h2d_time + d2h_compress_time + h2d_decompress_time + d2h_decompress_time) << " ms" << std::endl;
    std::cout << "  - 总体时间: " << total_time << " ms" << std::endl;
    
    // 计算吞吐量
    double compress_throughput = (original_size / (1024.0 * 1024.0)) / (compress_time / 1000.0);
    double decompress_throughput = (original_size / (1024.0 * 1024.0)) / (decompress_time / 1000.0);
    double total_compress_throughput = (original_size / (1024.0 * 1024.0)) / (total_compress_time / 1000.0);
    double total_decompress_throughput = (original_size / (1024.0 * 1024.0)) / (total_decompress_time / 1000.0);
    
    std::cout << "\n吞吐量统计:" << std::endl;
    std::cout << "  - GPU压缩核函数吞吐量: " << compress_throughput << " MB/s (" << compress_throughput/1024 << " GB/s)" << std::endl;
    std::cout << "  - GPU解压核函数吞吐量: " << decompress_throughput << " MB/s (" << decompress_throughput/1024 << " GB/s)" << std::endl;
    std::cout << "  - 完整压缩流程吞吐量: " << total_compress_throughput << " MB/s (" << total_compress_throughput/1024 << " GB/s)" << std::endl;
    std::cout << "  - 完整解压流程吞吐量: " << total_decompress_throughput << " MB/s (" << total_decompress_throughput/1024 << " GB/s)" << std::endl;

    // 计算压缩率
    size_t compressed_size_out = host_compressed_length * sizeof(ndzip::compressed_type<float>);
    double compression_ratio = static_cast<double>(compressed_size_out) / original_size;
    std::cout << "\n压缩效果:" << std::endl;
    std::cout << "  - 原始数据大小: " << original_size << " bytes (" 
              << original_size / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "  - 压缩后大小: " << compressed_size_out << " bytes (" 
              << compressed_size_out / (1024.0 * 1024.0) << " MB)" << std::endl;
    std::cout << "  - 压缩率: " << compression_ratio << std::endl;

    // 验证解压结果
    std::cout << "\n验证解压结果..." << std::endl;
    ASSERT_EQ(host_decompressed_data.size(), oriData.size()) << "解压失败，数据大小不一致";
    for (size_t i = 0; i < oriData.size(); ++i) {
        ASSERT_FLOAT_EQ(host_decompressed_data[i], oriData[i]) << "数据不一致，索引: " << i;
    }
    std::cout << "解压结果验证通过！" << std::endl;

    // 清理GPU内存
    cudaFree(device_compressed_data);
    cudaFree(device_compressed_length);
    cudaFree(device_decompressed_data);

    // 销毁CUDA事件
    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(end_h2d);
    cudaEventDestroy(start_compress);
    cudaEventDestroy(end_compress);
    cudaEventDestroy(start_d2h_compress);
    cudaEventDestroy(end_d2h_compress);
    cudaEventDestroy(start_h2d_decompress);
    cudaEventDestroy(end_h2d_decompress);
    cudaEventDestroy(start_decompress);
    cudaEventDestroy(end_decompress);
    cudaEventDestroy(start_d2h_decompress);
    cudaEventDestroy(end_d2h_decompress);

    std::cout << "CUDA 压缩与解压测试完成。\n";
}

// Google Test 测试用例
TEST(NDZipCudaTest, CompressionDecompression) {
    // std::string dir_path = "../test/data/big"; // 数据文件目录
    std::string dir_path = "../test/data/temp"; // 数据文件目录

    bool wram = 0;
    for (const auto &entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            if(!wram) {
                std::cout << "------------------warm---------------------------\n";
                wram = 1;
                test_cuda_compression(file_path);
                std::cout << "------------------warm_end---------------------------\n";
            }
            std::cout << "正在处理文件: " << file_path << "\n";
            test_cuda_compression(file_path);
            std::cout << "---------------------------------------------\n";
            cudaDeviceSynchronize();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}