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
    // 开始压缩
    std::cout << "CUDA 压缩开始\n";
    auto start_compress = std::chrono::high_resolution_clock::now();
    
    // 分配GPU内存
    float *device_data;
    cudaMalloc(&device_data, data_size * sizeof(float));
    cudaMemcpy(device_data, oriData.data(), data_size * sizeof(float), cudaMemcpyHostToDevice);

    // 创建CUDA压缩器
    auto compressor = ndzip::make_cuda_compressor<float>(
        ndzip::compressor_requirements{data_extent});

    // 分配压缩结果存储的GPU内存
    size_t compressed_size = ndzip::compressed_length_bound<float>(data_extent);//最大数据量
    ndzip::compressed_type<float> *device_compressed_data;
    cudaMalloc(&device_compressed_data, compressed_size * sizeof(ndzip::compressed_type<float>));

    ndzip::index_type *device_compressed_length;
    cudaMalloc(&device_compressed_length, sizeof(ndzip::index_type));

    // // 开始压缩
    // std::cout << "CUDA 压缩开始\n";
    // auto start_compress = std::chrono::high_resolution_clock::now();
    compressor->compress(device_data, data_extent, device_compressed_data, device_compressed_length);
    // auto end_compress = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> compress_duration = end_compress - start_compress;
    
    // 拷贝压缩结果到主机
    ndzip::index_type host_compressed_length;//压缩数据大小
    cudaMemcpy(&host_compressed_length, device_compressed_length, sizeof(ndzip::index_type), cudaMemcpyDeviceToHost);

    std::vector<ndzip::compressed_type<float>> host_compressed_data(host_compressed_length);//压缩数据
    cudaMemcpy(host_compressed_data.data(), device_compressed_data,
               host_compressed_length * sizeof(ndzip::compressed_type<float>), cudaMemcpyDeviceToHost);

    auto end_compress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compress_duration = end_compress - start_compress;

    // 输出压缩时间
    std::cout << "压缩时间: " << compress_duration.count() << " 秒\n";
    double compress_throughput = (original_size / (1024.0 * 1024.0)) / compress_duration.count();
    std::cout << "解压吞吐量: " << compress_throughput << " MB/s\n       " << compress_throughput/1024 <<" GB/S\n";
    
    // std::cout << "压缩流总元素数: " << host_compressed_length 
    //       << " 元素大小: " << sizeof(ndzip::compressed_type<float>) << "字节\n";

    //  std::cout << "原始数据大小: " 
    //          << original_size << " bytes ("
    //          << std::fixed << std::setprecision(2) 
    //          << original_size / (1024.0 * 1024.0) << " MB / "
    //          << original_size / (1024.0 * 1024.0 * 1024.0) << " GB)\n";
    // static_assert(
    //     sizeof(ndzip::compressed_type<float>) == 4,
    //     "compressed_type<float>应为4字节"
    // );
    // 计算压缩率
    size_t compressed_size_out = host_compressed_length * sizeof(ndzip::compressed_type<float>);
    double compression_ratio = static_cast<double>(compressed_size_out) / original_size;

    // 输出压缩率
    std::cout << "压缩率: " << compression_ratio << "\n";

    // 创建CUDA解压缩器
    auto decompressor = ndzip::make_cuda_decompressor<float>(1);

    // 分配解压缩结果的GPU内存
    float *device_decompressed_data;
    cudaMalloc(&device_decompressed_data, data_size * sizeof(float));

    // 开始解压缩
    std::cout << "CUDA 解压缩开始\n";
    auto start_decompress = std::chrono::high_resolution_clock::now();
    decompressor->decompress(device_compressed_data, device_decompressed_data, data_extent);
    auto end_decompress = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;

    // 拷贝解压缩结果到主机
    std::vector<float> host_decompressed_data(data_size);
    cudaMemcpy(host_decompressed_data.data(), device_decompressed_data, data_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 输出解压时间
    std::cout << "解压时间: " << decompress_duration.count() << " 秒\n";
    double decompress_throughput = (original_size / (1024.0 * 1024.0)) / decompress_duration.count();
    std::cout << "解压吞吐量: " << decompress_throughput << " MB/s\n       " << decompress_throughput/1024 <<" GB/S\n";
    
    // 验证解压结果
    ASSERT_EQ(host_decompressed_data.size(), oriData.size()) << "解压失败，数据大小不一致";
    for (size_t i = 0; i < oriData.size(); ++i) {
        ASSERT_FLOAT_EQ(host_decompressed_data[i], oriData[i]) << "数据不一致，索引: " << i;
    }

    // 清理GPU内存
    cudaFree(device_data);
    cudaFree(device_compressed_data);
    cudaFree(device_compressed_length);
    cudaFree(device_decompressed_data);

    std::cout << "CUDA 压缩与解压测试完成。\n";
}

// Google Test 测试用例
TEST(NDZipCudaTest, CompressionDecompression) {
    std::string dir_path = "../test/data/float"; // 数据文件目录
    for (const auto &entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "正在处理文件: " << file_path << "\n";
            test_cuda_compression(file_path);
            std::cout << "---------------------------------------------\n";
        }
    }
}

