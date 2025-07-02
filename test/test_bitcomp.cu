#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <memory>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>

// nvcomp bitcomp 头文件
#include "nvcomp/bitcomp.hpp"
#include "nvcomp.hpp"
#include "data/dataset_utils.hpp"

namespace fs = std::filesystem;

// 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            FAIL() << "CUDA error: " << cudaGetErrorString(err); \
        } \
    } while(0)

#define CHECK_NVCOMP(call) \
    do { \
        nvcompStatus_t status = call; \
        if (status != nvcompSuccess) { \
            std::cerr << "nvcomp error at " << __FILE__ << ":" << __LINE__ \
                      << " - Status code: " << status << std::endl; \
            FAIL() << "nvcomp error, status code: " << status; \
        } \
    } while(0)

void comp_Bitcomp(std::vector<double> oriData,CompressionInfo &a) ;
CompressionInfo  test_compression(const std::string& file_path);

CompressionInfo  test_compression(const std::string& file_path) {
    // 读取数据
    CompressionInfo tmp;
    std::vector<double> oriData = read_data(file_path);
    comp_Bitcomp(oriData,tmp);
    return tmp;
}

void comp_Bitcomp(std::vector<double> oriData,CompressionInfo &a) {
    std::cout << "Testing nvcomp bitcomp compression (chunked)..." << std::endl;
    
    // 转换为字节数据
    const size_t data_size = oriData.size() * sizeof(double);
    const uint8_t* input_bytes = reinterpret_cast<const uint8_t*>(oriData.data());
    
    std::cout << "Input size: " << data_size << " bytes (" << oriData.size() << " doubles)" << std::endl;
    
    // 数据预检查 - 检查数据范围和特征
    double min_val = *std::min_element(oriData.begin(), oriData.end());
    double max_val = *std::max_element(oriData.begin(), oriData.end());
    std::cout << "Data range: [" << min_val << ", " << max_val << "]" << std::endl;
    
    // 检查是否有无穷大或NaN值
    bool has_invalid = false;
    for (const auto& val : oriData) {
        if (!std::isfinite(val)) {
            has_invalid = true;
            break;
        }
    }
    
    if (has_invalid) {
        std::cout << "⚠️ 数据包含无穷大或NaN值，bitcomp可能不支持" << std::endl;
    }
    
    // 分块参数
    const size_t CHUNK_SIZE = 16 * 1024 * 1024; // 32MB
    size_t num_chunks = (data_size + CHUNK_SIZE - 1) / CHUNK_SIZE;
    std::vector<const void*> h_input_ptrs(num_chunks);
    std::vector<size_t> h_input_sizes(num_chunks);
    std::vector<uint8_t*> d_chunk_inputs(num_chunks);
    
    // 分配每个chunk的device内存并拷贝
    for (size_t i = 0; i < num_chunks; ++i) {
        size_t offset = i * CHUNK_SIZE;
        size_t chunk_bytes = std::min(CHUNK_SIZE, data_size - offset);
        uint8_t* d_chunk = nullptr;
        CHECK_CUDA(cudaMalloc(&d_chunk, chunk_bytes));
        CHECK_CUDA(cudaMemcpy(d_chunk, input_bytes + offset, chunk_bytes, cudaMemcpyHostToDevice));
        d_chunk_inputs[i] = d_chunk;
        h_input_ptrs[i] = d_chunk;
        h_input_sizes[i] = chunk_bytes;
    }
    
    // 计时变量
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_kernel = start_total, end_kernel = start_total;
    
    // 1. 批量压缩
    // 1.1 分配device指针数组
    const void** d_input_ptrs = nullptr;
    size_t* d_input_sizes = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input_ptrs, num_chunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_input_sizes, num_chunks * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_input_ptrs, h_input_ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input_sizes, h_input_sizes.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice));
    
    // 1.2 获取压缩所需的临时内存和最大输出大小
    size_t temp_bytes = 0;
    CHECK_NVCOMP(nvcompBatchedBitcompCompressGetTempSize(
        num_chunks, CHUNK_SIZE, nvcompBatchedBitcompDefaultOpts, &temp_bytes));
    size_t max_output_chunk_size = 0;
    CHECK_NVCOMP(nvcompBatchedBitcompCompressGetMaxOutputChunkSize(
        CHUNK_SIZE, nvcompBatchedBitcompDefaultOpts, &max_output_chunk_size));
    
    // 1.3 分配输出内存
    std::vector<uint8_t*> d_compressed_chunks(num_chunks);
    std::vector<void*> h_compressed_ptrs(num_chunks);
    std::vector<size_t> h_compressed_sizes(num_chunks, max_output_chunk_size);
    for (size_t i = 0; i < num_chunks; ++i) {
        CHECK_CUDA(cudaMalloc(&d_compressed_chunks[i], max_output_chunk_size));
        h_compressed_ptrs[i] = d_compressed_chunks[i];
    }
    void** d_compressed_ptrs = nullptr;
    size_t* d_compressed_sizes = nullptr;
    size_t* d_actual_compressed_sizes = nullptr;
    CHECK_CUDA(cudaMalloc(&d_compressed_ptrs, num_chunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_compressed_sizes, num_chunks * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_actual_compressed_sizes, num_chunks * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_compressed_ptrs, h_compressed_ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_compressed_sizes, h_compressed_sizes.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice));
    
    // 1.4 分配临时内存
    void* d_temp = nullptr;
    if (temp_bytes > 0) CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    
    // 1.5 执行批量压缩
    start_kernel = std::chrono::high_resolution_clock::now();
    nvcompStatus_t status = nvcompBatchedBitcompCompressAsync(
        d_input_ptrs, d_input_sizes, CHUNK_SIZE, num_chunks,
        d_temp, temp_bytes, d_compressed_ptrs, d_actual_compressed_sizes,
        nvcompBatchedBitcompDefaultOpts, 0);
    CHECK_CUDA(cudaStreamSynchronize(0));
    end_kernel = std::chrono::high_resolution_clock::now();
    if (status != nvcompSuccess) {
        std::cout << "❌ bitcomp批量压缩失败，状态码: " << status << std::endl;
        // 清理内存
        if (d_temp) cudaFree(d_temp);
        if (d_input_ptrs) cudaFree(d_input_ptrs);
        if (d_input_sizes) cudaFree(d_input_sizes);
        if (d_compressed_ptrs) cudaFree(d_compressed_ptrs);
        if (d_compressed_sizes) cudaFree(d_compressed_sizes);
        if (d_actual_compressed_sizes) cudaFree(d_actual_compressed_sizes);
        for (auto p : d_chunk_inputs) cudaFree(p);
        for (auto p : d_compressed_chunks) cudaFree(p);
        return;
    }
    // 1.6 获取实际压缩大小
    std::vector<size_t> actual_compressed_sizes(num_chunks);
    CHECK_CUDA(cudaMemcpy(actual_compressed_sizes.data(), d_actual_compressed_sizes, num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
    // 1.7 拷贝所有压缩数据回host
    std::vector<std::vector<uint8_t>> compressed_datas(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        compressed_datas[i].resize(actual_compressed_sizes[i]);
        CHECK_CUDA(cudaMemcpy(compressed_datas[i].data(), d_compressed_chunks[i], actual_compressed_sizes[i], cudaMemcpyDeviceToHost));
    }
    // 1.8 统计压缩信息
    size_t total_compressed_bytes = 0;
    for (auto sz : actual_compressed_sizes) total_compressed_bytes += sz;
    auto end_total_compress = std::chrono::high_resolution_clock::now();
    double compression_kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    double compression_total_time = std::chrono::duration<double, std::milli>(end_total_compress - start_total).count();
    // ==================== 解压缩流程 ====================
    // 2.1 将压缩数据重新拷贝到device
    auto start_total_decompress = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t*> d_decompressed_chunks(num_chunks);
    std::vector<void*> h_decompressed_ptrs(num_chunks);
    std::vector<size_t> h_decompressed_sizes(num_chunks);
    std::vector<uint8_t*> d_recompressed_chunks(num_chunks);
    std::vector<void*> h_recompressed_ptrs(num_chunks);
    std::vector<size_t> h_recompressed_sizes(num_chunks);
    std::vector<uint8_t*> d_compressed_chunks_reload(num_chunks);
    std::vector<void*> h_compressed_ptrs_reload(num_chunks);
    std::vector<size_t> h_compressed_sizes_reload(num_chunks);
    for (size_t i = 0; i < num_chunks; ++i) {
        // 压缩数据重新分配到device
        uint8_t* d_cmp = nullptr;
        CHECK_CUDA(cudaMalloc(&d_cmp, actual_compressed_sizes[i]));
        CHECK_CUDA(cudaMemcpy(d_cmp, compressed_datas[i].data(), actual_compressed_sizes[i], cudaMemcpyHostToDevice));
        d_compressed_chunks_reload[i] = d_cmp;
        h_compressed_ptrs_reload[i] = d_cmp;
        h_compressed_sizes_reload[i] = actual_compressed_sizes[i];
    }
    // 2.2 分配device指针数组
    void** d_compressed_ptrs_reload = nullptr;
    size_t* d_compressed_sizes_reload = nullptr;
    CHECK_CUDA(cudaMalloc(&d_compressed_ptrs_reload, num_chunks * sizeof(void*)));
    CHECK_CUDA(cudaMalloc(&d_compressed_sizes_reload, num_chunks * sizeof(size_t)));
    CHECK_CUDA(cudaMemcpy(d_compressed_ptrs_reload, h_compressed_ptrs_reload.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_compressed_sizes_reload, h_compressed_sizes_reload.data(), num_chunks * sizeof(size_t), cudaMemcpyHostToDevice));
    // 2.3 获取解压所需的临时内存
    size_t decomp_temp_bytes = 0;
    CHECK_NVCOMP(nvcompBatchedBitcompDecompressGetTempSize(num_chunks, CHUNK_SIZE, &decomp_temp_bytes));
    // 2.4 获取原始数据大小
    size_t* d_decomp_sizes = nullptr;
    CHECK_CUDA(cudaMalloc(&d_decomp_sizes, num_chunks * sizeof(size_t)));
    CHECK_NVCOMP(nvcompBatchedBitcompGetDecompressSizeAsync(
        d_compressed_ptrs_reload, d_compressed_sizes_reload, d_decomp_sizes, num_chunks, 0));
    CHECK_CUDA(cudaStreamSynchronize(0));
    std::vector<size_t> decomp_sizes(num_chunks);
    CHECK_CUDA(cudaMemcpy(decomp_sizes.data(), d_decomp_sizes, num_chunks * sizeof(size_t), cudaMemcpyDeviceToHost));
    // 2.5 分配解压输出内存
    for (size_t i = 0; i < num_chunks; ++i) {
        CHECK_CUDA(cudaMalloc(&d_decompressed_chunks[i], decomp_sizes[i]));
        h_decompressed_ptrs[i] = d_decompressed_chunks[i];
        h_decompressed_sizes[i] = decomp_sizes[i];
    }
    void** d_decompressed_ptrs = nullptr;
    CHECK_CUDA(cudaMalloc(&d_decompressed_ptrs, num_chunks * sizeof(void*)));
    CHECK_CUDA(cudaMemcpy(d_decompressed_ptrs, h_decompressed_ptrs.data(), num_chunks * sizeof(void*), cudaMemcpyHostToDevice));
    // 2.6 分配状态数组
    nvcompStatus_t* d_statuses = nullptr;
    CHECK_CUDA(cudaMalloc(&d_statuses, num_chunks * sizeof(nvcompStatus_t)));
    // 2.7 分配临时内存
    void* d_decomp_temp = nullptr;
    if (decomp_temp_bytes > 0) CHECK_CUDA(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
    // 2.8 执行批量解压
    start_kernel = std::chrono::high_resolution_clock::now();
    CHECK_NVCOMP(nvcompBatchedBitcompDecompressAsync(
        d_compressed_ptrs_reload, d_compressed_sizes_reload, d_decomp_sizes, d_decomp_sizes, num_chunks,
        d_decomp_temp, decomp_temp_bytes, d_decompressed_ptrs, d_statuses, 0));
    CHECK_CUDA(cudaStreamSynchronize(0));
    end_kernel = std::chrono::high_resolution_clock::now();
    // 2.9 检查解压状态
    std::vector<nvcompStatus_t> statuses(num_chunks);
    CHECK_CUDA(cudaMemcpy(statuses.data(), d_statuses, num_chunks * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
    bool all_success = true;
    for (size_t i = 0; i < num_chunks; ++i) {
        if (statuses[i] != nvcompSuccess) {
            std::cout << "❌ 解压缩失败: chunk " << i << " status: " << statuses[i] << std::endl;
            all_success = false;
        }
    }
    // 2.10 D2H 解压缩数据传输
    std::vector<uint8_t> output_data(data_size);
    size_t offset = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        CHECK_CUDA(cudaMemcpy(output_data.data() + offset, d_decompressed_chunks[i], decomp_sizes[i], cudaMemcpyDeviceToHost));
        offset += decomp_sizes[i];
    }
    auto end_total_decompress = std::chrono::high_resolution_clock::now();
    double decompression_kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
    double decompression_total_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_total_decompress).count();
    // std::cout << "=== 压缩结果 ===" << std::endl;
    // std::cout << "压缩后总大小: " << total_compressed_bytes << " bytes" << std::endl;
    double compression_total_throughput_gbps = (data_size / 1e9) / (compression_total_time / 1000.0);
    double decompression_total_throughput_gbps = (data_size / 1e9) / (decompression_total_time / 1000.0);
    a= CompressionInfo {
        data_size/1024.0/1024.0,
        total_compressed_bytes/1024.0/1024.0,
        (double)total_compressed_bytes / data_size,
        compression_kernel_time,
        compression_total_time,
        compression_total_throughput_gbps,
        decompression_kernel_time,
        decompression_total_time,
        decompression_total_throughput_gbps
    };
    // std::cout << "压缩率: " << std::fixed << std::setprecision(4) << (double)total_compressed_bytes / data_size << std::endl;
    // std::cout << "压缩核函数时间: "<< std::fixed << std::setprecision(4)  << compression_kernel_time << " ms" << std::endl;
    // std::cout << "压缩全流程时间: " << std::fixed << std::setprecision(4) << compression_total_time << " ms" << std::endl;
    // std::cout << "解压缩核函数时间: " << std::fixed << std::setprecision(4) << decompression_kernel_time << " ms" << std::endl;
    // std::cout << "解压全流程时间: " << std::fixed << std::setprecision(4) << decompression_total_time << " ms" << std::endl;
    // std::cout << "解压全流程吞吐量: " << std::fixed << std::setprecision(4) << decompression_total_throughput_gbps << " GB/s" << std::endl;
    // 2.11 验证数据
    bool success = (offset == data_size) && (memcmp(input_bytes, output_data.data(), data_size) == 0);
    if (success && all_success) {
        std::cout << "✓ 压缩和解压缩验证成功!" << std::endl;
    } else {
        std::cout << "❌ 数据验证失败!" << std::endl;
        std::cout << "   原始大小: " << data_size << ", 解压后大小: " << offset << std::endl;
    }
    // 清理GPU内存
    if (d_temp) cudaFree(d_temp);
    if (d_decomp_temp) cudaFree(d_decomp_temp);
    if (d_input_ptrs) cudaFree(d_input_ptrs);
    if (d_input_sizes) cudaFree(d_input_sizes);
    if (d_compressed_ptrs) cudaFree(d_compressed_ptrs);
    if (d_compressed_sizes) cudaFree(d_compressed_sizes);
    if (d_actual_compressed_sizes) cudaFree(d_actual_compressed_sizes);
    if (d_compressed_ptrs_reload) cudaFree(d_compressed_ptrs_reload);
    if (d_compressed_sizes_reload) cudaFree(d_compressed_sizes_reload);
    if (d_decomp_sizes) cudaFree(d_decomp_sizes);
    if (d_decompressed_ptrs) cudaFree(d_decompressed_ptrs);
    if (d_statuses) cudaFree(d_statuses);
    for (auto p : d_chunk_inputs) cudaFree(p);
    for (auto p : d_compressed_chunks) cudaFree(p);
    for (auto p : d_compressed_chunks_reload) cudaFree(p);
    for (auto p : d_decompressed_chunks) cudaFree(p);
    // return a;
}

// Google Test 测试用例
TEST(BitcompCompressorTest, CompressionDecompression) {
    // 读取数据并测试压缩和解压
    std::string dir_path = "../test/data/source"; 
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_compression(file_path);
            std::cout << "==============================================" << std::endl;
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
                    test_compression(file_path);
                    warm=1;
                    // std::cout << "-------------------warm_end------------------------" << std::endl;
                }
                std::cout << "\nProcessing file: " << file_path << std::endl;
                for(int i=0;i<3;i++)
                {
                    a+=test_compression(file_path);
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