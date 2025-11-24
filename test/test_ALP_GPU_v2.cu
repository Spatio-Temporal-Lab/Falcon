#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <cassert>

// ALP-G 头文件
#include "alp/alp-bindings.cuh"
#include "flsgpu/flsgpu-api.cuh"
#include "flsgpu/structs.cuh"
#include "data/dataset_utils.hpp"
#include "generated-bindings/kernel-bindings.cuh"
#include "engine/enums.cuh"
#include "engine/data.cuh"
#include "engine/verification.cuh"

namespace fs = std::filesystem;

// 函数声明
CompressionInfo comp_ALP_G(std::vector<double> oriData);
CompressionInfo test_compression(const std::string& file_path);
CompressionInfo test_beta_compression(const std::string& file_path, int beta);

// ==================== 主压缩函数 ====================
CompressionInfo comp_ALP_G(std::vector<double> oriData) {
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(double);
    
    if (original_num_elements == 0) {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return CompressionInfo{};
    }
    
    // ==================== 配置参数 ====================
    constexpr size_t VECTOR_SIZE = 1024;
    // 重要：UNPACK_N_VECTORS > 1 需要特殊的向量分组逻辑
    // 当 UNPACK_N_VECTORS = 4 时，GPU内核期望处理连续的4个向量组
    // 目前建议使用 UNPACK_N_VECTORS = 1 以确保正确性
    constexpr unsigned UNPACK_N_VECTORS = 1;  // 推荐值：1（安全），4（高性能但需要特殊处理）
    
    // 根据 ALP-G 源码中 FillWarpThreadblockMapping 的实际定义计算线程块参数
    // 对于 double 类型：
    // utils::get_n_lanes<double>() = 16
    // consts::THREADS_PER_WARP = 32  
    // N_WARPS_PER_BLOCK = max(16/32, 2) = max(0, 2) = 2
    // N_THREADS_PER_BLOCK = 2 * 32 = 64
    // N_CONCURRENT_VECTORS_PER_BLOCK = 64 / 16 = 4
    constexpr size_t N_LANES_DOUBLE = 16;
    constexpr size_t THREADS_PER_WARP = 32;
    constexpr size_t N_WARPS_PER_BLOCK = 2;  // max(16/32, 2) = 2
    constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;  // 2 * 32 = 64
    constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_DOUBLE;  // 64 / 16 = 4
    constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK;  // 4 * 4 = 16
    
    // ==================== 数据填充策略 ====================
    size_t num_elements = original_num_elements;
    std::vector<double> paddedData;
    const double* data_ptr = oriData.data();
    
    // 检查数据可压缩性
    if(alp::is_compressable(data_ptr, num_elements)) {
        std::cout << "✓ 数据可压缩" << std::endl;
    } else {
        std::cout << "⚠️ 数据可压缩性较差" << std::endl;
    }
    
    // 计算需要的向量数
    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    
    // 关键修复：确保向量数量能够被线程块完全处理
    // 每个线程块处理 VECTORS_PER_BLOCK 个向量，必须向上取整
    size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
    
    // std::cout << "🔧 线程块配置详情:" << std::endl;
    // std::cout << "   UNPACK_N_VECTORS = " << UNPACK_N_VECTORS << std::endl;
    // std::cout << "   N_CONCURRENT_VECTORS_PER_BLOCK = " << N_CONCURRENT_VECTORS_PER_BLOCK << std::endl; 
    // std::cout << "   VECTORS_PER_BLOCK = " << VECTORS_PER_BLOCK << std::endl;
    // std::cout << "   原始向量数 = " << n_vecs << ", 填充后向量数 = " << n_vecs_padded << std::endl;
    // std::cout << "   预期线程块数 = " << (n_vecs_padded / VECTORS_PER_BLOCK) << std::endl;
    
    // if (UNPACK_N_VECTORS > 1) {
    //     std::cout << "⚠️ 注意：UNPACK_N_VECTORS > 1 使用连续向量组处理模式" << std::endl;
    // }
    
    if (num_elements_padded != original_num_elements) {
        size_t padding_needed = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;
        
        // std::cout << "⚠️ 数据需要填充: " << std::endl;
        // std::cout << "   原始元素数=" << original_num_elements << std::endl;
        // std::cout << "   原始向量数=" << n_vecs << std::endl;
        // std::cout << "   填充后向量数=" << n_vecs_padded 
        //           << " (每块处理 " << VECTORS_PER_BLOCK << " 个向量)" << std::endl;
        // std::cout << "   填充后元素数=" << num_elements 
        //           << " (+" << padding_needed << ")" << std::endl;
        
        paddedData.reserve(num_elements);
        paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
        double padding_value = oriData.back();
        paddedData.insert(paddedData.end(), padding_needed, padding_value);
        data_ptr = paddedData.data();
    }
    
    const size_t data_size = num_elements * sizeof(double);
    
    // ==================== 压缩阶段 ====================
    auto start_total_compress = std::chrono::high_resolution_clock::now();
    auto start_kernel = start_total_compress;

    flsgpu::host::ALPColumn<double> host_compressed_column;
    try {
        start_kernel = std::chrono::high_resolution_clock::now();
        host_compressed_column = alp::encode<double>(data_ptr, num_elements, false);
        auto end_kernel = std::chrono::high_resolution_clock::now();
    } catch (const std::exception& e) {
        std::cerr << "❌ ALP-G 压缩失败: " << e.what() << std::endl;
        return CompressionInfo{};
    }
    
    size_t compressed_size = host_compressed_column.compressed_size_bytes_alp;
    double compression_ratio = static_cast<double>(compressed_size) / original_size;
    
    if (compressed_size == 0) {
        std::cerr << "❌ 压缩失败: 压缩大小为0" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }
    
    std::cout << "✓ 压缩完成: " << compressed_size << " bytes, 比率=" 
              << compression_ratio << "x" << std::endl;
    
    // GPU 数据转移
    flsgpu::device::ALPColumn<double> device_column;
    try {
        device_column = host_compressed_column.copy_to_device();
        cudaDeviceSynchronize();
    } catch (const std::exception& e) {
        std::cerr << "❌ GPU 数据转移失败: " << e.what() << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }
    
    auto end_total_compress = std::chrono::high_resolution_clock::now();
    double compression_kernel_time = std::chrono::duration<double, std::milli>(end_total_compress - start_kernel).count();
    double compression_total_time = std::chrono::duration<double, std::milli>(end_total_compress - start_total_compress).count();
    
    // ==================== 解压阶段 ====================
    auto start_total_decompress = std::chrono::high_resolution_clock::now();
    
    // 调试信息
    size_t actual_n_vecs = utils::get_n_vecs_from_size(device_column.n_values);
    size_t expected_blocks = (actual_n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK;
    
    // std::cout << "📊 解压配置:" << std::endl;
    // std::cout << "   数据大小: " << device_column.n_values << " 元素" << std::endl;
    // std::cout << "   向量数: " << actual_n_vecs << std::endl;
    // std::cout << "   unpack_n_vectors: " << UNPACK_N_VECTORS << std::endl;
    // std::cout << "   每块处理向量数: " << VECTORS_PER_BLOCK << std::endl;
    // std::cout << "   预期线程块数: " << expected_blocks << std::endl;
    
    double* host_decompressed_data = nullptr;
    try {
        start_kernel = std::chrono::high_resolution_clock::now();
        
        // GPU 解压（返回的是 CPU 主机指针）
        host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPColumn<double>>(
            device_column,
            UNPACK_N_VECTORS,  // 使用配置的参数
            1,                 // unpack_n_values
            enums::Unpacker::StatefulBranchless,
            enums::Patcher::Stateless,  // 使用 Stateless 以获得更好的性能
            1                  // n_samples
        );
        
        cudaDeviceSynchronize();
        auto end_kernel = std::chrono::high_resolution_clock::now();
        
        if (!host_decompressed_data) {
            throw std::runtime_error("解压返回 nullptr");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ALP-G 解压失败: " << e.what() << std::endl;
        if (host_decompressed_data) delete[] host_decompressed_data;
        flsgpu::host::free_column(device_column);
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }
    
    auto end_total_decompress = std::chrono::high_resolution_clock::now();
    double decompression_kernel_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_kernel).count();
    double decompression_total_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_total_decompress).count();
    
    // ==================== 数据验证 ====================
    const uint8_t* padded_bytes = reinterpret_cast<const uint8_t*>(data_ptr);
    const uint8_t* decompressed_bytes = reinterpret_cast<const uint8_t*>(host_decompressed_data);
    size_t actual_decomp_size = device_column.n_values * sizeof(double);
    
    bool validation_passed = true;
    if (memcmp(padded_bytes, decompressed_bytes, actual_decomp_size) != 0) {
        std::cout << "❌ 数据验证失败!" << std::endl;
        
        const double* padded_data = data_ptr;
        const double* decomp_data = host_decompressed_data;
        int error_count = 0;
        
        // 检查原始数据部分
        for (size_t i = 0; i < device_column.n_values && error_count < 10; ++i) {
            if (std::abs(padded_data[i] - decomp_data[i]) > 1e-10) {
                std::cout << "  数据不匹配 [" << i << "]: expected=" << padded_data[i] 
                          << ", got=" << decomp_data[i] << std::endl;
                error_count++;
            }
        }
        validation_passed = false;
    } else {
        std::cout << "✓ 数据验证成功" << std::endl;
    }
    
    // ==================== 计算吞吐量 ====================
    double compression_total_throughput_gbps = (original_size / 1e9) / (compression_total_time / 1000.0);
    double decompression_total_throughput_gbps = (original_size / 1e9) / (decompression_total_time / 1000.0);
    
    CompressionInfo result = {
        original_size / (1024.0 * 1024.0),
        compressed_size / (1024.0 * 1024.0),
        compression_ratio,
        compression_kernel_time,
        compression_total_time,
        compression_total_throughput_gbps,
        decompression_kernel_time,
        decompression_total_time,
        decompression_total_throughput_gbps
    };
    
    // ==================== 清理资源 ====================
    delete[] host_decompressed_data;
    flsgpu::host::free_column(device_column);
    flsgpu::host::free_column(host_compressed_column);
    cudaDeviceSynchronize();
    
    return result;
}
/*
CompressionInfo comp_ALP_G(std::vector<double> oriData) {
    // std::cout << "Testing ALP-G compression..." << std::endl;
    
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(double);
    
    // std::cout << "Input size: " << original_size << " bytes (" << original_num_elements << " doubles)" << std::endl;
    
    // if (original_num_elements == 0) {
    //     std::cerr << "❌ 输入数据为空" << std::endl;
    //     return CompressionInfo{};
    // }
    
    // // 数据预检查
    // double min_val = *std::min_element(oriData.begin(), oriData.end());
    // double max_val = *std::max_element(oriData.begin(), oriData.end());
    // std::cout << "Data range: [" << min_val << ", " << max_val << "]" << std::endl;
    
    // // 检查是否有无穷大或NaN值
    // bool has_invalid = false;
    // for (const auto& val : oriData) {
    //     if (!std::isfinite(val)) {
    //         has_invalid = true;
    //         break;
    //     }
    // }
    
    // if (has_invalid) {
    //     std::cout << "⚠️ 数据包含无穷大或NaN值" << std::endl;
    // }
    
    // ==================== 数据填充到 1024 的整数倍 ====================
    // constexpr size_t VECTOR_SIZE = 1024;
    // size_t num_elements = original_num_elements;
    // std::vector<double> paddedData;
    // const double* data_ptr = oriData.data();
    // if(alp::is_compressable(data_ptr,num_elements))
    // {
    //     printf("Data is compressable\n");
    // }
    // else{
    //     printf("Data is not compressable\n");
    // }
    // if (num_elements % VECTOR_SIZE != 0) {
    //     size_t padding_needed = VECTOR_SIZE - (num_elements % VECTOR_SIZE);
    //     num_elements = original_num_elements + padding_needed;
        
    //     std::cout << "⚠️ 数据需要填充: 原始=" << original_num_elements 
    //               << ", 填充后=" << num_elements 
    //               << " (+" << padding_needed << ")" << std::endl;
        
    //     paddedData.reserve(num_elements);
    //     paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
    //     double padding_value = oriData.back();
    //     paddedData.insert(paddedData.end(), padding_needed, padding_value);
    //     data_ptr = paddedData.data();
    // }
    
    constexpr size_t VECTOR_SIZE = 1024;
    constexpr size_t VECTORS_PER_BLOCK = 256;  // N_CONCURRENT_VECTORS_PER_BLOCK when unpack_n_vectors=1

    size_t num_elements = original_num_elements;
    std::vector<double> paddedData;
    const double* data_ptr = oriData.data();

    if(alp::is_compressable(data_ptr,num_elements))
    {
        printf("Data is compressable\n");
    }
    else{
        printf("Data is not compressable\n");
    }

    // 计算需要的向量数
    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;

    // 向上取整到 VECTORS_PER_BLOCK 的倍数
    size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;

    if (num_elements_padded != original_num_elements) {
        size_t padding_needed = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;
        
        std::cout << "⚠️ 数据需要填充: 原始=" << original_num_elements 
                << ", 向量数=" << n_vecs 
                << ", 填充后向量数=" << n_vecs_padded
                << ", 填充后大小=" << num_elements 
                << " (+" << padding_needed << ")" << std::endl;
        
        paddedData.reserve(num_elements);
        paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
        double padding_value = oriData.back();
        paddedData.insert(paddedData.end(), padding_needed, padding_value);
        data_ptr = paddedData.data();
    }

    const size_t data_size = num_elements * sizeof(double);
    
    // ==================== 压缩阶段 ====================
    auto start_total_compress = std::chrono::high_resolution_clock::now();
    auto start_kernel = start_total_compress;

    flsgpu::host::ALPColumn<double> host_compressed_column;
    try {
        start_kernel = std::chrono::high_resolution_clock::now();
        host_compressed_column = alp::encode<double>(data_ptr, num_elements, false);
        auto end_kernel = std::chrono::high_resolution_clock::now();
        
        // 虽然这是CPU端压缩，但为了统一接口，我们仍称之为"核函数时间"
    } catch (const std::exception& e) {
        std::cerr << "❌ ALP-G 压缩失败: " << e.what() << std::endl;
        return CompressionInfo{};
    }
    
    size_t compressed_size = host_compressed_column.compressed_size_bytes_alp;
    double compression_ratio = static_cast<double>(compressed_size) / original_size;
    
    if (compressed_size == 0) {
        std::cerr << "❌ 压缩失败: 压缩大小为0" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }
    
    std::cout << "压缩后总大小: " << compressed_size << " bytes" << std::endl;
    
    // GPU 数据转移
    flsgpu::device::ALPColumn<double> device_column;
    try {
        device_column = host_compressed_column.copy_to_device();
        cudaDeviceSynchronize();
    } catch (const std::exception& e) {
        std::cerr << "❌ GPU 数据转移失败: " << e.what() << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }
    
    auto end_total_compress = std::chrono::high_resolution_clock::now();
    double compression_kernel_time = std::chrono::duration<double, std::milli>(end_total_compress - start_kernel).count();
    double compression_total_time = std::chrono::duration<double, std::milli>(end_total_compress - start_total_compress).count();
    
    // ==================== 解压阶段 ====================
    auto start_total_decompress = std::chrono::high_resolution_clock::now();
    
    // 调试：检查向量数量和线程块计算
    // size_t n_vecs = utils::get_n_vecs_from_size(device_column.n_values);
    std::cout << "向量数量计算: n_values=" << device_column.n_values 
              << ", n_vecs=" << n_vecs << std::endl;
    std::cout << "VALUES_PER_VECTOR=" << consts::VALUES_PER_VECTOR << std::endl;
    
    double* host_decompressed_data = nullptr;
    try {
        start_kernel = std::chrono::high_resolution_clock::now();
        
        // 尝试使用CPU版本的解压来对比
        // host_decompressed_data = new double[device_column.n_values];
        // alp::decode(host_compressed_column, host_decompressed_data);
        
        // GPU 解压（返回的是 CPU 主机指针）
        host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPColumn<double>>(
            device_column,
            1,  // unpack_n_vectors - 增加到4以提高处理效率
            1,  // unpack_n_values
            enums::Unpacker::StatefulBranchless,
            enums::Patcher::Stateful,
            1   // n_samples
        );
        
        cudaDeviceSynchronize();
        auto end_kernel = std::chrono::high_resolution_clock::now();
        
        if (!host_decompressed_data) {
            throw std::runtime_error("解压返回 nullptr");
        }
        
    } catch (const std::exception& e) {
        std::cerr << "❌ ALP-G 解压失败: " << e.what() << std::endl;
        if (host_decompressed_data) delete[] host_decompressed_data;
        flsgpu::host::free_column(device_column);
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }
    
    auto end_total_decompress = std::chrono::high_resolution_clock::now();
    double decompression_kernel_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_kernel).count();
    double decompression_total_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_total_decompress).count();
    
    // ==================== 数据验证 ====================
    std::cout << "解压缩后数据大小检查: column.n_values = " << device_column.n_values << std::endl;
    std::cout << "预期填充后大小: " << num_elements << std::endl;
    std::cout << "原始数据大小: " << original_num_elements << std::endl;
    
    // 检查解压出的前几个和后几个值
    std::cout << "解压数据前5个值: ";
    for (int i = 0; i < 5 && i < device_column.n_values; i++) {
        std::cout << host_decompressed_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "解压数据后5个值: ";
    for (size_t i = device_column.n_values - 5; i < device_column.n_values; i++) {
        std::cout << host_decompressed_data[i] << " ";
    }
    std::cout << std::endl;
    
    // GPU 解压缩返回的数据应该是填充后的完整数据
    const uint8_t* padded_bytes = reinterpret_cast<const uint8_t*>(data_ptr);
    const uint8_t* decompressed_bytes = reinterpret_cast<const uint8_t*>(host_decompressed_data);
    
    // 验证填充后的完整数据，但是要使用 device_column.n_values 作为实际大小
    size_t actual_decomp_size = device_column.n_values * sizeof(double);
    
    if (memcmp(padded_bytes, decompressed_bytes, actual_decomp_size) != 0) {
        std::cout << "❌ 数据验证失败!" << std::endl;
        
        // 详细比较前几个值
        const double* padded_data = data_ptr;
        const double* decomp_data = host_decompressed_data;
        int error_count = 0;
        
        // 检查原始数据部分
        for (size_t i = 0; i < std::min(original_num_elements, device_column.n_values) && error_count < 10; ++i) {
            if (std::abs(padded_data[i] - decomp_data[i]) > 1e-10) {
                std::cout << "  数据不匹配 [" << i << "]: padded=" << padded_data[i] 
                          << ", decomp=" << decomp_data[i] << std::endl;
                error_count++;
            }
        }
        
        // 检查填充部分（如果解压的数据包含填充）
        if (device_column.n_values > original_num_elements) {
            std::cout << "  检查填充部分 (" << original_num_elements << " to " << device_column.n_values << ")" << std::endl;
            for (size_t i = original_num_elements; i < device_column.n_values && i < original_num_elements + 5; ++i) {
                if (std::abs(padded_data[i] - decomp_data[i]) > 1e-10) {
                    std::cout << "  填充数据不匹配 [" << i << "]: padded=" << padded_data[i] 
                              << ", decomp=" << decomp_data[i] << std::endl;
                    error_count++;
                }
            }
        }
        
    } else {
        std::cout << "✓ 压缩和解压缩验证成功!" << std::endl;
    }
    
    // ==================== 计算吞吐量 ====================
    double compression_total_throughput_gbps = (original_size / 1e9) / (compression_total_time / 1000.0);
    double decompression_total_throughput_gbps = (original_size / 1e9) / (decompression_total_time / 1000.0);
    
    CompressionInfo result = {
        original_size / (1024.0 * 1024.0),          // original_size_mb
        compressed_size / (1024.0 * 1024.0),        // compressed_size_mb
        compression_ratio,                           // compression_ratio
        compression_kernel_time,                     // comp_kernel_time
        compression_total_time,                      // comp_time
        compression_total_throughput_gbps,           // comp_throughput
        decompression_kernel_time,                   // decomp_kernel_time
        decompression_total_time,                    // decomp_time
        decompression_total_throughput_gbps          // decomp_throughput
    };
    
    // ==================== 清理资源 ====================
    delete[] host_decompressed_data;
    flsgpu::host::free_column(device_column);
    flsgpu::host::free_column(host_compressed_column);
    cudaDeviceSynchronize();
    
    return result;
}
*/
// ==================== 文件测试包装函数 ====================
CompressionInfo test_compression(const std::string& file_path) {
    std::vector<double> oriData = read_data(file_path);
    return comp_ALP_G(oriData);
}

CompressionInfo test_beta_compression(const std::string& file_path, int beta) {
    std::vector<double> oriData = read_data(file_path, beta);
    return comp_ALP_G(oriData);
}

// ==================== Google Test 测试用例 ====================
TEST(ALPGCompressorTest, CompressionDecompression) {
    std::string dir_path = "../test/data/mew_tsbs";
    bool warmup = false;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            std::string file_path = entry.path().string();
            
            CompressionInfo result;
            
            if (!warmup) {
                // 预热运行
                test_compression(file_path);
                cudaDeviceSynchronize();
                warmup = true;
            }
            
            // 正式测试
            result = test_compression(file_path);
            
            // 验证结果
            EXPECT_GT(result.compression_ratio, 0.0);
            EXPECT_GT(result.comp_throughput, 0.0);
            EXPECT_GT(result.decomp_throughput, 0.0);
        }
    }
}

int main(int argc, char *argv[]) {
    
    cudaFree(0);  // 初始化 CUDA
    
    if (argc < 2) {
        // 默认运行 Google Test
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {
        // 目录批处理模式
        std::string dir_path = argv[2];
        std::cout << "📁 处理目录: " << dir_path << std::endl;
        
        // 读取所有CSV文件
        std::vector<std::string> csv_files;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                csv_files.push_back(entry.path().string());
            }
        }
        
        if (csv_files.empty()) {
            std::cerr << "❌ 未找到 CSV 文件" << std::endl;
            return 1;
        }
        
        std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;
        
        // 预热
        std::cout << "\n=== 预热阶段 ===" << std::endl;
        test_compression(csv_files[0]);
        cudaDeviceSynchronize();
        
        // 对每个文件进行测试
        for (const auto& file_path : csv_files) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "文件: " << fs::path(file_path).filename() << std::endl;
            std::cout << "========================================" << std::endl;
            
            CompressionInfo total_result;
            
            // 3次迭代
            for (int i = 0; i < 3; ++i) {
                std::cout << "\n--- 迭代 " << (i+1) << " ---" << std::endl;
                CompressionInfo result = test_compression(file_path);
                total_result += result;
                cudaDeviceSynchronize();
            }
            
            // 计算平均值
            total_result = total_result / 3;
            
            // 输出结果（模仿 LZ4 格式）
            std::cout << "\n=== 平均结果 ===" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "压缩率: " << total_result.compression_ratio << std::endl;
            std::cout << "压缩核函数时间: " << total_result.comp_kernel_time << " ms" << std::endl;
            std::cout << "压缩全流程时间: " << total_result.comp_time << " ms" << std::endl;
            std::cout << "压缩全流程吞吐量: " << total_result.comp_throughput << " GB/s" << std::endl;
            std::cout << "解压核函数时间: " << total_result.decomp_kernel_time << " ms" << std::endl;
            std::cout << "解压全流程时间: " << total_result.decomp_time << " ms" << std::endl;
            std::cout << "解压全流程吞吐量: " << total_result.decomp_throughput << " GB/s" << std::endl;
        }
        
        return 0;
    }
    else if (arg == "--file-beta" && argc >= 3) {
        // Beta 参数扫描模式
        std::string file_path = argv[2];
        std::cout << "🔬 Beta 参数扫描: " << file_path << std::endl;
        
        // 预热
        test_compression(file_path);
        cudaDeviceSynchronize();
        
        for (int beta = 4; beta <= 17; ++beta) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Beta = " << beta << std::endl;
            std::cout << "========================================" << std::endl;
            
            CompressionInfo total_result;
            
            // 3次迭代
            for (int i = 0; i < 3; ++i) {
                CompressionInfo result = test_beta_compression(file_path, beta);
                total_result += result;
                cudaDeviceSynchronize();
            }
            
            // 计算平均值
            total_result = total_result / 3;
            
            // 输出结果
            std::cout << "\n=== 平均结果 ===" << std::endl;
            std::cout << std::fixed << std::setprecision(4);
            std::cout << "压缩率: " << total_result.compression_ratio << std::endl;
            std::cout << "压缩全流程时间: " << total_result.comp_time << " ms" << std::endl;
            std::cout << "压缩全流程吞吐量: " << total_result.comp_throughput << " GB/s" << std::endl;
            std::cout << "解压全流程时间: " << total_result.decomp_time << " ms" << std::endl;
            std::cout << "解压全流程吞吐量: " << total_result.decomp_throughput << " GB/s" << std::endl;
        }
        
        return 0;
    }
    else {
        // 单文件模式
        std::string file_path = arg;
        std::cout << "📂 处理文件: " << file_path << std::endl;
        
        // 预热
        std::cout << "\n=== 预热 ===" << std::endl;
        test_compression(file_path);
        cudaDeviceSynchronize();
        
        CompressionInfo total_result;
        
        // 3次迭代
        for (int i = 0; i < 3; ++i) {
            std::cout << "\n========================================" << std::endl;
            std::cout << "迭代 " << (i+1) << std::endl;
            std::cout << "========================================" << std::endl;
            
            CompressionInfo result = test_compression(file_path);
            total_result += result;
            cudaDeviceSynchronize();
        }
        
        // 计算平均值
        total_result = total_result / 3;
        total_result.print();
        // // 输出结果
        // std::cout << "\n========================================" << std::endl;
        // std::cout << "平均结果" << std::endl;
        // std::cout << "========================================" << std::endl;
        // std::cout << std::fixed << std::setprecision(4);
        // std::cout << "压缩率: " << total_result.compression_ratio << std::endl;
        // std::cout << "压缩核函数时间: " << total_result.comp_kernel_time << " ms" << std::endl;
        // std::cout << "压缩全流程时间: " << total_result.comp_time << " ms" << std::endl;
        // std::cout << "压缩全流程吞吐量: " << total_result.comp_throughput << " GB/s" << std::endl;
        // std::cout << "解压核函数时间: " << total_result.decomp_kernel_time << " ms" << std::endl;
        // std::cout << "解压全流程时间: " << total_result.decomp_time << " ms" << std::endl;
        // std::cout << "解压全流程吞吐量: " << total_result.decomp_throughput << " GB/s" << std::endl;
        
        return 0;
    }
}