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
CompressionInfo test_compression(const std::string &file_path);
CompressionInfo test_beta_compression(const std::string &file_path, int beta);
// 扩展版函数声明(不修改 ALP-G 源码,仅使用其公开接口)
CompressionInfo comp_ALP_G_Extended(std::vector<double> oriData);
CompressionInfo test_compression_extended(const std::string &file_path);
CompressionInfo test_compression_extended_debug(const std::string &file_path);
// 修复版:手动实现正确的扩展列转换
CompressionInfo comp_ALP_G_Extended_Fixed(std::vector<double> oriData);
CompressionInfo test_compression_extended_fixed(const std::string &file_path);
// 分析扩展格式兼容性
void analyze_extended_compatibility(const std::string &file_path);

// ==================== 主压缩函数 ====================
CompressionInfo comp_ALP_G(std::vector<double> oriData)
{
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(double);

    if (original_num_elements == 0)
    {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return CompressionInfo{};
    }

    // ==================== 配置参数 ====================
    constexpr size_t VECTOR_SIZE = 1024;
    // 重要：UNPACK_N_VECTORS > 1 需要特殊的向量分组逻辑
    // 当 UNPACK_N_VECTORS = 4 时，GPU内核期望处理连续的4个向量组
    // 目前建议使用 UNPACK_N_VECTORS = 1 以确保正确性
    constexpr unsigned UNPACK_N_VECTORS = 1; // 推荐值：1（安全），4（高性能但需要特殊处理）

    // 根据 ALP-G 源码中 FillWarpThreadblockMapping 的实际定义计算线程块参数
    // 对于 double 类型：
    // utils::get_n_lanes<double>() = 16
    // consts::THREADS_PER_WARP = 32
    // N_WARPS_PER_BLOCK = max(16/32, 2) = max(0, 2) = 2
    // N_THREADS_PER_BLOCK = 2 * 32 = 64
    // N_CONCURRENT_VECTORS_PER_BLOCK = 64 / 16 = 4
    constexpr size_t N_LANES_DOUBLE = 16;
    constexpr size_t THREADS_PER_WARP = 32;
    constexpr size_t N_WARPS_PER_BLOCK = 2;                                                 // max(16/32, 2) = 2
    constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;            // 2 * 32 = 64
    constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_DOUBLE; // 64 / 16 = 4
    constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK; // 4 * 4 = 16

    // ==================== 数据填充策略 ====================
    size_t num_elements = original_num_elements;
    std::vector<double> paddedData;
    const double *data_ptr = oriData.data();

    // 检查数据可压缩性
    if (alp::is_compressable(data_ptr, num_elements))
    {
        std::cout << "✓ 数据可压缩" << std::endl;
    }
    else
    {
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

    if (num_elements_padded != original_num_elements)
    {
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
    try
    {
        start_kernel = std::chrono::high_resolution_clock::now();
        host_compressed_column = alp::encode<double>(data_ptr, num_elements, false);
        auto end_kernel = std::chrono::high_resolution_clock::now();
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ ALP-G 压缩失败: " << e.what() << std::endl;
        return CompressionInfo{};
    }

    size_t compressed_size = host_compressed_column.compressed_size_bytes_alp;
    double compression_ratio = static_cast<double>(compressed_size) / original_size;

    if (compressed_size == 0)
    {
        std::cerr << "❌ 压缩失败: 压缩大小为0" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }

    std::cout << "✓ 压缩完成: " << compressed_size << " bytes, 比率="
              << compression_ratio << "x" << std::endl;

    // GPU 数据转移
    flsgpu::device::ALPColumn<double> device_column;
    try
    {
        device_column = host_compressed_column.copy_to_device();
        cudaDeviceSynchronize();
    }
    catch (const std::exception &e)
    {
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

    double *host_decompressed_data = nullptr;
    try
    {
        start_kernel = std::chrono::high_resolution_clock::now();

        // GPU 解压（返回的是 CPU 主机指针）
        host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPColumn<double>>(
            device_column,
            UNPACK_N_VECTORS, // 使用配置的参数
            1,                // unpack_n_values
            enums::Unpacker::StatefulBranchless,
            enums::Patcher::Stateless, // 使用 Stateless 以获得更好的性能
            1                          // n_samples
        );

        cudaDeviceSynchronize();
        auto end_kernel = std::chrono::high_resolution_clock::now();

        if (!host_decompressed_data)
        {
            throw std::runtime_error("解压返回 nullptr");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ ALP-G 解压失败: " << e.what() << std::endl;
        if (host_decompressed_data)
            delete[] host_decompressed_data;
        flsgpu::host::free_column(device_column);
        flsgpu::host::free_column(host_compressed_column);
        return CompressionInfo{};
    }

    auto end_total_decompress = std::chrono::high_resolution_clock::now();
    double decompression_kernel_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_kernel).count();
    double decompression_total_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_total_decompress).count();

    // ==================== 数据验证 ====================
    const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
    const uint8_t *decompressed_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
    size_t actual_decomp_size = device_column.n_values * sizeof(double);

    bool validation_passed = true;
    if (memcmp(padded_bytes, decompressed_bytes, actual_decomp_size) != 0)
    {
        std::cout << "❌ 数据验证失败!" << std::endl;

        const double *padded_data = data_ptr;
        const double *decomp_data = host_decompressed_data;
        int error_count = 0;

        // 检查原始数据部分
        for (size_t i = 0; i < device_column.n_values && error_count < 10; ++i)
        {
            if (std::abs(padded_data[i] - decomp_data[i]) > 1e-10)
            {
                std::cout << "  数据不匹配 [" << i << "]: expected=" << padded_data[i]
                          << ", got=" << decomp_data[i] << std::endl;
                error_count++;
            }
        }
        validation_passed = false;
    }
    else
    {
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
        decompression_total_throughput_gbps};

    // ==================== 清理资源 ====================
    delete[] host_decompressed_data;
    flsgpu::host::free_column(device_column);
    flsgpu::host::free_column(host_compressed_column);
    cudaDeviceSynchronize();

    return result;
}
// ==================== 扩展版：基础压缩 -> 扩展列转换 -> 扩展GPU解压 ====================
CompressionInfo comp_ALP_G_Extended(std::vector<double> oriData)
{
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(double);
    if (original_num_elements == 0)
    {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return CompressionInfo{};
    }

    // 与基础函数保持一致的填充/线程块推导
    constexpr size_t VECTOR_SIZE = 1024;
    constexpr unsigned UNPACK_N_VECTORS = 1; // 扩展版安全配置
    constexpr size_t N_LANES_DOUBLE = 16;
    constexpr size_t THREADS_PER_WARP = 32;
    constexpr size_t N_WARPS_PER_BLOCK = 2;
    constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;
    constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_DOUBLE;
    constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK;

    size_t num_elements = original_num_elements;
    std::vector<double> paddedData;
    const double *data_ptr = oriData.data();

    // 可压缩性提示
    if (!alp::is_compressable(data_ptr, num_elements))
    {
        std::cout << "⚠️ 数据可压缩性较差" << std::endl;
    }

    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
    if (num_elements_padded != original_num_elements)
    {
        size_t padding_needed = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;
        paddedData.reserve(num_elements);
        paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
        double padding_value = oriData.back();
        paddedData.insert(paddedData.end(), padding_needed, padding_value);
        data_ptr = paddedData.data();
    }

    auto start_total_compress = std::chrono::high_resolution_clock::now();
    auto start_kernel = start_total_compress;
    flsgpu::host::ALPColumn<double> host_base_column;
    try
    {
        start_kernel = std::chrono::high_resolution_clock::now();
        host_base_column = alp::encode<double>(data_ptr, num_elements, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ 基础压缩失败: " << e.what() << std::endl;
        return CompressionInfo{};
    }

    // 扩展尺寸计算（使用基础列记录的扩展字节数）
    size_t compressed_size_ext = host_base_column.compressed_size_bytes_alp_extended;
    double compression_ratio = static_cast<double>(compressed_size_ext) / original_size;
    if (compressed_size_ext == 0)
    {
        std::cerr << "❌ 扩展格式压缩大小为0" << std::endl;
        flsgpu::host::free_column(host_base_column);
        return CompressionInfo{};
    }
    std::cout << "✓ 基础压缩完成(扩展字节计): " << compressed_size_ext << " bytes, 比率=" << compression_ratio << "x" << std::endl;

    // 转换为扩展列
    flsgpu::host::ALPExtendedColumn<double> host_extended_column = host_base_column.create_extended_column();

    // 复制到设备
    flsgpu::device::ALPExtendedColumn<double> device_extended_column;
    try
    {
        device_extended_column = host_extended_column.copy_to_device();
        cudaDeviceSynchronize();
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ 扩展列复制到 GPU 失败: " << e.what() << std::endl;
        flsgpu::host::free_column(host_extended_column);
        flsgpu::host::free_column(host_base_column);
        return CompressionInfo{};
    }
    auto end_total_compress = std::chrono::high_resolution_clock::now();
    double compression_kernel_time = std::chrono::duration<double, std::milli>(end_total_compress - start_kernel).count();
    double compression_total_time = std::chrono::duration<double, std::milli>(end_total_compress - start_total_compress).count();

    // 解压
    auto start_total_decompress = std::chrono::high_resolution_clock::now();
    double *host_decompressed_data = nullptr;
    try
    {
        start_kernel = std::chrono::high_resolution_clock::now();
        host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPExtendedColumn<double>>(
            device_extended_column,
            UNPACK_N_VECTORS,
            1,
            enums::Unpacker::StatefulBranchless,
            enums::Patcher::NaiveBranchless,
            1);
        cudaDeviceSynchronize();
        if (!host_decompressed_data)
        {
            throw std::runtime_error("扩展GPU解压返回 nullptr");
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ 扩展GPU解压失败: " << e.what() << std::endl;
        if (host_decompressed_data)
            delete[] host_decompressed_data;
        flsgpu::host::free_column(device_extended_column);
        flsgpu::host::free_column(host_extended_column);
        flsgpu::host::free_column(host_base_column);
        return CompressionInfo{};
    }
    auto end_total_decompress = std::chrono::high_resolution_clock::now();
    double decompression_kernel_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_kernel).count();
    double decompression_total_time = std::chrono::duration<double, std::milli>(end_total_decompress - start_total_decompress).count();

    // 验证
    const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
    const uint8_t *decompressed_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
    size_t actual_decomp_size = device_extended_column.n_values * sizeof(double);
    if (memcmp(padded_bytes, decompressed_bytes, actual_decomp_size) != 0)
    {
        std::cout << "❌ 扩展版数据验证失败" << std::endl;
        int shown = 0;
        const double *pd = data_ptr;
        const double *dd = host_decompressed_data;
        for (size_t i = 0; i < device_extended_column.n_values && shown < 10; ++i)
        {
            if (std::abs(pd[i] - dd[i]) > 1e-10)
            {
                std::cout << "  不匹配[" << i << "]: " << pd[i] << " vs " << dd[i] << std::endl;
                ++shown;
            }
        }
    }
    else
    {
        std::cout << "✓ 扩展版数据验证成功" << std::endl;
    }

    double compression_total_throughput_gbps = (original_size / 1e9) / (compression_total_time / 1000.0);
    double decompression_total_throughput_gbps = (original_size / 1e9) / (decompression_total_time / 1000.0);
    CompressionInfo result = {
        original_size / (1024.0 * 1024.0),
        compressed_size_ext / (1024.0 * 1024.0),
        compression_ratio,
        compression_kernel_time,
        compression_total_time,
        compression_total_throughput_gbps,
        decompression_kernel_time,
        decompression_total_time,
        decompression_total_throughput_gbps};

    delete[] host_decompressed_data;
    flsgpu::host::free_column(device_extended_column);
    flsgpu::host::free_column(host_extended_column);
    flsgpu::host::free_column(host_base_column);
    cudaDeviceSynchronize();
    return result;
}

// ==================== 修复版扩展列转换:手动重建正确的lane-divided格式 ====================
CompressionInfo comp_ALP_G_Extended_Fixed(std::vector<double> oriData)
{
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(double);
    if (original_num_elements == 0) return CompressionInfo{};

    constexpr size_t VECTOR_SIZE = 1024;
    constexpr unsigned UNPACK_N_VECTORS = 1;
    constexpr size_t N_LANES_DOUBLE = 16;
    constexpr size_t THREADS_PER_WARP = 32;
    constexpr size_t N_WARPS_PER_BLOCK = 2;
    constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;
    constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_DOUBLE;
    constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK;

    size_t num_elements = original_num_elements;
    std::vector<double> paddedData;
    const double *data_ptr = oriData.data();

    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
    if (num_elements_padded != original_num_elements) {
        size_t padding_needed = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;
        paddedData.reserve(num_elements);
        paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
        paddedData.insert(paddedData.end(), padding_needed, oriData.back());
        data_ptr = paddedData.data();
    }

    auto start_compress = std::chrono::high_resolution_clock::now();
    auto base_col = alp::encode<double>(data_ptr, num_elements, false);
    
    std::cout << "✓ 基础压缩完成, 开始手动修复扩展列转换..." << std::endl;
    
    // 手动实现正确的lane-divided转换
    constexpr size_t N_LANES = 16;
    constexpr size_t VALUES_PER_LANE = 64; // 1024/16
    
    double* fixed_exceptions = new double[base_col.n_exceptions];
    uint16_t* fixed_positions = new uint16_t[base_col.n_exceptions];
    uint16_t* fixed_offsets_counts = new uint16_t[base_col.ffor.bp.get_n_vecs() * N_LANES];
    
    double vec_exc[VECTOR_SIZE];
    uint16_t vec_pos[VECTOR_SIZE];
    uint16_t lane_counts[N_LANES];
    
    size_t global_out_idx = 0;
    for (size_t vec_idx = 0; vec_idx < base_col.ffor.bp.get_n_vecs(); ++vec_idx) {
        uint32_t vec_exc_count = base_col.counts[vec_idx];
        size_t vec_exc_offset = base_col.exceptions_offsets[vec_idx]; // 使用正确的偏移!
        
        memset(lane_counts, 0, sizeof(lane_counts));
        
        // 按lane分组异常
        for (uint32_t i = 0; i < vec_exc_count; ++i) {
            double exc = base_col.exceptions[vec_exc_offset + i];
            uint16_t pos = base_col.positions[vec_exc_offset + i];
            uint32_t lane = pos % N_LANES;
            uint32_t lane_idx = lane_counts[lane]++;
            vec_exc[lane * VALUES_PER_LANE + lane_idx] = exc;
            vec_pos[lane * VALUES_PER_LANE + lane_idx] = pos;
        }
        
        // 按lane顺序输出
        uint32_t vec_out_count = 0;
        for (size_t lane = 0; lane < N_LANES; ++lane) {
            uint32_t cnt = lane_counts[lane];
            for (uint32_t i = 0; i < cnt; ++i) {
                fixed_exceptions[global_out_idx] = vec_exc[lane * VALUES_PER_LANE + i];
                fixed_positions[global_out_idx] = vec_pos[lane * VALUES_PER_LANE + i];
                ++global_out_idx;
                ++vec_out_count;
            }
            fixed_offsets_counts[vec_idx * N_LANES + lane] = (cnt << 10) | (vec_out_count - cnt);
        }
    }
    
    // 构建修复后的扩展列
    flsgpu::host::ALPExtendedColumn<double> fixed_ext_col{
        flsgpu::host::FFORColumn<uint64_t>{
            flsgpu::host::BPColumn<uint64_t>{
                base_col.ffor.bp.n_values,
                base_col.ffor.bp.n_packed_values,
                utils::copy_array(base_col.ffor.bp.packed_array, base_col.ffor.bp.n_packed_values),
                utils::copy_array(base_col.ffor.bp.bit_widths, base_col.ffor.bp.get_n_vecs()),
                utils::copy_array(base_col.ffor.bp.vector_offsets, base_col.ffor.bp.get_n_vecs()),
            },
            utils::copy_array(base_col.ffor.bases, base_col.ffor.bp.get_n_vecs()),
        },
        utils::copy_array(base_col.factor_indices, base_col.ffor.bp.get_n_vecs()),
        utils::copy_array(base_col.fraction_indices, base_col.ffor.bp.get_n_vecs()),
        base_col.n_exceptions,
        utils::copy_array(base_col.exceptions_offsets, base_col.ffor.bp.get_n_vecs()),
        fixed_exceptions,
        fixed_positions,
        fixed_offsets_counts,
        base_col.compressed_size_bytes_alp_extended
    };
    
    auto d_ext = fixed_ext_col.copy_to_device();
    cudaDeviceSynchronize();
    auto end_compress = std::chrono::high_resolution_clock::now();
    
    // GPU解压
    auto start_decomp = std::chrono::high_resolution_clock::now();
    double* decomp = bindings::decompress_column<double, flsgpu::device::ALPExtendedColumn<double>>(
        d_ext, UNPACK_N_VECTORS, 1, enums::Unpacker::StatefulBranchless, enums::Patcher::NaiveBranchless, 1);
    cudaDeviceSynchronize();
    auto end_decomp = std::chrono::high_resolution_clock::now();
    
    // 验证
    bool ok = memcmp(data_ptr, decomp, d_ext.n_values * sizeof(double)) == 0;
    std::cout << (ok ? "✓ 修复版验证成功!" : "❌ 修复版验证失败!") << std::endl;
    if (!ok) {
        int shown = 0;
        for (size_t i = 0; i < d_ext.n_values && shown < 10; ++i) {
            if (std::abs(data_ptr[i] - decomp[i]) > 1e-10) {
                std::cout << "  不匹配[" << i << "]: " << data_ptr[i] << " vs " << decomp[i] << std::endl;
                ++shown;
            }
        }
    }
    
    double comp_time = std::chrono::duration<double, std::milli>(end_compress - start_compress).count();
    double decomp_time = std::chrono::duration<double, std::milli>(end_decomp - start_decomp).count();
    double comp_tp = (original_size / 1e9) / (comp_time / 1000.0);
    double decomp_tp = (original_size / 1e9) / (decomp_time / 1000.0);
    double ratio = (double)base_col.compressed_size_bytes_alp_extended / (double)original_size;
    
    delete[] decomp;
    flsgpu::host::free_column(d_ext);
    flsgpu::host::free_column(fixed_ext_col);
    flsgpu::host::free_column(base_col);
    cudaDeviceSynchronize();
    
    return CompressionInfo{
        original_size / (1024.0 * 1024.0),
        base_col.compressed_size_bytes_alp_extended / (1024.0 * 1024.0),
        ratio, comp_time, comp_time, comp_tp, decomp_time, decomp_time, decomp_tp
    };
}

// ==================== 扩展版调试：CPU扩展解压与多GPU patcher对比 ====================
CompressionInfo test_compression_extended_debug(const std::string &file_path)
{
    std::vector<double> oriData = read_data(file_path);
    size_t original_num_elements = oriData.size();
    if (original_num_elements == 0) return CompressionInfo{};
    constexpr size_t VECTOR_SIZE = 1024; constexpr unsigned UNPACK_N_VECTORS = 1;
    size_t num_elements = original_num_elements; std::vector<double> padded; const double* data_ptr = oriData.data();
    size_t n_vecs_orig = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    size_t n_vecs_padded = ((n_vecs_orig + 4 - 1) / 4) * 4; // 与扩展主函数保持 4 vectors/block
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
    std::cout << "[DEBUG-INFO] 原始元素=" << original_num_elements << ", 原始向量=" << n_vecs_orig 
              << ", 填充后向量=" << n_vecs_padded << ", 填充后元素=" << num_elements_padded << std::endl;
    if (num_elements_padded != original_num_elements) { size_t pad = num_elements_padded - original_num_elements; num_elements = num_elements_padded; padded.reserve(num_elements); padded.insert(padded.end(), oriData.begin(), oriData.end()); double pv = oriData.back(); padded.insert(padded.end(), pad, pv); data_ptr = padded.data(); }
    auto base_col = alp::encode<double>(data_ptr, num_elements, false);
    std::cout << "[DEBUG-INFO] 基础列 n_values=" << base_col.ffor.bp.n_values << ", 向量数=" << utils::get_n_vecs_from_size(base_col.ffor.bp.n_values) << std::endl;
    auto ext_col = base_col.create_extended_column();
    std::cout << "[DEBUG-INFO] 扩展列 n_values=" << ext_col.ffor.bp.n_values << ", 向量数=" << utils::get_n_vecs_from_size(ext_col.ffor.bp.n_values) << std::endl;
    // 检查出错向量717的异常信息
    size_t err_vec = 717; constexpr size_t N_LANES = 16;
    if (err_vec < utils::get_n_vecs_from_size(ext_col.ffor.bp.n_values)) {
        std::cout << "[DEBUG-INFO] 向量" << err_vec << " 异常详情:" << std::endl;
        std::cout << "  基础版 count=" << base_col.counts[err_vec] << ", exc_offset=" << base_col.exceptions_offsets[err_vec] << std::endl;
        
        // ===== 关键发现:扩展格式的根本限制 =====
        std::cout << "\n[关键限制] offsets_counts 编码格式分析:" << std::endl;
        std::cout << "  uint16_t编码: [15:10]=count(6位,最大63), [9:0]=offset(10位,最大1023)" << std::endl;
        std::cout << "  向量717异常数=" << base_col.counts[err_vec] << ", 需分配到16个lane" << std::endl;
        std::cout << "  平均每lane=" << (base_col.counts[err_vec] / 16.0) << " 个异常" << std::endl;
        
        // 检查是否有lane超出6位count限制
        std::cout << "\n  各lane异常分布 (检查是否超出63限制):" << std::endl;
        uint16_t lane_exc_counts[N_LANES] = {0};
        for (uint32_t i = 0; i < base_col.counts[err_vec]; ++i) {
            uint16_t pos = base_col.positions[base_col.exceptions_offsets[err_vec] + i];
            uint32_t lane = pos % N_LANES;
            lane_exc_counts[lane]++;
        }
        bool has_overflow = false;
        for (size_t lane = 0; lane < N_LANES; ++lane) {
            if (lane_exc_counts[lane] > 63) {
                std::cout << "    ❌ lane" << lane << ": " << lane_exc_counts[lane] 
                          << " 个异常 (超出6位限制63!)" << std::endl;
                has_overflow = true;
            } else if (lane_exc_counts[lane] > 0) {
                std::cout << "    ✓ lane" << lane << ": " << lane_exc_counts[lane] << " 个异常" << std::endl;
            }
        }
        
        if (has_overflow) {
            std::cout << "\n  ⚠️ 结论: 扩展格式设计缺陷 - 6位count无法编码超过63个异常的lane!" << std::endl;
            std::cout << "  当向量异常数过多时,部分lane会超出63限制,导致count溢出截断。" << std::endl;
            std::cout << "  被截断的异常无法被GPU patcher正确恢复,导致解压失败。" << std::endl;
            std::cout << "\n  源码证据 (alp.cuh):" << std::endl;
            std::cout << "    count[v] = offset_count >> 10;  // 右移10位,count仅占高6位" << std::endl;
            std::cout << "    offset = (offset_count & 0x3FF);  // 0x3FF=1023,offset占低10位" << std::endl;
            std::cout << "    uint16_t编码: bits[15:10]=count(max=63), bits[9:0]=offset(max=1023)" << std::endl;
        }
        
        uint16_t total_exc = 0;
        for (size_t lane = 0; lane < N_LANES; ++lane) {
            uint16_t oc = ext_col.offsets_counts[err_vec * N_LANES + lane];
            uint16_t cnt = oc >> 10; uint16_t off = oc & 0x3FF;
            total_exc += cnt;
        }
        std::cout << "\n  扩展列实际存储总异常=" << total_exc << " (原始=" << base_col.counts[err_vec] 
                  << ", 丢失=" << (base_col.counts[err_vec] - total_exc) << ")" << std::endl;
        
        std::cout << "\n[解决方案建议]" << std::endl;
        std::cout << "  1. 使用基础格式(ALPColumn)解压 - 无lane异常数限制" << std::endl;
        std::cout << "  2. 修改ALP-G源码,将offsets_counts改为uint32_t(10位count,22位offset)" << std::endl;
        std::cout << "  3. 针对高异常向量进行数据预处理或降低压缩系数" << std::endl;
    }
    std::cout << "[DEBUG] CPU 扩展解压: SKIP (CPU不支持扩展列)" << std::endl;
    // GPU 多 patcher
    auto d_ext = ext_col.copy_to_device(); cudaDeviceSynchronize();
    struct V{ enums::Patcher p; const char* name; }; V vars[]={{enums::Patcher::Naive,"Naive"},{enums::Patcher::NaiveBranchless,"NaiveBranchless"},{enums::Patcher::PrefetchAll,"PrefetchAll"},{enums::Patcher::PrefetchAllBranchless,"PrefetchAllBranchless"}};
    for(auto &v: vars){ double* out=nullptr; try{ out = bindings::decompress_column<double, flsgpu::device::ALPExtendedColumn<double>>(d_ext, UNPACK_N_VECTORS, 1, enums::Unpacker::StatefulBranchless, v.p, 1); cudaDeviceSynchronize(); bool ok = memcmp(data_ptr, out, d_ext.n_values*sizeof(double))==0; std::cout << "[DEBUG] GPU patcher="<<v.name<<" 验证="<<(ok?"OK":"FAIL") << std::endl; if(!ok){ int shown=0; for(size_t i=0;i<d_ext.n_values && shown<5;++i){ if (std::abs(data_ptr[i]-out[i])>1e-10){ std::cout<<"  GPU差异("<<v.name<<") i="<<i<<" exp="<<data_ptr[i]<<" got="<<out[i]<<std::endl; ++shown; } } } } catch(const std::exception &e){ std::cout << "[DEBUG] GPU patcher="<<v.name<<" 异常: "<< e.what() << std::endl; } if(out) delete[] out; }
    flsgpu::host::free_column(d_ext); flsgpu::host::free_column(ext_col); flsgpu::host::free_column(base_col); cudaDeviceSynchronize();
    size_t original_size = original_num_elements*sizeof(double); double ratio = (double)base_col.compressed_size_bytes_alp_extended / (double)original_size;
    return CompressionInfo{ original_size/(1024.0*1024.0), base_col.compressed_size_bytes_alp_extended/(1024.0*1024.0), ratio,0,0,0,0,0,0};
}

// ==================== 扩展格式兼容性分析 ====================
void analyze_extended_compatibility(const std::string &file_path) {
    std::vector<double> oriData = read_data(file_path);
    size_t original_num_elements = oriData.size();
    if (original_num_elements == 0) return;
    
    constexpr size_t VECTOR_SIZE = 1024;
    constexpr size_t N_LANES = 16;
    constexpr uint16_t MAX_LANE_COUNT = 63; // 6位count限制
    
    // ===== 步骤1: 先压缩原始数据(无填充) =====
    std::cout << "\n========== 步骤1: 压缩原始数据(无填充) ==========" << std::endl;
    size_t n_vecs_orig = (original_num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    std::cout << "原始元素数: " << original_num_elements << std::endl;
    std::cout << "原始向量数: " << n_vecs_orig << std::endl;
    
    // 临时填充到1024倍数(ALP要求,但不填充到4倍数)
    size_t temp_padded = n_vecs_orig * VECTOR_SIZE;
    std::vector<double> temp_data;
    const double* temp_ptr = oriData.data();
    if (temp_padded != original_num_elements) {
        temp_data.reserve(temp_padded);
        temp_data.insert(temp_data.end(), oriData.begin(), oriData.end());
        temp_data.insert(temp_data.end(), temp_padded - original_num_elements, oriData.back());
        temp_ptr = temp_data.data();
    }
    auto orig_col = alp::encode<double>(temp_ptr, temp_padded, false);
    
    std::cout << "原始压缩总异常数: " << orig_col.n_exceptions << std::endl;
    std::cout << "原始平均每向量异常: " << (double)orig_col.n_exceptions / n_vecs_orig << std::endl;
    
    // 检查最后一个原始向量(n_vecs_orig-1)的异常数
    if (n_vecs_orig > 0) {
        size_t last_orig_vec = n_vecs_orig - 1;
        std::cout << "最后原始向量[" << last_orig_vec << "] 异常数: " << orig_col.counts[last_orig_vec] << std::endl;
    }
    
    // ===== 步骤2: 压缩填充后数据(4倍数对齐) =====
    std::cout << "\n========== 步骤2: 压缩4倍数对齐填充数据 ==========" << std::endl;
    size_t num_elements = original_num_elements;
    std::vector<double> padded;
    const double* data_ptr = oriData.data();
    
    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    size_t n_vecs_padded = ((n_vecs + 4 - 1) / 4) * 4;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
    
    std::cout << "填充后向量数: " << n_vecs_padded << " (增加 " << (n_vecs_padded - n_vecs_orig) << " 个向量)" << std::endl;
    std::cout << "填充后元素数: " << num_elements_padded << " (增加 " << (num_elements_padded - original_num_elements) << " 个元素)" << std::endl;
    
    if (num_elements_padded != original_num_elements) {
        size_t pad = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;
        padded.reserve(num_elements);
        padded.insert(padded.end(), oriData.begin(), oriData.end());
        padded.insert(padded.end(), pad, oriData.back());
        data_ptr = padded.data();
    }
    
    auto base_col = alp::encode<double>(data_ptr, num_elements, false);
    
    std::cout << "填充后压缩总异常数: " << base_col.n_exceptions << std::endl;
    std::cout << "填充后平均每向量异常: " << (double)base_col.n_exceptions / n_vecs_padded << std::endl;
    
    // 对比填充向量的异常数
    std::cout << "\n========== 填充向量异常分析 ==========" << std::endl;
    if (n_vecs_padded > n_vecs_orig) {
        std::cout << "填充向量范围: [" << n_vecs_orig << " - " << (n_vecs_padded-1) << "]" << std::endl;
        size_t padding_exceptions = 0;
        for (size_t v = n_vecs_orig; v < n_vecs_padded; ++v) {
            std::cout << "  填充向量[" << v << "] 异常数: " << base_col.counts[v] << std::endl;
            padding_exceptions += base_col.counts[v];
        }
        std::cout << "填充向量总异常数: " << padding_exceptions << std::endl;
        std::cout << "异常增量: " << (base_col.n_exceptions - orig_col.n_exceptions) << std::endl;
        
        if (padding_exceptions > 0) {
            std::cout << "\n⚠️ 结论: 填充向量产生了 " << padding_exceptions << " 个异常!" << std::endl;
            std::cout << "   这是因为填充值(" << oriData.back() << ")与压缩模型不匹配。" << std::endl;
            std::cout << "   建议: 使用原始向量数(无4倍数对齐)或改进填充策略。" << std::endl;
        } else {
            std::cout << "\n✓ 结论: 填充向量未产生额外异常。" << std::endl;
        }
    }
    
    flsgpu::host::free_column(orig_col);
    
    // ===== 步骤3: 扩展格式兼容性检查 =====
    std::cout << "\n========== 步骤3: 扩展格式兼容性检查 ==========" << std::endl;
    std::cout << "文件: " << file_path << std::endl;
    std::cout << "向量总数: " << base_col.ffor.bp.get_n_vecs() << std::endl;
    std::cout << "异常总数: " << base_col.n_exceptions << std::endl;
    std::cout << "平均每向量异常数: " << (double)base_col.n_exceptions / base_col.ffor.bp.get_n_vecs() << std::endl;
    
    size_t problematic_vecs = 0;
    size_t total_overflow_lanes = 0;
    size_t total_lost_exceptions = 0;
    size_t max_vec_exceptions = 0;
    size_t max_lane_exceptions = 0;
    
    for (size_t vec_idx = 0; vec_idx < base_col.ffor.bp.get_n_vecs(); ++vec_idx) {
        uint32_t vec_exc_count = base_col.counts[vec_idx];
        if (vec_exc_count > max_vec_exceptions) max_vec_exceptions = vec_exc_count;
        
        if (vec_exc_count == 0) continue;
        
        size_t vec_exc_offset = base_col.exceptions_offsets[vec_idx];
        uint16_t lane_counts[N_LANES] = {0};
        
        for (uint32_t i = 0; i < vec_exc_count; ++i) {
            uint16_t pos = base_col.positions[vec_exc_offset + i];
            lane_counts[pos % N_LANES]++;
        }
        
        bool vec_has_overflow = false;
        for (size_t lane = 0; lane < N_LANES; ++lane) {
            if (lane_counts[lane] > max_lane_exceptions) max_lane_exceptions = lane_counts[lane];
            if (lane_counts[lane] > MAX_LANE_COUNT) {
                vec_has_overflow = true;
                total_overflow_lanes++;
                total_lost_exceptions += (lane_counts[lane] - MAX_LANE_COUNT);
            }
        }
        
        if (vec_has_overflow) {
            problematic_vecs++;
            if (problematic_vecs <= 5) {
                std::cout << "  ✗ 不兼容向量[" << vec_idx << "]: 总异常=" << vec_exc_count;
                bool is_padding = (vec_idx >= n_vecs_orig && vec_idx < n_vecs_padded);
                if (is_padding) std::cout << " (填充向量!)";
                std::cout << ", 溢出lane:";
                for (size_t lane = 0; lane < N_LANES; ++lane) {
                    if (lane_counts[lane] > MAX_LANE_COUNT) {
                        std::cout << " lane" << lane << "=" << lane_counts[lane];
                    }
                }
                std::cout << std::endl;
            }
        }
    }
    
    std::cout << "\n========== 分析结果 ==========" << std::endl;
    std::cout << "✗ 不兼容向量数: " << problematic_vecs << " / " << base_col.ffor.bp.get_n_vecs() 
              << " (" << (100.0 * problematic_vecs / base_col.ffor.bp.get_n_vecs()) << "%)" << std::endl;
    std::cout << "✗ 溢出lane总数: " << total_overflow_lanes << std::endl;
    std::cout << "✗ 丢失异常总数: " << total_lost_exceptions << " / " << base_col.n_exceptions 
              << " (" << (100.0 * total_lost_exceptions / base_col.n_exceptions) << "%)" << std::endl;
    std::cout << "📊 最大向量异常数: " << max_vec_exceptions << std::endl;
    std::cout << "📊 最大lane异常数: " << max_lane_exceptions << " (限制=63)" << std::endl;
    
    if (problematic_vecs == 0) {
        std::cout << "\n✓ 结论: 此数据集完全兼容扩展格式!" << std::endl;
    } else {
        std::cout << "\n✗ 结论: 此数据集不兼容扩展格式!" << std::endl;
        std::cout << "  建议使用基础格式(ALPColumn)进行GPU解压。" << std::endl;
    }
    
    flsgpu::host::free_column(base_col);
    cudaDeviceSynchronize();
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
CompressionInfo test_compression(const std::string &file_path)
{
    std::vector<double> oriData = read_data(file_path);
    return comp_ALP_G(oriData);
}
CompressionInfo test_compression_extended(const std::string &file_path)
{
    std::vector<double> oriData = read_data(file_path);
    return comp_ALP_G_Extended(oriData);
}
CompressionInfo test_compression_extended_fixed(const std::string &file_path)
{
    std::vector<double> oriData = read_data(file_path);
    return comp_ALP_G_Extended_Fixed(oriData);
}

CompressionInfo test_beta_compression(const std::string &file_path, int beta)
{
    std::vector<double> oriData = read_data(file_path, beta);
    return comp_ALP_G(oriData);
}

// ==================== Google Test 测试用例 ====================
TEST(ALPGCompressorTest, CompressionDecompression)
{
    std::string dir_path = "../test/data/mew_tsbs";
    bool warmup = false;

    for (const auto &entry : fs::directory_iterator(dir_path))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".csv")
        {
            std::string file_path = entry.path().string();

            CompressionInfo result;

            if (!warmup)
            {
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

int main(int argc, char *argv[])
{

    cudaFree(0); // 初始化 CUDA

    if (argc < 2)
    {
        // 默认运行 Google Test
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }

    std::string arg = argv[1];

    if (arg == "--dir" && argc >= 3)
    {
        // 目录批处理模式
        std::string dir_path = argv[2];
        std::cout << "📁 处理目录: " << dir_path << std::endl;

        // 读取所有CSV文件
        std::vector<std::string> csv_files;
        for (const auto &entry : fs::directory_iterator(dir_path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".csv")
            {
                csv_files.push_back(entry.path().string());
            }
        }

        if (csv_files.empty())
        {
            std::cerr << "❌ 未找到 CSV 文件" << std::endl;
            return 1;
        }

        std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;

        // 预热
        std::cout << "\n=== 预热阶段 ===" << std::endl;
        test_compression(csv_files[0]);
        cudaDeviceSynchronize();

        // 对每个文件进行测试
        for (const auto &file_path : csv_files)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "文件: " << fs::path(file_path).filename() << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo total_result;

            // 3次迭代
            for (int i = 0; i < 3; ++i)
            {
                std::cout << "\n--- 迭代 " << (i + 1) << " ---" << std::endl;
                CompressionInfo result = test_compression(file_path);
                total_result += result;
                cudaDeviceSynchronize();
            }

            // 计算平均值
            total_result = total_result / 3;
            total_result.print();
        }

        return 0;
    }
    else if (arg == "--file-beta" && argc >= 3)
    {
        // Beta 参数扫描模式
        std::string file_path = argv[2];
        std::cout << "🔬 Beta 参数扫描: " << file_path << std::endl;

        // 预热
        test_compression(file_path);
        cudaDeviceSynchronize();

        for (int beta = 4; beta <= 17; ++beta)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Beta = " << beta << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo total_result;

            // 3次迭代
            for (int i = 0; i < 3; ++i)
            {
                CompressionInfo result = test_beta_compression(file_path, beta);
                total_result += result;
                cudaDeviceSynchronize();
            }

            // 计算平均值
            total_result = total_result / 3;
            total_result.print();
            return 0;
        }
    }
    else if (arg == "--extended" && argc >= 3)
    {
        // 扩展版单文件模式
        std::string file_path = argv[2];
        std::cout << "📂 扩展版处理文件: " << file_path << std::endl;
        std::cout << "\n=== 预热(扩展) ===" << std::endl;
        test_compression_extended(file_path);
        cudaDeviceSynchronize();
        CompressionInfo total_result;
        for (int i = 0; i < 3; ++i)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "扩展迭代 " << (i + 1) << std::endl;
            std::cout << "========================================" << std::endl;
            CompressionInfo r = test_compression_extended(file_path);
            total_result += r;
            cudaDeviceSynchronize();
        }
        total_result = total_result / 3;
        std::cout << "\n=== 扩展版平均结果 ===" << std::endl;
        total_result.print();
        return 0;
    }
    else if (arg == "--extended-debug" && argc >= 3)
    {
        std::string file_path = argv[2];
        std::cout << "🧪 扩展版调试: " << file_path << std::endl;
        test_compression_extended_debug(file_path);
        return 0;
    }
    else if (arg == "--analyze-extended" && argc >= 3)
    {
        std::string file_path = argv[2];
        analyze_extended_compatibility(file_path);
        return 0;
    }
    else if (arg == "--extended-fixed" && argc >= 3)
    {
        std::string file_path = argv[2];
        std::cout << "🔧 扩展版修复测试: " << file_path << std::endl;
        test_compression_extended_fixed(file_path);
        cudaDeviceSynchronize();
        std::cout << "\n重复3次验证稳定性:" << std::endl;
        for (int i = 0; i < 3; ++i) {
            std::cout << "\n--- 迭代 " << (i+1) << " ---" << std::endl;
            test_compression_extended_fixed(file_path);
            cudaDeviceSynchronize();
        }
        return 0;
    }
    else
    {
        // 单文件模式
        std::string file_path = arg;
        std::cout << "📂 处理文件: " << file_path << std::endl;

        // 预热
        std::cout << "\n=== 预热 ===" << std::endl;
        test_compression(file_path);
        cudaDeviceSynchronize();

        CompressionInfo total_result;

        // 3次迭代
        for (int i = 0; i < 3; ++i)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "迭代 " << (i + 1) << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo result = test_compression(file_path);
            total_result += result;
            cudaDeviceSynchronize();
        }

        // 计算平均值
        total_result = total_result / 3;
        total_result.print();
        return 0;
    }
}