#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <array>
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

// 函数声明(保持与基础版文件一致命名,内部改为扩展格式)
CompressionInfo comp_ALP_G(std::vector<double> oriData);
CompressionInfo test_compression(const std::string &file_path);
CompressionInfo test_beta_compression(const std::string &file_path, int beta);

template <typename T>
bool can_use_extended(const flsgpu::host::ALPColumn<T> &column)
{
    constexpr size_t N_LANES = utils::get_n_lanes<T>();
    constexpr uint16_t MAX_LANE_COUNT = (1u << 6) - 1; // offsets_counts count field is 6 bits

    const size_t n_vecs = column.ffor.bp.get_n_vecs();
    for (size_t vec = 0; vec < n_vecs; ++vec)
    {
        const uint32_t exc_count = column.counts[vec];
        if (exc_count == 0)
        {
            continue;
        }

        const size_t exc_offset = column.exceptions_offsets[vec];
        std::array<uint16_t, N_LANES> lane_counts{};
        for (uint32_t i = 0; i < exc_count; ++i)
        {
            const uint16_t pos = column.positions[exc_offset + i];
            const uint16_t lane = static_cast<uint16_t>(pos % N_LANES);
            ++lane_counts[lane];
        }

        for (uint16_t lane_count : lane_counts)
        {
            if (lane_count > MAX_LANE_COUNT)
            {
                return false;
            }
        }
    }

    return true;
}

// ==================== 扩展版压缩 + GPU 解压 ====================
CompressionInfo comp_ALP_G(std::vector<double> oriData)
{
    const size_t original_num_elements = oriData.size();
    const size_t original_size = original_num_elements * sizeof(double);
    if (original_num_elements == 0)
    {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return CompressionInfo{};
    }

    // 与基础版统一的参数
    constexpr size_t VECTOR_SIZE = 1024;
    constexpr unsigned UNPACK_N_VECTORS = 1; // 安全配置
    constexpr size_t N_LANES_DOUBLE = 16;
    constexpr size_t THREADS_PER_WARP = 32;
    constexpr size_t N_WARPS_PER_BLOCK = 2;
    constexpr size_t N_THREADS_PER_BLOCK = N_WARPS_PER_BLOCK * THREADS_PER_WARP;
    constexpr size_t N_CONCURRENT_VECTORS_PER_BLOCK = N_THREADS_PER_BLOCK / N_LANES_DOUBLE; // 64/16=4
    constexpr size_t VECTORS_PER_BLOCK = UNPACK_N_VECTORS * N_CONCURRENT_VECTORS_PER_BLOCK; // 4

    size_t num_elements = original_num_elements;
    std::vector<double> paddedData;
    const double *data_ptr = oriData.data();

    // 1.数据预处理
    // 填充到线程块向量组的倍数
    size_t n_vecs = (num_elements + VECTOR_SIZE - 1) / VECTOR_SIZE;
    size_t n_vecs_padded = ((n_vecs + VECTORS_PER_BLOCK - 1) / VECTORS_PER_BLOCK) * VECTORS_PER_BLOCK;
    size_t num_elements_padded = n_vecs_padded * VECTOR_SIZE;
    if (num_elements_padded != original_num_elements)
    {
        size_t pad_needed = num_elements_padded - original_num_elements;
        num_elements = num_elements_padded;
        paddedData.reserve(num_elements);
        paddedData.insert(paddedData.end(), oriData.begin(), oriData.end());
        paddedData.insert(paddedData.end(), pad_needed, oriData.back());
        data_ptr = paddedData.data();
    }
    // 2.压缩阶段
    auto encode_start = std::chrono::high_resolution_clock::now();
    flsgpu::host::ALPColumn<double> host_base_column;
    try
    {
        host_base_column = alp::encode<double>(data_ptr, num_elements, false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "❌ 基础压缩失败: " << e.what() << std::endl;
        return CompressionInfo{};
    }
    // 计时
    auto encode_end = std::chrono::high_resolution_clock::now();
    const double compression_time_ms = std::chrono::duration<double, std::milli>(encode_end - encode_start).count();
    constexpr double compression_kernel_time_ms = 0.0;
    
    // 判断使用扩展版 or 基础版
    const bool use_extended = can_use_extended(host_base_column);
    const size_t compressed_size = use_extended ? host_base_column.compressed_size_bytes_alp_extended
                                                : host_base_column.compressed_size_bytes_alp;
    const double compression_ratio = static_cast<double>(compressed_size) / original_size;
    std::cout << "✓ 压缩完成(" << (use_extended ? "扩展" : "基础") << "): "
              << compressed_size << " bytes, 比率=" << compression_ratio << "x" << std::endl;

    // 创建 CUDA 事件用于计时
    cudaEvent_t kernel_start{};
    cudaEvent_t kernel_stop{};
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    auto decomp_start = std::chrono::high_resolution_clock::now();
    // 解压过程
    if (use_extended)
    {

        flsgpu::host::ALPExtendedColumn<double> host_extended_column = host_base_column.create_extended_column();
        flsgpu::device::ALPExtendedColumn<double> device_extended_column;
        // decomp_start = std::chrono::high_resolution_clock::now();
        // 数据转移
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
        
        float kernel_elapsed_ms = 0.0f;
        double *host_decompressed_data = nullptr;
        // 解压缩
        try
        {
            cudaEventRecord(kernel_start);
            host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPExtendedColumn<double>>(
                device_extended_column,
                UNPACK_N_VECTORS,
                1,
                enums::Unpacker::StatefulBranchless,
                enums::Patcher::PrefetchAll,
                1);
            cudaDeviceSynchronize();
            cudaEventRecord(kernel_stop);
            cudaEventSynchronize(kernel_stop);
            cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
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
            cudaEventDestroy(kernel_start);
            cudaEventDestroy(kernel_stop);
            return CompressionInfo{};
        }
        auto decomp_end = std::chrono::high_resolution_clock::now();

        // 时间统计
        const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
        const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);
        
        // 验证解压数据
        const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
        const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
        const size_t decomp_size = device_extended_column.n_values * sizeof(double);
        if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
        {
            std::cout << "❌ 扩展格式数据验证失败" << std::endl;
            const double *p = data_ptr;
            const double *d = host_decompressed_data;
            int shown = 0;
            for (size_t i = 0; i < device_extended_column.n_values && shown < 10; ++i)
            {
                if (std::abs(p[i] - d[i]) > 1e-10)
                {
                    std::cout << "  不匹配[" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
                    ++shown;
                }
            }
        }
        else
        {
            std::cout << "✓ 扩展格式数据验证成功" << std::endl;
        }

        // 结果返回
        const double comp_tp = compression_time_ms > 0.0 ? (original_size / 1e9) / (compression_time_ms / 1000.0) : 0.0;
        const double decomp_tp = decompression_time_ms > 0.0 ? (original_size / 1e9) / (decompression_time_ms / 1000.0) : 0.0;

        CompressionInfo result{
            original_size / (1024.0 * 1024.0),
            compressed_size / (1024.0 * 1024.0),
            compression_ratio,
            compression_kernel_time_ms,
            compression_time_ms,
            comp_tp,
            decompression_kernel_time_ms,
            decompression_time_ms,
            decomp_tp};

        delete[] host_decompressed_data;
        flsgpu::host::free_column(device_extended_column);
        flsgpu::host::free_column(host_extended_column);
        flsgpu::host::free_column(host_base_column);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        cudaDeviceSynchronize();
        return result;
    }
    else{
        // 数据传输
        flsgpu::device::ALPColumn<double> device_base_column;
        try
        {
            device_base_column = host_base_column.copy_to_device();
            cudaDeviceSynchronize();
        }
        catch (const std::exception &e)
        {
            std::cerr << "❌ 基础列复制到 GPU 失败: " << e.what() << std::endl;
            flsgpu::host::free_column(host_base_column);
            return CompressionInfo{};
        }

        float kernel_elapsed_ms = 0.0f;
        double *host_decompressed_data = nullptr;
        // 基础版本解压缩
        try
        {
            cudaEventRecord(kernel_start);
            host_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPColumn<double>>(
                device_base_column,
                UNPACK_N_VECTORS,
                1,
                enums::Unpacker::StatefulBranchless,
                enums::Patcher::Stateful,
                1);
            cudaEventRecord(kernel_stop);
            cudaEventSynchronize(kernel_stop);
            cudaEventElapsedTime(&kernel_elapsed_ms, kernel_start, kernel_stop);
            cudaDeviceSynchronize();
            if (!host_decompressed_data)
            {
                throw std::runtime_error("基础GPU解压返回 nullptr");
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "❌ 基础GPU解压失败: " << e.what() << std::endl;
            if (host_decompressed_data)
                delete[] host_decompressed_data;
            flsgpu::host::free_column(device_base_column);
            flsgpu::host::free_column(host_base_column);
            cudaEventDestroy(kernel_start);
            cudaEventDestroy(kernel_stop);
            return CompressionInfo{};
        }
        // 时间统计
        auto decomp_end = std::chrono::high_resolution_clock::now();
        const double decompression_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
        const double decompression_kernel_time_ms = static_cast<double>(kernel_elapsed_ms);

        // 数据验证
        const uint8_t *padded_bytes = reinterpret_cast<const uint8_t *>(data_ptr);
        const uint8_t *decomp_bytes = reinterpret_cast<const uint8_t *>(host_decompressed_data);
        const size_t decomp_size = device_base_column.n_values * sizeof(double);
        if (memcmp(padded_bytes, decomp_bytes, decomp_size) != 0)
        {
            std::cout << "❌ 基础格式数据验证失败" << std::endl;
            const double *p = data_ptr;
            const double *d = host_decompressed_data;
            int shown = 0;
            for (size_t i = 0; i < device_base_column.n_values && shown < 10; ++i)
            {
                if (std::abs(p[i] - d[i]) > 1e-10)
                {
                    std::cout << "  不匹配[" << i << "]: " << p[i] << " vs " << d[i] << std::endl;
                    ++shown;
                }
            }
        }
        else
        {
            std::cout << "✓ 基础格式数据验证成功" << std::endl;
        }

        // 结果返回
        const double comp_tp = compression_time_ms > 0.0 ? (original_size / 1e9) / (compression_time_ms / 1000.0) : 0.0;
        const double decomp_tp = decompression_time_ms > 0.0 ? (original_size / 1e9) / (decompression_time_ms / 1000.0) : 0.0;

        CompressionInfo result{
            original_size / (1024.0 * 1024.0),
            compressed_size / (1024.0 * 1024.0),
            compression_ratio,
            compression_kernel_time_ms,
            compression_time_ms,
            comp_tp,
            decompression_kernel_time_ms,
            decompression_time_ms,
            decomp_tp};

        delete[] host_decompressed_data;
        flsgpu::host::free_column(device_base_column);
        flsgpu::host::free_column(host_base_column);
        cudaEventDestroy(kernel_start);
        cudaEventDestroy(kernel_stop);
        cudaDeviceSynchronize();
        return result;
    }
}

// ==================== 文件测试包装函数 ====================
CompressionInfo test_compression(const std::string &file_path)
{
    std::vector<double> oriData = read_data(file_path);
    return comp_ALP_G(oriData); // 调用扩展版实现
}

CompressionInfo test_beta_compression(const std::string &file_path, int beta)
{
    std::vector<double> oriData = read_data(file_path, beta);
    return comp_ALP_G(oriData);
}

// ==================== Google Test 用例(保持结构一致) ====================
TEST(ALPGExtendedCompressorTest, CompressionDecompression)
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
                test_compression(file_path);
                cudaDeviceSynchronize();
                warmup = true;
            }
            result = test_compression(file_path);
            EXPECT_GT(result.compression_ratio, 0.0);
            EXPECT_GT(result.comp_throughput, 0.0);
            EXPECT_GT(result.decomp_throughput, 0.0);
        }
    }
}

// ==================== 主程序 ====================
int main(int argc, char *argv[])
{
    cudaFree(0); // 初始化 CUDA

    if (argc < 2)
    {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }

    std::string arg = argv[1];

    if (arg == "--dir" && argc >= 3)
    {
        std::string dir_path = argv[2];
        std::cout << "📁 扩展格式处理目录: " << dir_path << std::endl;

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
        std::cout << "\n=== 预热阶段(扩展) ===" << std::endl;
        test_compression(csv_files[0]);
        cudaDeviceSynchronize();

        for (const auto &file_path : csv_files)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "文件: " << fs::path(file_path).filename() << std::endl;
            std::cout << "========================================" << std::endl;

            CompressionInfo total_result;
            for (int i = 0; i < 3; ++i)
            {
                std::cout << "\n--- 迭代 " << (i + 1) << " ---" << std::endl;
                CompressionInfo r = test_compression(file_path);
                total_result += r;
                cudaDeviceSynchronize();
            }
            total_result = total_result / 3;
            total_result.print();
        }
        return 0;
    }
    else if (arg == "--file-beta" && argc >= 3)
    {
        std::string file_path = argv[2];
        std::cout << "🔬 扩展格式 Beta 参数扫描: " << file_path << std::endl;
        test_compression(file_path);
        cudaDeviceSynchronize();
        for (int beta = 4; beta <= 17; ++beta)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "Beta = " << beta << std::endl;
            std::cout << "========================================" << std::endl;
            CompressionInfo total_result;
            for (int i = 0; i < 3; ++i)
            {
                CompressionInfo r = test_beta_compression(file_path, beta);
                total_result += r;
                cudaDeviceSynchronize();
            }
            total_result = total_result / 3;
            total_result.print();
            return 0;
        }
    }
    else
    {
        std::string file_path = arg;
        std::cout << "📂 扩展格式处理文件: " << file_path << std::endl;
        std::cout << "\n=== 预热(扩展) ===" << std::endl;
        test_compression(file_path);
        cudaDeviceSynchronize();

        CompressionInfo total_result;
        for (int i = 0; i < 3; ++i)
        {
            std::cout << "\n========================================" << std::endl;
            std::cout << "迭代 " << (i + 1) << std::endl;
            std::cout << "========================================" << std::endl;
            CompressionInfo r = test_compression(file_path);
            total_result += r;
            cudaDeviceSynchronize();
        }
        total_result = total_result / 3;
        total_result.print();
        return 0;
    }
}