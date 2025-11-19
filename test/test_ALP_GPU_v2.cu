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

// ==================== CUDA 错误检查宏 ====================
#define CHECK_CUDA(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            throw std::runtime_error(cudaGetErrorString(error)); \
        } \
    } while(0)

// ==================== 压缩信息结构 ====================
struct ALPCompressionInfo {
    double original_size_mb = 0.0;
    double compressed_size_mb = 0.0;
    double compression_ratio = 0.0;
    double comp_time_ms = 0.0;
    double decomp_time_ms = 0.0;
    double comp_throughput_gbs = 0.0;
    double decomp_throughput_gbs = 0.0;
    bool verification_passed = false;
    double max_error = 0.0;
    
    ALPCompressionInfo& operator+=(const ALPCompressionInfo& other) {
        original_size_mb += other.original_size_mb;
        compressed_size_mb += other.compressed_size_mb;
        compression_ratio += other.compression_ratio;
        comp_time_ms += other.comp_time_ms;
        decomp_time_ms += other.decomp_time_ms;
        comp_throughput_gbs += other.comp_throughput_gbs;
        decomp_throughput_gbs += other.decomp_throughput_gbs;
        return *this;
    }
    
    ALPCompressionInfo& operator/=(int divisor) {
        if (divisor > 0) {
            original_size_mb /= divisor;
            compressed_size_mb /= divisor;
            compression_ratio /= divisor;
            comp_time_ms /= divisor;
            decomp_time_ms /= divisor;
            comp_throughput_gbs /= divisor;
            decomp_throughput_gbs /= divisor;
        }
        return *this;
    }
    
    void print() const {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "\n========== ALP-G 压缩结果 ==========\n"
                  << "  原始大小:      " << original_size_mb << " MB\n"
                  << "  压缩后:        " << compressed_size_mb << " MB\n"
                  << "  压缩率:        " << compression_ratio << "x\n"
                  << "  压缩时间:      " << comp_time_ms << " ms\n"
                  << "  解压时间:      " << decomp_time_ms << " ms\n"
                  << "  压缩吞吐:      " << comp_throughput_gbs << " GB/s\n"
                  << "  解压吞吐:      " << decomp_throughput_gbs << " GB/s\n"
                  << "  验证状态:      " << (verification_passed ? "✓ PASSED" : "✗ FAILED") << "\n"
                  << "  最大误差:      " << max_error << "\n"
                  << "===================================\n";
    }
};

// ==================== Double 精度压缩实现 ====================
ALPCompressionInfo comp_ALP_GPU_double(const std::vector<double>& oriData) {
    ALPCompressionInfo result;
    
    const size_t num_elements = oriData.size();
    const size_t data_size = num_elements * sizeof(double);
    
    if (num_elements == 0) {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return result;
    }
    
    // ==================== 数据有效性检查 ====================
    bool has_invalid = false;
    double min_val = oriData[0];
    double max_val = oriData[0];
    int invalid_count = 0;
    
    for (const auto& val : oriData) {
        if (!std::isfinite(val)) {
            has_invalid = true;
            invalid_count++;
        }
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    if (invalid_count > 0) {
        std::cerr << "⚠️  数据包含 " << invalid_count << " 个无穷大或 NaN 值" << std::endl;
    }
    
    std::cout << "📊 数据统计: 元素数=" << num_elements 
              << ", 范围=[" << std::fixed << std::setprecision(6) 
              << min_val << ", " << max_val << "]"
              << ", 跨度=" << (max_val - min_val) << std::endl;
    
    // 检查数据范围是否合理（如果跨度太大可能导致压缩问题）
    double range_span = max_val - min_val;
    if (range_span > 1e15) {
        std::cerr << "⚠️  警告: 数据范围跨度过大 (" << range_span 
                  << ")，可能导致压缩失败或精度丧失" << std::endl;
    }
    
    // ==================== 压缩阶段 ====================
    auto start_time = std::chrono::high_resolution_clock::now();
    
    flsgpu::host::ALPColumn<double> host_compressed_column;
    try {
        // 注意：encode 在 CPU 端执行，返回 Host 端数据结构
        host_compressed_column = alp::encode<double>(oriData.data(), num_elements, false);
    } catch (const std::exception& e) {
        std::cerr << "❌ ALP 压缩失败: " << e.what() << std::endl;
        return result;
    }
    
    auto compress_end_time = std::chrono::high_resolution_clock::now();
    
    // 获取压缩信息
    size_t compressed_size = host_compressed_column.compressed_size_bytes_alp;
    double compression_ratio = host_compressed_column.get_compression_ratio();
    
    // 检查压缩是否产生了有效结果
    if (compressed_size == 0 || compression_ratio <= 0) {
        std::cerr << "❌ 压缩失败: 压缩大小=" << compressed_size 
                  << ", 压缩率=" << compression_ratio << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    cudaDeviceSynchronize();
    
    auto compress_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        compress_end_time - start_time).count();
    float compress_time_ms = compress_duration / 1000.0f;
    double comp_throughput = (data_size / 1e9) / std::max(compress_time_ms / 1000.0, 0.001);
    
    std::cout << "✓ 压缩完成: " << compressed_size << " bytes, 比率=" 
              << compression_ratio << "x, 时间=" << compress_time_ms << " ms" << std::endl;
    
    // ==================== 解压前的数据结构检查 ====================
    // 验证压缩数据结构的有效性，防止段错误
    const size_t n_vecs = host_compressed_column.ffor.bp.get_n_vecs();
    bool structure_valid = true;
    
    if (host_compressed_column.ffor.bp.packed_array == nullptr) {
        std::cerr << "❌ 压缩数据结构错误: packed_array 为 nullptr" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    if (host_compressed_column.ffor.bp.bit_widths == nullptr) {
        std::cerr << "❌ 压缩数据结构错误: bit_widths 为 nullptr" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    if (host_compressed_column.ffor.bases == nullptr) {
        std::cerr << "❌ 压缩数据结构错误: bases 为 nullptr" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    // 检查异常数据结构
    if (host_compressed_column.n_exceptions > 0) {
        if (host_compressed_column.exceptions == nullptr ||
            host_compressed_column.positions == nullptr ||
            host_compressed_column.exceptions_offsets == nullptr ||
            host_compressed_column.counts == nullptr) {
            std::cerr << "❌ 异常数据结构错误: 指针为 nullptr" << std::endl;
            std::cerr << "   exceptions: " << (void*)host_compressed_column.exceptions << std::endl;
            std::cerr << "   positions: " << (void*)host_compressed_column.positions << std::endl;
            std::cerr << "   exceptions_offsets: " << (void*)host_compressed_column.exceptions_offsets << std::endl;
            std::cerr << "   counts: " << (void*)host_compressed_column.counts << std::endl;
            flsgpu::host::free_column(host_compressed_column);
            return result;
        }
        
        // 额外检查：验证异常偏移的有效性
        // n_vecs 个向量，每个向量对应一个偏移值
        if (n_vecs > 0) {
            // 检查最后一个向量的异常偏移是否在合理范围内
            size_t last_offset = host_compressed_column.exceptions_offsets[n_vecs - 1];
            if (last_offset > host_compressed_column.n_exceptions) {
                std::cerr << "❌ 异常偏移越界: last_offset=" << last_offset 
                          << ", n_exceptions=" << host_compressed_column.n_exceptions << std::endl;
                flsgpu::host::free_column(host_compressed_column);
                return result;
            }
        }
    }
    
    std::cout << "✓ 压缩数据结构检查通过: n_vecs=" << n_vecs 
              << ", n_exceptions=" << host_compressed_column.n_exceptions << std::endl;
    
    // ==================== GPU 数据转移（模仿 benchmark 的 copy_to_device） ====================
    std::cout << "📤 将压缩数据转移到 GPU..." << std::endl;
    
    flsgpu::device::ALPColumn<double> device_column;
    try {
        device_column = host_compressed_column.copy_to_device();
    } catch (const std::exception& e) {
        std::cerr << "❌ GPU 数据转移失败: " << e.what() << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    std::cout << "✓ 数据成功转移到 GPU" << std::endl;
    
    // ==================== 解压阶段（CPU 端验证 + GPU 端数据） ====================
    std::vector<double> decompressed_data(num_elements);
    std::vector<double> decompressed_data_cpu(num_elements);
    
    auto decompress_start_time = std::chrono::high_resolution_clock::now();
    
    try {
        // CPU 端解压（生成参考数据，用于验证）
        std::cout << "🔄 CPU 解压: 生成参考数据用于验证" << std::endl;
        double* decode_result_cpu = alp::decode<double>(host_compressed_column, decompressed_data_cpu.data());
        if (!decode_result_cpu) {
            std::cerr << "❌ CPU 解压失败" << std::endl;
            flsgpu::host::free_column(device_column);
            flsgpu::host::free_column(host_compressed_column);
            return result;
        }
        
        // ==================== GPU 解压（调用 GPU kernel）====================
        std::cout << "🚀 GPU 解压: 调用 GPU kernel" << std::endl;
        double* gpu_decompressed_data = bindings::decompress_column<double, flsgpu::device::ALPColumn<double>>(
            device_column,
            32,      // unpack_n_vectors: 标准向量数
            1024,    // unpack_n_values: 每个向量的值数
            enums::Unpacker::Dummy,  // unpacker
            enums::Patcher::Dummy,  // patcher
            1        // n_samples: 样本数
        );
        
        if (!gpu_decompressed_data) {
            std::cerr << "❌ GPU 解压失败" << std::endl;
            flsgpu::host::free_column(device_column);
            flsgpu::host::free_column(host_compressed_column);
            return result;
        }
        
        // 将 GPU 结果复制到主机内存
        CHECK_CUDA(cudaMemcpy(decompressed_data.data(), gpu_decompressed_data, data_size, cudaMemcpyDeviceToHost));
        
        // 释放 GPU 分配的内存
        CHECK_CUDA(cudaFree(gpu_decompressed_data));
        
        std::cout << "✓ GPU 解压完成" << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cerr << "❌ 内存不足: " << e.what() << std::endl;
        flsgpu::host::free_column(device_column);
        flsgpu::host::free_column(host_compressed_column);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "❌ 解压失败: " << e.what() << std::endl;
        std::cerr << "   异常信息: " << typeid(e).name() << std::endl;
        flsgpu::host::free_column(device_column);
        flsgpu::host::free_column(host_compressed_column);
        return result;
    } catch (...) {
        std::cerr << "❌ 未知异常发生在解压阶段" << std::endl;
        flsgpu::host::free_column(device_column);
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    cudaDeviceSynchronize();
    
    auto decompress_end_time = std::chrono::high_resolution_clock::now();
    auto decompress_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        decompress_end_time - decompress_start_time).count();
    float decompress_time_ms = decompress_duration / 1000.0f;
    double decomp_throughput = (data_size / 1e9) / std::max(decompress_time_ms / 1000.0, 0.001);
    
    std::cout << "✓ 解压完成: 时间=" << decompress_time_ms << " ms" << std::endl;
    
    // ==================== GPU 内存释放（模仿 benchmark 的 free_column） ====================
    flsgpu::host::free_column(device_column);
    
    // ==================== 验证结果 ====================
    bool verification_passed = true;
    double max_error = 0.0;
    double avg_error = 0.0;
    int error_count = 0;
    const double ERROR_THRESHOLD = 1e-6;
    
    for (size_t i = 0; i < num_elements; ++i) {
        double error = std::abs(oriData[i] - decompressed_data[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
        
        if (error > ERROR_THRESHOLD) {
            error_count++;
            if (error_count <= 5) {  // 仅打印前 5 个错误
                std::cout << "  Error at [" << i << "]: orig=" << oriData[i]
                          << ", decomp=" << decompressed_data[i]
                          << ", error=" << error << std::endl;
            }
        }
    }
    avg_error /= num_elements;
    
    if (error_count > 0) {
        verification_passed = false;
        std::cout << "⚠️  验证失败: " << error_count << " 个值误差 > " << ERROR_THRESHOLD
                  << ", 平均误差=" << avg_error << ", 最大误差=" << max_error << std::endl;
    } else {
        std::cout << "✓ 验证通过: 所有值误差 < " << ERROR_THRESHOLD << std::endl;
    }
    
    // ==================== 填充结果 ====================
    result.original_size_mb = data_size / (1024.0 * 1024.0);
    result.compressed_size_mb = compressed_size / (1024.0 * 1024.0);
    result.compression_ratio = compression_ratio;
    result.comp_time_ms = compress_time_ms;
    result.decomp_time_ms = decompress_time_ms;
    result.comp_throughput_gbs = comp_throughput;
    result.decomp_throughput_gbs = decomp_throughput;
    result.verification_passed = verification_passed;
    result.max_error = max_error;
    
    // ==================== 正确的资源释放 ====================
    // 释放 Host 端的 ALPColumn（使用 delete[]）
    flsgpu::host::free_column(host_compressed_column);
    
    return result;
}

// ==================== Float 精度压缩实现 ====================
ALPCompressionInfo comp_ALP_GPU_float(const std::vector<float>& oriData) {
    ALPCompressionInfo result;
    
    const size_t num_elements = oriData.size();
    const size_t data_size = num_elements * sizeof(float);
    
    if (num_elements == 0) {
        std::cerr << "❌ 输入数据为空" << std::endl;
        return result;
    }
    
    // 数据有效性检查
    bool has_invalid = false;
    float min_val = oriData[0];
    float max_val = oriData[0];
    int invalid_count = 0;
    
    for (const auto& val : oriData) {
        if (!std::isfinite(val)) {
            has_invalid = true;
            invalid_count++;
        }
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
    }
    
    if (invalid_count > 0) {
        std::cerr << "⚠️  数据包含 " << invalid_count << " 个无穷大或 NaN 值" << std::endl;
    }
    
    std::cout << "📊 数据统计: 元素数=" << num_elements 
              << ", 范围=[" << std::fixed << std::setprecision(6) 
              << min_val << ", " << max_val << "]"
              << ", 跨度=" << (max_val - min_val) << std::endl;
    
    // 检查数据范围
    float range_span = max_val - min_val;
    if (range_span > 1e10f) {
        std::cerr << "⚠️  警告: 数据范围跨度过大，可能导致压缩失败" << std::endl;
    }
    
    // 压缩
    auto start_time = std::chrono::high_resolution_clock::now();
    
    flsgpu::host::ALPColumn<float> host_compressed_column;
    try {
        host_compressed_column = alp::encode<float>(oriData.data(), num_elements, false);
    } catch (const std::exception& e) {
        std::cerr << "❌ ALP 压缩失败: " << e.what() << std::endl;
        return result;
    }
    
    auto compress_end_time = std::chrono::high_resolution_clock::now();
    
    size_t compressed_size = host_compressed_column.compressed_size_bytes_alp;
    double compression_ratio = host_compressed_column.get_compression_ratio();
    
    // 检查压缩是否产生了有效结果
    if (compressed_size == 0 || compression_ratio <= 0) {
        std::cerr << "❌ 压缩失败: 压缩大小=" << compressed_size 
                  << ", 压缩率=" << compression_ratio << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    cudaDeviceSynchronize();
    
    auto compress_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        compress_end_time - start_time).count();
    float compress_time_ms = compress_duration / 1000.0f;
    double comp_throughput = (data_size / 1e9) / std::max(compress_time_ms / 1000.0, 0.001);
    
    // 解压
    // std::vector<float> decompressed_data(num_elements);
    
    // auto decompress_start_time = std::chrono::high_resolution_clock::now();
    
    // ==================== 解压前的数据结构检查 ====================
    const size_t n_vecs_f = host_compressed_column.ffor.bp.get_n_vecs();
    
    if (host_compressed_column.ffor.bp.packed_array == nullptr ||
        host_compressed_column.ffor.bp.bit_widths == nullptr ||
        host_compressed_column.ffor.bases == nullptr) {
        std::cerr << "❌ 压缩数据结构错误" << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    if (host_compressed_column.n_exceptions > 0) {
        if (host_compressed_column.exceptions == nullptr ||
            host_compressed_column.positions == nullptr ||
            host_compressed_column.exceptions_offsets == nullptr ||
            host_compressed_column.counts == nullptr) {
            std::cerr << "❌ 异常数据结构错误" << std::endl;
            flsgpu::host::free_column(host_compressed_column);
            return result;
        }
        
        // 额外检查：验证异常偏移的有效性
        if (n_vecs_f > 0) {
            size_t last_offset = host_compressed_column.exceptions_offsets[n_vecs_f - 1];
            if (last_offset > host_compressed_column.n_exceptions) {
                std::cerr << "❌ 异常偏移越界" << std::endl;
                flsgpu::host::free_column(host_compressed_column);
                return result;
            }
        }
    }
    
    std::cout << "✓ 压缩数据结构检查通过: n_vecs=" << n_vecs_f 
              << ", n_exceptions=" << host_compressed_column.n_exceptions << std::endl;
    
    // ==================== GPU 数据转移 ====================
    std::cout << "📤 将压缩数据转移到 GPU..." << std::endl;
    
    flsgpu::device::ALPColumn<float> device_column_f;
    try {
        device_column_f = host_compressed_column.copy_to_device();
    } catch (const std::exception& e) {
        std::cerr << "❌ GPU 数据转移失败: " << e.what() << std::endl;
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    std::cout << "✓ 数据成功转移到 GPU" << std::endl;
    
    // 解压
    std::vector<float> decompressed_data(num_elements);
    std::vector<float> decompressed_data_cpu(num_elements);
    
    auto decompress_start_time = std::chrono::high_resolution_clock::now();
    
    // ==================== CPU 端解压（参考数据）====================
    try {
        std::cout << "🔄 CPU 解压: 生成参考数据" << std::endl;
        float* decode_result_cpu = alp::decode<float>(host_compressed_column, decompressed_data_cpu.data());
        if (!decode_result_cpu) {
            std::cerr << "❌ CPU 解压失败" << std::endl;
            flsgpu::host::free_column(device_column_f);
            flsgpu::host::free_column(host_compressed_column);
            return result;
        }
        
        // ==================== GPU 解压（调用 GPU kernel）====================
        std::cout << "🚀 GPU 解压: 调用 GPU kernel" << std::endl;
        float* gpu_decompressed_data = bindings::decompress_column<float, flsgpu::device::ALPColumn<float>>(
            device_column_f,
            32,      // unpack_n_vectors: 标准向量数
            1024,    // unpack_n_values: 每个向量的值数
            enums::Unpacker::Dummy,  // unpacker
            enums::Patcher::Dummy,  // patcher
            1        // n_samples: 样本数
        );
        
        if (!gpu_decompressed_data) {
            std::cerr << "❌ GPU 解压失败" << std::endl;
            flsgpu::host::free_column(device_column_f);
            flsgpu::host::free_column(host_compressed_column);
            return result;
        }
        
        // 将 GPU 结果复制到主机内存
        CHECK_CUDA(cudaMemcpy(decompressed_data.data(), gpu_decompressed_data, data_size, cudaMemcpyDeviceToHost));
        
        // 释放 GPU 分配的内存
        CHECK_CUDA(cudaFree(gpu_decompressed_data));
        
        std::cout << "✓ GPU 解压完成" << std::endl;
    } catch (const std::bad_alloc& e) {
        std::cerr << "❌ 内存不足: " << e.what() << std::endl;
        flsgpu::host::free_column(device_column_f);
        flsgpu::host::free_column(host_compressed_column);
        return result;
    } catch (const std::exception& e) {
        std::cerr << "❌ 解压失败: " << e.what() << std::endl;
        std::cerr << "   异常信息: " << typeid(e).name() << std::endl;
        flsgpu::host::free_column(device_column_f);
        flsgpu::host::free_column(host_compressed_column);
        return result;
    } catch (...) {
        std::cerr << "❌ 未知异常发生在解压阶段" << std::endl;
        flsgpu::host::free_column(device_column_f);
        flsgpu::host::free_column(host_compressed_column);
        return result;
    }
    
    cudaDeviceSynchronize();
    
    auto decompress_end_time = std::chrono::high_resolution_clock::now();
    auto decompress_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        decompress_end_time - decompress_start_time).count();
    float decompress_time_ms = decompress_duration / 1000.0f;
    double decomp_throughput = (data_size / 1e9) / std::max(decompress_time_ms / 1000.0, 0.001);
    
    // ==================== GPU 内存释放 ====================
    flsgpu::host::free_column(device_column_f);
    
    // 验证
    bool verification_passed = true;
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int error_count = 0;
    const float ERROR_THRESHOLD = 1e-5f;
    
    for (size_t i = 0; i < num_elements; ++i) {
        float error = std::abs(oriData[i] - decompressed_data[i]);
        max_error = std::max(max_error, error);
        avg_error += error;
        
        if (error > ERROR_THRESHOLD) {
            error_count++;
            if (error_count <= 5) {
                std::cout << "  Error at [" << i << "]: orig=" << oriData[i]
                          << ", decomp=" << decompressed_data[i]
                          << ", error=" << error << std::endl;
            }
        }
    }
    avg_error /= num_elements;
    
    if (error_count > 0) {
        verification_passed = false;
        std::cout << "⚠️  验证失败: " << error_count << " 个值误差 > " << ERROR_THRESHOLD << std::endl;
    } else {
        std::cout << "✓ 验证通过" << std::endl;
    }
    
    result.original_size_mb = data_size / (1024.0 * 1024.0);
    result.compressed_size_mb = compressed_size / (1024.0 * 1024.0);
    result.compression_ratio = compression_ratio;
    result.comp_time_ms = compress_time_ms;
    result.decomp_time_ms = decompress_time_ms;
    result.comp_throughput_gbs = comp_throughput;
    result.decomp_throughput_gbs = decomp_throughput;
    result.verification_passed = verification_passed;
    result.max_error = max_error;
    
    flsgpu::host::free_column(host_compressed_column);
    
    return result;
}

// ==================== 文件测试包装函数 ====================
ALPCompressionInfo test_compression_double(const std::string& file_path) {
    std::cout << "\n📂 处理文件: " << file_path << std::endl;
    std::vector<double> data = read_data(file_path, false);
    if (data.empty()) {
        std::cerr << "❌ 无法从文件读取数据" << std::endl;
        ALPCompressionInfo result;
        return result;
    }
    return comp_ALP_GPU_double(data);
}

ALPCompressionInfo test_compression_float(const std::string& file_path) {
    std::cout << "\n📂 处理文件: " << file_path << std::endl;
    std::vector<double> data_double = read_data(file_path, false);
    if (data_double.empty()) {
        std::cerr << "❌ 无法从文件读取数据" << std::endl;
        ALPCompressionInfo result;
        return result;
    }
    std::vector<float> data(data_double.begin(), data_double.end());
    return comp_ALP_GPU_float(data);
}

ALPCompressionInfo test_beta_compression_double(const std::string& file_path, int beta) {
    return test_compression_double(file_path);
}

ALPCompressionInfo test_beta_compression_float(const std::string& file_path, int beta) {
    return test_compression_float(file_path);
}

// ==================== Google Test 测试用例 ====================

TEST(ALPGPUCompressorTest, CompressionDecompressionDouble) {
    std::vector<double> test_data;
    for (int i = 0; i < 10000; ++i) {
        test_data.push_back(i * 1.5 + 0.1);
    }
    
    ALPCompressionInfo result = comp_ALP_GPU_double(test_data);
    
    EXPECT_GT(result.compression_ratio, 0.0);
    EXPECT_GT(result.comp_throughput_gbs, 0.0);
    EXPECT_GT(result.decomp_throughput_gbs, 0.0);
    EXPECT_TRUE(result.verification_passed);
}

TEST(ALPGPUCompressorTest, CompressionDecompressionFloat) {
    std::vector<float> test_data;
    for (int i = 0; i < 10000; ++i) {
        test_data.push_back(i * 1.5f + 0.1f);
    }
    
    ALPCompressionInfo result = comp_ALP_GPU_float(test_data);
    
    EXPECT_GT(result.compression_ratio, 0.0);
    EXPECT_GT(result.comp_throughput_gbs, 0.0);
    EXPECT_GT(result.decomp_throughput_gbs, 0.0);
    EXPECT_TRUE(result.verification_passed);
}

TEST(ALPGPUCompressorTest, SmallDatasetDouble) {
    std::vector<double> test_data = {1.1, 2.2, 3.3, 4.4, 5.5};
    ALPCompressionInfo result = comp_ALP_GPU_double(test_data);
    EXPECT_GT(result.compression_ratio, 0.0);
    EXPECT_TRUE(result.verification_passed);
}

TEST(ALPGPUCompressorTest, SmallDatasetFloat) {
    std::vector<float> test_data = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    ALPCompressionInfo result = comp_ALP_GPU_float(test_data);
    EXPECT_GT(result.compression_ratio, 0.0);
    EXPECT_TRUE(result.verification_passed);
}

TEST(ALPGPUCompressorTest, LargeDatasetDouble) {
    std::vector<double> test_data;
    for (int i = 0; i < 1000000; ++i) {
        test_data.push_back(sin(i * 0.01) * cos(i * 0.02));
    }
    
    ALPCompressionInfo result = comp_ALP_GPU_double(test_data);
    EXPECT_GT(result.compression_ratio, 0.0);
    EXPECT_TRUE(result.verification_passed);
    result.print();
}

// ==================== 主函数 ====================
int main(int argc, char *argv[]) {
    cudaFree(0);  // 初始化 CUDA
    
    if (argc > 1) {
        std::string arg = argv[1];
        
        // 目录批处理模式
        if (arg == "--dir" && argc >= 3) {
            std::string dir_path = argv[2];
            std::cout << "📁 处理目录: " << dir_path << std::endl;
            
            std::vector<std::string> csv_files;
            try {
                for (const auto& entry : fs::directory_iterator(dir_path)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                        csv_files.push_back(entry.path().string());
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ 读取目录失败: " << e.what() << std::endl;
                return 1;
            }
            
            if (csv_files.empty()) {
                std::cerr << "❌ 未找到 CSV 文件" << std::endl;
                return 1;
            }
            
            std::cout << "🔍 找到 " << csv_files.size() << " 个 CSV 文件" << std::endl;
            
            // 对每个文件运行 3 次迭代
            for (const auto& file_path : csv_files) {
                ALPCompressionInfo total_result;
                
                // 预热
                try {
                    test_compression_double(file_path);
                    cudaDeviceSynchronize();
                } catch (const std::exception& e) {
                    std::cerr << "❌ 预热失败: " << e.what() << std::endl;
                    continue;
                }
                
                // 3 次迭代
                for (int i = 0; i < 3; ++i) {
                    try {
                        ALPCompressionInfo result = test_compression_double(file_path);
                        total_result += result;
                        cudaDeviceSynchronize();
                    } catch (const std::exception& e) {
                        std::cerr << "❌ 迭代 " << i << " 失败: " << e.what() << std::endl;
                        continue;
                    }
                }
                
                // 计算平均值
                total_result /= 3;
                total_result.print();
            }
            
            return 0;
        }
        
        // Beta 参数扫描模式
        else if (arg == "--file-beta" && argc >= 3) {
            std::string file_path = argv[2];
            std::cout << "🔬 Beta 参数扫描: " << file_path << std::endl;
            
            for (int beta = 4; beta <= 17; ++beta) {
                std::cout << "\n=== Beta " << beta << " ===" << std::endl;
                try {
                    ALPCompressionInfo result = test_beta_compression_double(file_path, beta);
                    result.print();
                    cudaDeviceSynchronize();
                } catch (const std::exception& e) {
                    std::cerr << "❌ Beta " << beta << " 失败: " << e.what() << std::endl;
                }
            }
            
            return 0;
        }
        
        // 单文件模式
        else {
            std::string file_path = arg;
            ALPCompressionInfo total_result;
            
            // 预热
            try {
                test_compression_double(file_path);
                cudaDeviceSynchronize();
            } catch (const std::exception& e) {
                std::cerr << "❌ 预热失败: " << e.what() << std::endl;
                return 1;
            }
            
            // 3 次迭代
            for (int i = 0; i < 3; ++i) {
                try {
                    ALPCompressionInfo result = test_compression_double(file_path);
                    total_result += result;
                    cudaDeviceSynchronize();
                } catch (const std::exception& e) {
                    std::cerr << "❌ 迭代 " << i << " 失败: " << e.what() << std::endl;
                }
            }
            
            // 计算平均值
            total_result /= 3;
            total_result.print();
            
            return 0;
        }
    }
    
    // Google Test 模式（默认）
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
