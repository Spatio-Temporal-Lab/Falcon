#include "alp.hpp"
#include "alp_gpu.hpp"
#include "gtest/gtest.h"
#include "data/dataset_utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cstring>

#include <set>
namespace fs = std::filesystem;

// ALP GPU 压缩解压测试（基本版本）
std::vector<std::pair<uint8_t, uint8_t>> gpu_vector_records = alp_gpu::get_gpu_vector_ef_records_const();
std::vector<std::vector<std::pair<std::pair<int, int>, int>>> gpu_rowgroup_records = alp_gpu::get_gpu_rowgroup_k_combinations_const();

std::vector<std::pair<uint8_t, uint8_t>> &cpu_vector_records = alp::get_vector_ef_records();
std::vector<std::vector<std::pair<std::pair<int, int>, int>>> &cpu_rowgroup_records = alp::get_rowgroup_k_combinations();

void compare_alp_records(
    const std::vector<std::pair<uint8_t, uint8_t>>& gpu_vector_records,
    const std::vector<std::vector<std::pair<std::pair<int, int>, int>>>& gpu_rowgroup_records,
    const std::vector<std::pair<uint8_t, uint8_t>>& cpu_vector_records,
    const std::vector<std::vector<std::pair<std::pair<int, int>, int>>>& cpu_rowgroup_records
) {
    std::cout << "========== ALP CPU vs GPU 记录对比分析 ==========" << std::endl;
    
    // 1. 基础统计对比
    std::cout << "\n1. 基础统计对比:" << std::endl;
    std::cout << "向量记录数量 - CPU: " << cpu_vector_records.size() 
              << ", GPU: " << gpu_vector_records.size() << std::endl;
    std::cout << "行组记录数量 - CPU: " << cpu_rowgroup_records.size() 
              << ", GPU: " << gpu_rowgroup_records.size() << std::endl;
    
    // 2. 向量(e,f)记录对比
    std::cout << "\n2. 向量(e,f)记录对比:" << std::endl;
    
    if (cpu_vector_records.size() != gpu_vector_records.size()) {
        std::cout << "⚠️  向量记录数量不匹配!" << std::endl;
    }
    
    // 统计(e,f)组合分布
    std::map<std::pair<uint8_t, uint8_t>, int> cpu_ef_count, gpu_ef_count;
    for (const auto& pair : cpu_vector_records) {
        cpu_ef_count[pair]++;
    }
    for (const auto& pair : gpu_vector_records) {
        gpu_ef_count[pair]++;
    }
    
    std::cout << "\n(e,f)组合分布对比:" << std::endl;
    std::set<std::pair<uint8_t, uint8_t>> all_ef_pairs;
    for (const auto& p : cpu_ef_count) all_ef_pairs.insert(p.first);
    for (const auto& p : gpu_ef_count) all_ef_pairs.insert(p.first);
    
    std::cout << std::left << std::setw(8) << "(e,f)" 
              << std::setw(12) << "CPU数量" 
              << std::setw(12) << "GPU数量" 
              << std::setw(12) << "差异" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    int total_diff = 0;
    for (const auto& ef : all_ef_pairs) {
        int cpu_count = cpu_ef_count[ef];
        int gpu_count = gpu_ef_count[ef];
        int diff = gpu_count - cpu_count;
        total_diff += std::abs(diff);
        
        std::cout << "(" << (int)ef.first << "," << (int)ef.second << ")" 
                  << std::setw(8) << cpu_count 
                  << std::setw(12) << gpu_count 
                  << std::setw(12) << diff;
        if (diff != 0) std::cout << " ⚠️";
        std::cout << std::endl;
    }
    std::cout << "总差异数量: " << total_diff << std::endl;
    
    // 3. 逐个向量对比（前20个）
    std::cout << "\n3. 向量记录逐项对比 (前20个):" << std::endl;
    std::cout << std::left << std::setw(8) << "索引" 
              << std::setw(12) << "CPU(e,f)" 
              << std::setw(12) << "GPU(e,f)" 
              << std::setw(10) << "匹配" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    size_t compare_count = std::min({cpu_vector_records.size(), gpu_vector_records.size()});
    int mismatch_count = 0;
    
    for (size_t i = 0; i < 20; ++i) {
        auto cpu_ef = cpu_vector_records[i];
        auto gpu_ef = gpu_vector_records[i];
        bool match = (cpu_ef.first == gpu_ef.first && cpu_ef.second == gpu_ef.second);
        if (!match) mismatch_count++;
        
        std::cout << std::setw(8) << i
                  << "(" << (int)cpu_ef.first << "," << (int)cpu_ef.second << ")"
                  << std::setw(8) << ""
                  << "(" << (int)gpu_ef.first << "," << (int)gpu_ef.second << ")"
                  << std::setw(8) << ""
                  << (match ? "✓" : "✗") << std::endl;
    }
    
    if (cpu_vector_records.size() > 20 || gpu_vector_records.size() > 20) {
        // 检查剩余部分的匹配度
        size_t min_size = std::min(cpu_vector_records.size(), gpu_vector_records.size());
        for (size_t i = 20; i < min_size; ++i) {
            auto cpu_ef = cpu_vector_records[i];
            auto gpu_ef = gpu_vector_records[i];
            bool match = (cpu_ef.first == gpu_ef.first && cpu_ef.second == gpu_ef.second);
            if (!match) 
            {
                mismatch_count++;
                if(mismatch_count<20)
                std::cout << std::setw(8) << i
                    << "(" << (int)cpu_ef.first << "," << (int)cpu_ef.second << ")"
                    << std::setw(8) << ""
                    << "(" << (int)gpu_ef.first << "," << (int)gpu_ef.second << ")"
                    << std::setw(8) << ""
                    << (match ? "✓" : "✗") << std::endl;
            }
        }
        std::cout << "... (剩余" << (min_size - 20) << "项检查完成)" << std::endl;
    }
    
    std::cout << "向量记录不匹配数量: " << mismatch_count << "/" << compare_count << std::endl;
    
    // 4. 行组k组合记录对比
    std::cout << "\n4. 行组k组合记录对比:" << std::endl;
    
    if (cpu_rowgroup_records.size() != gpu_rowgroup_records.size()) {
        std::cout << "⚠️  行组记录数量不匹配!" << std::endl;
    }
    
    size_t rowgroup_compare_count = std::min(cpu_rowgroup_records.size(), gpu_rowgroup_records.size());
    int rowgroup_mismatch_count = 0;
    
    std::cout << std::left << std::setw(8) << "行组ID" 
              << std::setw(15) << "CPU组合数" 
              << std::setw(15) << "GPU组合数" 
              << std::setw(10) << "匹配" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (size_t rg = 0; rg < std::min(rowgroup_compare_count, size_t(10)); ++rg) {
        const auto& cpu_combinations = cpu_rowgroup_records[rg];
        const auto& gpu_combinations = gpu_rowgroup_records[rg];
        
        bool rowgroup_match = true;
        
        // 比较组合数量
        if (cpu_combinations.size() != gpu_combinations.size()) {
            rowgroup_match = false;
        } else {
            // 比较每个组合（按(e,f)排序后比较）
            auto cpu_sorted = cpu_combinations;
            auto gpu_sorted = gpu_combinations;
            
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            std::sort(gpu_sorted.begin(), gpu_sorted.end());
            
            for (size_t i = 0; i < cpu_sorted.size(); ++i) {
                if (cpu_sorted[i].first != gpu_sorted[i].first) {
                    rowgroup_match = false;
                    break;
                }
            }
        }
        
        if (!rowgroup_match) rowgroup_mismatch_count++;
        
        std::cout << std::setw(8) << rg 
                  << std::setw(15) << cpu_combinations.size()
                  << std::setw(15) << gpu_combinations.size()
                  << std::setw(10) << (rowgroup_match ? "✓" : "✗") << std::endl;
        
        // 显示具体组合内容（如果不匹配或前3个行组）
        if (!rowgroup_match || rg < 3) {
            std::cout << "  CPU组合: ";
            for (const auto& comb : cpu_combinations) {
                std::cout << "(" << comb.first.first << "," << comb.first.second << ":" << comb.second << ") ";
            }
            std::cout << std::endl;
            
            std::cout << "  GPU组合: ";
            for (const auto& comb : gpu_combinations) {
                std::cout << "(" << comb.first.first << "," << comb.first.second << ":" << comb.second << ") ";
            }
            std::cout << std::endl;
        }
    }
    
    if (rowgroup_compare_count > 10) {
        // 检查剩余行组
        for (size_t rg = 10; rg < rowgroup_compare_count; ++rg) {
            const auto& cpu_combinations = cpu_rowgroup_records[rg];
            const auto& gpu_combinations = gpu_rowgroup_records[rg];
            
            if (cpu_combinations.size() != gpu_combinations.size()) {
                rowgroup_mismatch_count++;
                // continue;
                std::cout << rg <<"  CPU组合: ";
                for (const auto& comb : cpu_combinations) {
                    std::cout << "(" << comb.first.first << "," << comb.first.second << ":" << comb.second << ") ";
                }
                std::cout << std::endl;
                
                std::cout << "  GPU组合: ";
                for (const auto& comb : gpu_combinations) {
                    std::cout << "(" << comb.first.first << "," << comb.first.second << ":" << comb.second << ") ";
                }
                std::cout << std::endl;
                continue;
            }
            
            auto cpu_sorted = cpu_combinations;
            auto gpu_sorted = gpu_combinations;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            std::sort(gpu_sorted.begin(), gpu_sorted.end());
            bool match = true;
            for (size_t i = 0; i < cpu_sorted.size(); ++i) {
                if (cpu_sorted[i].first != gpu_sorted[i].first) {
                    match = false;
                    break;
                }
            }
            // if (!match) 
            if (!match && rowgroup_compare_count<20) {
                rowgroup_mismatch_count++;
                std::cout << rg <<"  CPU组合: ";
                for (const auto& comb : cpu_combinations) {
                    std::cout << "(" << comb.first.first << "," << comb.first.second << ":" << comb.second << ") ";
                }
                std::cout << std::endl;
                
                std::cout << "  GPU组合: ";
                for (const auto& comb : gpu_combinations) {
                    std::cout << "(" << comb.first.first << "," << comb.first.second << ":" << comb.second << ") ";
                }
                std::cout << std::endl;
            }
        }
        std::cout << "... (剩余" << (rowgroup_compare_count - 10) << "个行组检查完成)" << std::endl;
    }
    
    std::cout << "行组记录不匹配数量: " << rowgroup_mismatch_count << "/" << rowgroup_compare_count << std::endl;
    
    // 5. 总体一致性分析
    std::cout << "\n5. 总体一致性分析:" << std::endl;
    
    bool size_consistent = (cpu_vector_records.size() == gpu_vector_records.size() && 
                           cpu_rowgroup_records.size() == gpu_rowgroup_records.size());
    
    double vector_match_rate = compare_count > 0 ? 
        (double)(compare_count - mismatch_count) / compare_count * 100.0 : 0.0;
    
    double rowgroup_match_rate = rowgroup_compare_count > 0 ? 
        (double)(rowgroup_compare_count - rowgroup_mismatch_count) / rowgroup_compare_count * 100.0 : 0.0;
    
    std::cout << "数据量一致性: " << (size_consistent ? "✓ 一致" : "✗ 不一致") << std::endl;
    std::cout << "向量记录匹配率: " << std::fixed << std::setprecision(2) << vector_match_rate << "%" << std::endl;
    std::cout << "行组记录匹配率: " << std::fixed << std::setprecision(2) << rowgroup_match_rate << "%" << std::endl;


    if (mismatch_count == 0 && rowgroup_mismatch_count == 0) {
        std::cout << "✓ CPU和GPU实现结果完全一致!" << std::endl;
    }
    
    std::cout << "\n========== 对比分析完成 ==========" << std::endl;
}


// 辅助函数：详细分析特定不匹配项
void analyze_specific_differences(
    const std::vector<std::pair<uint8_t, uint8_t>>& gpu_vector_records,
    const std::vector<std::pair<uint8_t, uint8_t>>& cpu_vector_records,
    int start_index = 0, int count = 10
) {
    std::cout << "\n========== 详细差异分析 ==========" << std::endl;
    
    size_t min_size = std::min(cpu_vector_records.size(), gpu_vector_records.size());
    size_t end_index = std::min(min_size, size_t(start_index + count));
    
    std::vector<size_t> mismatch_indices;
    
    for (size_t i = start_index; i < end_index; ++i) {
        if (cpu_vector_records[i] != gpu_vector_records[i]) {
            mismatch_indices.push_back(i);
        }
    }
    
    std::cout << "索引 " << start_index << "-" << (end_index-1) 
              << " 中发现 " << mismatch_indices.size() << " 处不匹配:" << std::endl;
    
    for (size_t idx : mismatch_indices) {
        auto cpu_ef = cpu_vector_records[idx];
        auto gpu_ef = gpu_vector_records[idx];
        
        std::cout << "索引 " << idx << ": CPU(" << (int)cpu_ef.first << "," << (int)cpu_ef.second 
                  << ") vs GPU(" << (int)gpu_ef.first << "," << (int)gpu_ef.second << ")" << std::endl;
        
        // 分析(e,f)的差异特征
        if (cpu_ef.first != gpu_ef.first) {
            std::cout << "  指数差异: " << (int)cpu_ef.first << " -> " << (int)gpu_ef.first 
                      << " (差值: " << ((int)gpu_ef.first - (int)cpu_ef.first) << ")" << std::endl;
        }
        if (cpu_ef.second != gpu_ef.second) {
            std::cout << "  因子差异: " << (int)cpu_ef.second << " -> " << (int)gpu_ef.second 
                      << " (差值: " << ((int)gpu_ef.second - (int)cpu_ef.second) << ")" << std::endl;
        }
    }
}

CompressionInfo comp_alp_gpu_stream(std::vector<float> oriData, std::vector<float> &decompData)
{
    size_t nbEle = oriData.size();
    alp_gpu::clear_gpu_alp_records();
    // 设置ALP GPU参数
    alp_gpu::Params params;
    params.vectorSize = 1024;
    params.blockSize = 1024 * 100;
    params.threadsPerBlock = 128; // 使用优化后的线程数
    params.enable_recording = true;
    alp_gpu::enable_gpu_alp_recording(true);
    params.use_alprd_cutting = true;
    params.prefer_alprd = false;
    params.debug = false;

    // 分配设备内存
    float *d_oriData = nullptr;
    float *d_decData = nullptr;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建CUDA事件用于计时
    cudaEvent_t comp_start, comp_end;
    cudaEvent_t decomp_start, decomp_end;
    cudaEvent_t total_start, total_end;

    cudaEventCreate(&comp_start);
    cudaEventCreate(&comp_end);
    cudaEventCreate(&decomp_start);
    cudaEventCreate(&decomp_end);
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_end);

    // ===== 优化的压缩流程 =====
    cudaEventRecord(total_start, stream);

    // 1. 分配设备内存并传输原始数据
    cudaMalloc(&d_oriData, nbEle * sizeof(float));
    cudaMemcpyAsync(d_oriData, oriData.data(), nbEle * sizeof(float),
                    cudaMemcpyHostToDevice, stream);

    // 2. 直接在GPU上压缩（数据保持在GPU）
    cudaEventRecord(comp_start, stream);
    alp_gpu::CompressedDevice compressed_device =
        alp_gpu::compress_float_device(d_oriData, nbEle, params, stream);
    cudaEventRecord(comp_end, stream);

    // 3. 只将压缩后的数据传回主机（用于存储或传输）
    alp_gpu::Compressed compressed = alp_gpu::device_to_host(compressed_device);

    cudaStreamSynchronize(stream);

    // 计算压缩时间
    float comp_kernel_time = 0.0f;
    cudaEventElapsedTime(&comp_kernel_time, comp_start, comp_end);

    double compression_ratio = static_cast<double>(compressed.bytes()) / (nbEle * sizeof(float));

    // 释放原始数据的设备内存（如果不再需要）
    cudaFree(d_oriData);

    // ===== 优化的解压流程 =====

    // 1. 分配解压输出的设备内存
    cudaMalloc(&d_decData, nbEle * sizeof(float));

    // 2. 直接在GPU上解压（压缩数据已在GPU上）
    cudaEventRecord(decomp_start, stream);
    alp_gpu::decompress_float_device(compressed_device, d_decData, nbEle, params, stream);
    cudaEventRecord(decomp_end, stream);

    // 3. 将解压后的数据传回主机（用于验证）
    decompData.resize(nbEle);
    cudaMemcpyAsync(decompData.data(), d_decData, nbEle * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);

    cudaEventRecord(total_end, stream);
    cudaStreamSynchronize(stream);

    // 计算时间
    float decomp_kernel_time = 0.0f;
    float total_time = 0.0f;
    cudaEventElapsedTime(&decomp_kernel_time, decomp_start, decomp_end);
    cudaEventElapsedTime(&total_time, total_start, total_end);

    // 计算总的压缩时间（包括H2D传输）
    float h2d_time = 0.0f;
    cudaEventElapsedTime(&h2d_time, total_start, comp_start);
    float total_compress_time = h2d_time + comp_kernel_time;

    // 计算总的解压时间（包括D2H传输）
    float d2h_time = 0.0f;
    cudaEventElapsedTime(&d2h_time, decomp_end, total_end);
    float total_decompress_time = decomp_kernel_time + d2h_time;

    // 计算吞吐量
    size_t in_bytes = oriData.size() * sizeof(float);
    double data_size_gb = in_bytes / (1024.0 * 1024.0 * 1024.0);
    double compress_throughput = data_size_gb / (total_compress_time / 1000.0);
    double decompress_throughput = data_size_gb / (total_decompress_time / 1000.0);

    gpu_vector_records = alp_gpu::get_gpu_vector_ef_records_const();
    gpu_rowgroup_records = alp_gpu::get_gpu_rowgroup_k_combinations_const();

    std::cout << "gpu记录了 " << gpu_vector_records.size() << " 个向量\n";
    std::cout << "gpu记录了 " << gpu_rowgroup_records.size() << " 个行组（ALP选择）\n";
    // 清理记录
    // printf("GOPU处理了%d个行组\n",(nbEle+params.blockSize)/params.blockSize);
    // 打印详细时间分析
    // std::cout << "\n===== 时间分析 =====" << std::endl;
    // std::cout << "压缩阶段：" << std::endl;
    // std::cout << "  H2D传输: " << h2d_time << " ms" << std::endl;
    // std::cout << "  GPU压缩: " << comp_kernel_time << " ms" << std::endl;
    // std::cout << "  总压缩时间: " << total_compress_time << " ms" << std::endl;
    // std::cout << "解压阶段：" << std::endl;
    // std::cout << "  GPU解压: " << decomp_kernel_time << " ms" << std::endl;
    // std::cout << "  D2H传输: " << d2h_time << " ms" << std::endl;
    // std::cout << "  总解压时间: " << total_decompress_time << " ms" << std::endl;

    CompressionInfo info{
        in_bytes / 1024.0 / 1024.0,                                // original_size_mb
        static_cast<double>(compressed.bytes()) / 1024.0 / 1024.0, // compressed_size_mb
        compression_ratio,                                         // compression_ratio
        comp_kernel_time,                                          // comp_kernel_time
        total_compress_time,                                       // comp_time
        compress_throughput,                                       // comp_throughput
        decomp_kernel_time,                                        // decomp_kernel_time
        total_decompress_time,                                     // decomp_time
        decompress_throughput                                      // decomp_throughput
    };

    // 释放设备内存
    cudaFree(d_decData);
    // compressed_device会自动释放其d_data

    // 释放流和事件
    cudaStreamDestroy(stream);
    cudaEventDestroy(comp_start);
    cudaEventDestroy(comp_end);
    cudaEventDestroy(decomp_start);
    cudaEventDestroy(decomp_end);
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_end);

    // 验证解压结果的正确性
    float max_error = 0.0;
    for (size_t i = 0; i < oriData.size(); ++i)
    {
        float error = std::abs(oriData[i] - decompData[i]);
        max_error = std::max(max_error, error);
        if (error > 1e-6)
        {
            std::cout << "第 " << i << " 个值不相等。\n";
            printf("ori:%.10f, dec:%.10f, error:%.10e\n", oriData[i], decompData[i], error);
            return info;
        }
    }

    std::cout << "ALP GPU压缩成功，最大误差: " << max_error << std::endl;
    return info;
}

CompressionInfo test_alp_gpu_stream_compression(const std::string &file_path)
{
    std::vector<float> oriData = read_data_float(file_path);
    std::vector<float> decompressedData;
    return comp_alp_gpu_stream(oriData, decompressedData);
}

namespace test
{
    template <typename T>
    void ALP_ASSERT(T original_val, T decoded_val)
    {
        if (original_val == 0.0 && std::signbit(original_val))
        {
            ASSERT_EQ(decoded_val, 0.0);
            ASSERT_TRUE(std::signbit(decoded_val));
        }
        else if (std::isnan(original_val))
        {
            ASSERT_TRUE(std::isnan(decoded_val));
        }
        else
        {
            ASSERT_EQ(original_val, decoded_val);
        }
    }
} // namespace test

// 文件压缩统计结构
struct FileCompressionStats
{
    std::string file_path;
    double total_original_size = 0;
    double total_compressed_size = 0;
    std::chrono::duration<double> total_encode_time{0.0};
    std::chrono::duration<double> total_decode_time{0.0};
    size_t total_data_points = 0;
    int batches_processed = 0;
    std::map<alp::Scheme, int> scheme_usage;

    double get_compression_ratio() const
    {
        return total_original_size > 0 ? total_compressed_size / total_original_size : 0.0;
    }

    double get_encode_throughput_mbps() const
    {
        double time_seconds = total_encode_time.count();
        if (time_seconds > 0)
        {
            double mb_per_second = (total_original_size / (1024.0 * 1024.0)) / time_seconds;
            return mb_per_second;
        }
        return 0.0;
    }

    double get_decode_throughput_mbps() const
    {
        double time_seconds = total_decode_time.count();
        if (time_seconds > 0)
        {
            double mb_per_second = (total_original_size / (1024.0 * 1024.0)) / time_seconds;
            return mb_per_second;
        }
        return 0.0;
    }
};

std::string dir_path = "../test/data/tsbs_csv";

TEST(alp_test, ratio)
{
    std::vector<std::string> csv_files;

    try
    {
        for (const auto &entry : fs::directory_iterator(dir_path))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".csv")
            {
                csv_files.push_back(entry.path().string());
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "读取目录失败: " << e.what() << std::endl;
        FAIL() << "无法读取数据目录: " << dir_path;
    }

    std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;

    std::vector<FileCompressionStats> all_file_stats;

    for (const auto &file_path : csv_files)
    {
        std::cout << "\nProcessing file: " << file_path << std::endl;

        FileCompressionStats file_stats;
        file_stats.file_path = file_path;

        try
        {
            // alp::clear_alp_records();
            // printf("----------------- CPU ------------------\n");
            // // 只运行一次，避免重复输出干扰分析
            // test_file_data<double>(file_path, file_stats);

            // CompressionInfo compression_info{
            //     file_stats.total_original_size / (1024.0 * 1024.0),
            //     file_stats.total_compressed_size / (1024.0 * 1024.0),
            //     file_stats.get_compression_ratio(),
            //     0,
            //     std::chrono::duration<double, std::milli>(file_stats.total_encode_time).count(),
            //     file_stats.get_encode_throughput_mbps() / 1024.0,
            //     0,
            //     std::chrono::duration<double, std::milli>(file_stats.total_decode_time).count(),
            //     file_stats.get_decode_throughput_mbps() / 1024.0};

            // compression_info.print();
            // printf("----------------- GPU ------------------\n");
            CompressionInfo info = test_alp_gpu_stream_compression(file_path);
            info.print();
            
            // compare_alp_records(gpu_vector_records, gpu_rowgroup_records, 
            //                 cpu_vector_records, cpu_rowgroup_records);
            all_file_stats.push_back(file_stats);
            std::cout << "========================================" << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "处理文件 " << file_path << " 时出错: " << e.what() << std::endl;
            continue; // 继续处理下一个文件
        }
    }

    // 输出总体统计信息
    if (!all_file_stats.empty())
    {
        // std::cout << "\n"
        //           << std::string(60, '=') << std::endl;
        // std::cout << "总体统计信息" << std::endl;
        // std::cout << std::string(60, '=') << std::endl;

        // double total_original = 0, total_compressed = 0;
        // std::chrono::duration<double> total_encode_time{0.0}, total_decode_time{0.0};
        // std::map<alp::Scheme, int> total_scheme_usage;
        // size_t total_rowgroups = 0;

        // for (const auto &stats : all_file_stats)
        // {
        //     total_original += stats.total_original_size;
        //     total_compressed += stats.total_compressed_size;
        //     total_encode_time += stats.total_encode_time;
        //     total_decode_time += stats.total_decode_time;
        //     total_rowgroups += stats.batches_processed;

        //     for (const auto &[scheme, count] : stats.scheme_usage)
        //     {
        //         total_scheme_usage[scheme] += count;
        //     }
        // }

        // std::cout << "总原始大小: " << std::fixed << std::setprecision(2)
        //           << total_original / (1024.0 * 1024.0) << " MB" << std::endl;
        // std::cout << "总压缩大小: " << std::fixed << std::setprecision(2)
        //           << total_compressed / (1024.0 * 1024.0) << " MB" << std::endl;
        // std::cout << "总压缩比: " << std::fixed << std::setprecision(4)
        //           << (total_original > 0 ? total_compressed / total_original : 0.0) << std::endl;
        // std::cout << "总编码时间: " << std::fixed << std::setprecision(2)
        //           << std::chrono::duration<double, std::milli>(total_encode_time).count() << " ms" << std::endl;
        // std::cout << "总解码时间: " << std::fixed << std::setprecision(2)
        //           << std::chrono::duration<double, std::milli>(total_decode_time).count() << " ms" << std::endl;

        // std::cout << "\n总体方案使用统计:" << std::endl;
        // for (const auto &[scheme, count] : total_scheme_usage)
        // {
        //     std::string scheme_name = (scheme == alp::Scheme::ALP) ? "ALP" : "ALP_RD";
        //     std::cout << "  " << scheme_name << ": " << count << " 个rowgroups" << std::endl;
        // }

        // std::cout << "总处理的rowgroups: " << total_rowgroups << std::endl;
        // std::cout << "处理的文件数: " << all_file_stats.size() << std::endl;
    }
}

int main(int argc, char *argv[])
{
    if (argc >= 3 && std::string(argv[1]) == "--dir")
    {
        dir_path = argv[2];
    }

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}