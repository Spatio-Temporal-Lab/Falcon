#include <filesystem>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include "data/dataset_utils.hpp"

// Chimp128 baseline headers
#include "chimp_compressor.h"
#include "chimp_decompressor.h"

namespace fs = std::filesystem;

// Chimp128 使用 previousValues=128
static const int CHIMP_PREVIOUS_VALUES = 128;
static const size_t BATCH_SIZE = 1024;

static CompressionInfo test_chimp_compression(const std::vector<double>& data) {
    // 过滤 NaN 值（Chimp 使用 NaN 作为终止符）
    std::vector<double> filtered_data;
    filtered_data.reserve(data.size());
    for (const auto& v : data) {
        if (!std::isnan(v)) {
            filtered_data.push_back(v);
        }
    }
    
    if (filtered_data.empty()) {
        std::cerr << "[Chimp] 警告: 数据为空或全为 NaN" << std::endl;
        return CompressionInfo{};
    }

    const size_t total_count = filtered_data.size();
    const size_t num_batches = (total_count + BATCH_SIZE - 1) / BATCH_SIZE;
    
    long total_compressed_bits = 0;
    std::vector<Array<uint8_t>> compressed_batches;
    compressed_batches.reserve(num_batches);
    
    // ========== 压缩阶段 ==========
    auto comp_start = std::chrono::high_resolution_clock::now();
    
    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start_idx = batch * BATCH_SIZE;
        size_t end_idx = std::min(start_idx + BATCH_SIZE, total_count);
        
        ChimpCompressor compressor(CHIMP_PREVIOUS_VALUES);
        for (size_t i = start_idx; i < end_idx; ++i) {
            compressor.addValue(filtered_data[i]);
        }
        compressor.close();
        
        total_compressed_bits += compressor.get_size();
        compressed_batches.push_back(compressor.get_compress_pack());
    }
    
    auto comp_end = std::chrono::high_resolution_clock::now();
    
    // ========== 解压阶段 ==========
    auto decomp_start = std::chrono::high_resolution_clock::now();
    
    std::vector<double> all_decompressed;
    all_decompressed.reserve(total_count);
    
    for (size_t batch = 0; batch < num_batches; ++batch) {
        ChimpDecompressor decompressor(compressed_batches[batch], CHIMP_PREVIOUS_VALUES);
        std::vector<double> decompressed = decompressor.decompress();
        all_decompressed.insert(all_decompressed.end(), decompressed.begin(), decompressed.end());
    }
    
    auto decomp_end = std::chrono::high_resolution_clock::now();

    // 验证正确性
    if (all_decompressed.size() != filtered_data.size()) {
        std::cerr << "[Chimp] 警告: 解压数据大小不匹配! 原始=" << filtered_data.size()
                  << ", 解压=" << all_decompressed.size() << std::endl;
    }

    const double original_bytes = filtered_data.size() * sizeof(double);
    const double compressed_bytes = (total_compressed_bits + 7) / 8;
    const double compress_ratio = (original_bytes > 0) ? (compressed_bytes / original_bytes) : 0.0;
    
    const double comp_ms = std::chrono::duration<double, std::milli>(comp_end - comp_start).count();
    const double decomp_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();

    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);
    const double comp_sec = comp_ms / 1000.0;
    const double decomp_sec = decomp_ms / 1000.0;

    const double comp_throughput = (comp_sec > 0) ? (original_GB / comp_sec) : 0.0;
    const double decomp_throughput = (decomp_sec > 0) ? (original_GB / decomp_sec) : 0.0;

    return CompressionInfo{
        original_bytes / 1024.0 / 1024.0,
        compressed_bytes / 1024.0 / 1024.0,
        compress_ratio,
        0.0,
        comp_ms,
        comp_throughput,
        0.0,
        decomp_ms,
        decomp_throughput
    };
}

int main(int argc, char** argv) {
    std::string data_dir;
    
    if (argc > 2 && std::string(argv[1]) == "--dir") {
        data_dir = argv[2];
    } else {
        data_dir = "../test/data/float";
    }
    
    if (!fs::exists(data_dir)) {
        std::cerr << "错误: 数据目录不存在: " << data_dir << std::endl;
        return 1;
    }
    
    CompressionInfo total_info;
    int count = 0;
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_regular_file()) continue;
        
        auto data = read_data(entry.path().string());
        if (data.empty()) continue;
        
        std::cout << "\n[Chimp] 正在处理文件: " << entry.path().filename() << std::endl;
        std::cout << "[Chimp] 数据量: " << data.size() << ", 分批数: " 
                  << (data.size() + BATCH_SIZE - 1) / BATCH_SIZE << std::endl;
        
        CompressionInfo avg_info;
        bool ok = false;
        
        for (int i = 0; i < 3; ++i) {
            auto info = test_chimp_compression(data);
            if (info.original_size_mb > 0) {
                avg_info += info;
                ok = true;
            }
        }
        
        if (ok) {
            avg_info = avg_info / 3.0;
            avg_info.print();
            total_info += avg_info;
            count++;
        }
        
        std::cout << "\n---------------------------------------------" << std::endl;
    }
    
    if (count > 0) {
        std::cout << "\n[Chimp] 总体统计，平均性能:" << std::endl;
        (total_info / count).print();
    } else {
        std::cout << "未找到有效数据文件" << std::endl;
    }
    
    return 0;
}
