#include "baseline/patas/patas.hpp"
#include "data/dataset_utils.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>
#include <filesystem>

// Patas compression/decompression for one dataset
CompressionInfo run_patas_once(const std::vector<double>& oriData) {
    CompressionInfo info;
    size_t data_count = oriData.size();
    
    // 分配缓冲区
    uint8_t*  data_arr        = new uint8_t[data_count * 16]; // 足够的空间
    uint16_t* packed_metadata = new uint16_t[data_count];
    uint64_t* dec_arr         = new uint64_t[data_count];
    alp_bench::patas::PatasUnpackedValueStats* unpacked_data = 
        new alp_bench::patas::PatasUnpackedValueStats[data_count];
    
    // 初始化编码器
    alp_bench::patas::PatasCompressionState<uint64_t, false> patas_state;
    patas_state.Reset();
    patas_state.SetOutputBuffer(data_arr);
    patas_state.packed_data_buffer.SetBuffer(packed_metadata);
    
    // 过滤NaN值
    std::vector<double> filtered_data;
    filtered_data.reserve(data_count);
    for (size_t i = 0; i < data_count; i++) {
        if (!std::isnan(oriData[i])) {
            filtered_data.push_back(oriData[i]);
        }
    }
    
    if (filtered_data.empty()) {
        std::cerr << "警告: 所有数据都是NaN" << std::endl;
        delete[] data_arr;
        delete[] packed_metadata;
        delete[] dec_arr;
        delete[] unpacked_data;
        return info;
    }
    
    size_t valid_count = filtered_data.size();
    uint64_t* uint64_p = reinterpret_cast<uint64_t*>(filtered_data.data());
    
    // ===== 压缩阶段 =====
    auto comp_start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < valid_count; i++) {
        alp_bench::patas::PatasCompression<uint64_t, false>::Store(uint64_p[i], patas_state);
    }
    
    size_t compressed_size = patas_state.byte_writer.BytesWritten();
    size_t metadata_size = valid_count * sizeof(uint16_t);
    
    auto comp_end = std::chrono::high_resolution_clock::now();
    
    // ===== 解压缩阶段 =====
    alp_bench::ByteReader byte_reader;
    byte_reader.SetStream(data_arr);
    
    auto decomp_start = std::chrono::high_resolution_clock::now();
    
    // 解包元数据
    for (size_t i = 0; i < valid_count; i++) {
        alp_bench::PackedDataUtils<uint64_t>::Unpack(packed_metadata[i],
                                                     (alp_bench::UnpackedData&)unpacked_data[i]);
    }
    
    // 解压缩数据
    dec_arr[0] = 0; 
    for (size_t i = 0; i < valid_count; i++) {
        dec_arr[i] = alp_bench::patas::PatasDecompression<uint64_t>::DecompressValue(
            byte_reader,
            unpacked_data[i].significant_bytes,
            unpacked_data[i].trailing_zeros,
            dec_arr[i - unpacked_data[i].index_diff]);
    }
    
    auto decomp_end = std::chrono::high_resolution_clock::now();
    
    // ===== 验证正确性 =====
    double* dec_dbl_p = reinterpret_cast<double*>(dec_arr);
    size_t error_count = 0;
    for (size_t i = 0; i < valid_count && i < 10; i++) {
        if (filtered_data[i] != dec_dbl_p[i]) {
            error_count++;
            if (error_count == 1) {
                std::cerr << "解压缩错误: 索引 " << i 
                          << " 原始=" << filtered_data[i] 
                          << " 解压=" << dec_dbl_p[i] << std::endl;
            }
        }
    }
    
    // 计算统计信息
    auto comp_time_ms = std::chrono::duration<double, std::milli>(comp_end - comp_start).count();
    auto decomp_time_ms = std::chrono::duration<double, std::milli>(decomp_end - decomp_start).count();
    
    size_t original_size = valid_count * sizeof(double);
    size_t total_compressed_size = compressed_size + metadata_size;
    
    // 转换为 MB
    double original_size_mb = original_size / (1024.0 * 1024.0);
    double compressed_size_mb = total_compressed_size / (1024.0 * 1024.0);
    
    info.original_size_mb = original_size_mb;
    info.compressed_size_mb = compressed_size_mb;
    info.compression_ratio = (double)original_size / (double)total_compressed_size;
    info.comp_kernel_time = 0.0;  // CPU版本无kernel时间
    info.comp_time = comp_time_ms;
    info.comp_throughput = original_size_mb / 1024.0 / (comp_time_ms / 1000.0);  // GB/s
    info.decomp_kernel_time = 0.0;  // CPU版本无kernel时间
    info.decomp_time = decomp_time_ms;
    info.decomp_throughput = original_size_mb / 1024.0 / (decomp_time_ms / 1000.0);  // GB/s
    
    // 清理
    delete[] data_arr;
    delete[] packed_metadata;
    delete[] dec_arr;
    delete[] unpacked_data;
    
    return info;
}

int main(int argc, char* argv[]) {
    std::string file_path;
    std::string dir_path;
    int iterations = 3;
    
    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--file" && i + 1 < argc) {
            file_path = argv[++i];
        } else if (arg == "--dir" && i + 1 < argc) {
            dir_path = argv[++i];
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        }
    }
    
    std::vector<std::string> files;
    
    if (!dir_path.empty()) {
        // 处理目录中的所有CSV文件
        for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
            if (entry.path().extension() == ".csv") {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
    } else if (!file_path.empty()) {
        files.push_back(file_path);
    } else {
        std::cerr << "用法: " << argv[0] << " --file <path> 或 --dir <path> [--iterations N]" << std::endl;
        return 1;
    }
    
    std::cout << "========== Patas压缩测试 ==========" << std::endl;
    std::cout << "测试迭代次数: " << iterations << std::endl;
    std::cout << "文件数量: " << files.size() << std::endl;
    std::cout << std::endl;
    
    for (const auto& fp : files) {
        std::cout << "文件: " << fp << std::endl;
        
        // 读取数据
        std::vector<double> data = read_data(fp);
        if (data.empty()) {
            std::cerr << "无法读取文件或文件为空" << std::endl;
            continue;
        }
        
        std::cout << "数据量: " << data.size() << std::endl;
        
        // 多次迭代测试
        CompressionInfo total_info;
        for (int iter = 0; iter < iterations; iter++) {
            CompressionInfo info = run_patas_once(data);
            total_info += info;
        }
        
        // 计算平均值
        CompressionInfo avg_info = total_info / iterations;
        
        // 打印结果
        avg_info.print();
        std::cout << std::endl;
    }
    
    return 0;
}
