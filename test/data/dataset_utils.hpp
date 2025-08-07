#ifndef DATASET_UTILS_HPP
#define DATASET_UTILS_HPP

#include <vector>
#include <string>
#include <filesystem>
#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cctype>
struct Column {
    uint64_t id;
    std::string name;
    std::string csv_file_path;
    std::string binary_file_path;
    uint8_t factor = 14;
    uint16_t exponent = 10;
    uint16_t exceptions_count = 5;
    uint8_t bit_width = 16;
    bool suitable_for_cutting = false;
};

std::vector<Column> get_dynamic_dataset(const std::string& directory_path, bool show_progress = false, char delimiter = ',');

// 读取浮点数数据文件
std::vector<double> read_data(const std::string& file_path, bool a=0, char delimiter = ','); 

struct CompressionInfo {
    // 数据大小信息 (MB)
    double original_size_mb = 0.0;        // 原始数据大小 (MB)
    double compressed_size_mb = 0.0;      // 压缩后数据大小 (MB)
    
    // 压缩相关信息
    double compression_ratio = 0.0;       // 压缩率
    double comp_kernel_time = 0.0;       // 压缩核函数时间 (ms)
    double comp_time = 0.0;               // 压缩总时间 (ms)
    double comp_throughput = 0.0;        // 压缩吞吐量 (GB/s)
    
    double decomp_kernel_time = 0.0;      // 解压核函数时间 (ms)
    double decomp_time = 0.0;             // 解压总时间 (ms)
    double decomp_throughput = 0.0;       // 解压吞吐量 (GB/s)
    
    // 加法运算符重载
    CompressionInfo operator+(const CompressionInfo& other) const {
        return {
            original_size_mb + other.original_size_mb,
            compressed_size_mb + other.compressed_size_mb,
            compression_ratio + other.compression_ratio,
            comp_kernel_time + other.comp_kernel_time,
            comp_time + other.comp_time,
            comp_throughput + other.comp_throughput,
            decomp_kernel_time + other.decomp_kernel_time,
            decomp_time + other.decomp_time,
            decomp_throughput + other.decomp_throughput
        };
    }
    
    // 除法运算符重载 (除以整数)
    CompressionInfo operator/(int divisor) const {
        if (divisor == 0) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        return {
            original_size_mb / divisor,
            compressed_size_mb / divisor,
            compression_ratio / divisor,
            comp_kernel_time / divisor,
            comp_time / divisor,
            comp_throughput / divisor,
            decomp_kernel_time / divisor,
            decomp_time / divisor,
            decomp_throughput / divisor
        };
    }
    
    // 除法运算符重载 (除以浮点数)
    CompressionInfo operator/(double divisor) const {
        if (divisor == 0.0) {
            throw std::invalid_argument("Division by zero is not allowed");
        }
        return {
            original_size_mb / divisor,
            compressed_size_mb / divisor,
            compression_ratio / divisor,
            comp_kernel_time / divisor,
            comp_time / divisor,
            comp_throughput / divisor,
            decomp_kernel_time / divisor,
            decomp_time / divisor,
            decomp_throughput / divisor
        };
    }
    
    // 加法赋值运算符重载
    CompressionInfo& operator+=(const CompressionInfo& other) {
        original_size_mb += other.original_size_mb;
        compressed_size_mb += other.compressed_size_mb;
        compression_ratio += other.compression_ratio;
        comp_kernel_time += other.comp_kernel_time;
        comp_time += other.comp_time;
        comp_throughput += other.comp_throughput;
        decomp_kernel_time += other.decomp_kernel_time;
        decomp_time += other.decomp_time;
        decomp_throughput += other.decomp_throughput;
        return *this;
    }
    
    // 打印函数（可选，用于调试）
    void print() const {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "压缩信息:\n";
        std::cout << "  原始数据大小: " << original_size_mb << " MB\n";
        std::cout << "  压缩后数据大小: " << compressed_size_mb << " MB\n";
        std::cout << "  压缩率: " << compression_ratio << "\n";
        std::cout << "  压缩核函数时间: " << comp_kernel_time << " ms\n";
        std::cout << "  压缩总时间: " << comp_time << " ms\n";
        std::cout << "  压缩吞吐量: " << comp_throughput << " GB/s\n";
        std::cout << "  解压核函数时间: " << decomp_kernel_time << " ms\n";
        std::cout << "  解压总时间: " << decomp_time << " ms\n";
        std::cout << "  解压吞吐量: " << decomp_throughput << " GB/s\n";
    }
};
#endif // DATASET_UTILS_HPP
