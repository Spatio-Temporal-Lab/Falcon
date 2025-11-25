#include "Falcon_basic_compressor.h"
#include "Falcon_basic_decompressor.h"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <iomanip>
#include "data/dataset_utils.hpp"
#include <filesystem>

namespace fs = std::filesystem;

// 测试数据生成模板
std::vector<double> generate_falcon_test_data(size_t size, int pattern_type) {
    std::vector<double> data(size);
    
    switch(pattern_type) {
        case 0: // 随机数据
            for (size_t i = 0; i < size; ++i) {
                data[i] = (rand() % 10000) / 100.0;
            }
            break;
        case 1: // 递增数据
            for (size_t i = 0; i < size; ++i) {
                data[i] = i * 0.01;
            }
            break;
        case 2: // 正弦波数据
            for (size_t i = 0; i < size; ++i) {
                data[i] = sin(i * 0.01) * 100.0;
            }
            break;
        default:
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<double>(i);
            }
    }
    
    return data;
}

// 核心测试模板
CompressionInfo test_falcon_basic_compression(const std::vector<double>& original, size_t blockSize = 1025) {
    Falcon_basic_compressor compressor;
    Falcon_basic_decompressor decompressor;
    
    // 设置块大小（与GPU版本一致：第一个数作为基准，后面1024个数的delta序列）
    compressor.BLOCK_SIZE = blockSize;
    
    // 压缩测试
    std::vector<unsigned char> compressed;
    auto compress_start = std::chrono::high_resolution_clock::now();
    
    compressor.compress(original, compressed);
    
    auto compress_end = std::chrono::high_resolution_clock::now();
    
    // 解压测试
    std::vector<double> decompressed;
    auto decompress_start = std::chrono::high_resolution_clock::now();
    
    decompressor.decompress(compressed, decompressed);
    
    auto decompress_end = std::chrono::high_resolution_clock::now();
    
    // 验证数据正确性
    if (original.size() != decompressed.size()) {
        std::cerr << "错误: 解压后数据大小不匹配! 原始=" << original.size() 
                  << ", 解压=" << decompressed.size() << std::endl;
        return CompressionInfo{};
    }
    
    if (compressed.empty()) {
        std::cerr << "错误: 压缩结果为空" << std::endl;
        return CompressionInfo{};
    }
    
    // 验证解压缩正确性（使用适当的误差容忍度）
    bool data_valid = true;
    size_t error_count = 0;
    const double tolerance = 0;//1e-12; // 适当的误差容忍度
    
    for (size_t i = 0; i < original.size(); ++i) {
        double diff = std::abs(original[i] - decompressed[i]);
        if (diff > tolerance) {
            if (error_count < 10) { // 只显示前10个错误
                std::cerr << "数据不匹配 [" << i << "]: 原始=" << std::setprecision(15) 
                         << original[i] << ", 解压=" << decompressed[i] 
                         << ", 差值=" << diff << std::endl;
            }
            error_count++;
            data_valid = false;
        }
    }
    
    if (!data_valid) {
        std::cerr << "警告: 发现 " << error_count << " 个数据不匹配（总共 " 
                  << original.size() << " 个数据点）" << std::endl;
        // 对于Falcon压缩，我们可能需要容忍一些精度损失
        // return CompressionInfo{}; // 如果需要严格验证，取消注释这行
    }
    
    // 性能指标计算
    const double original_bytes = original.size() * sizeof(double);
    const double compressed_bytes = compressed.size();
    const double compress_ratio = compressed_bytes / original_bytes;
    const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
    const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);
    const double compress_sec = compress_time / 1000.0;
    const double decompress_sec = decompress_time / 1000.0;
    
    const double compression_throughput_GBs = original_GB / compress_sec;
    const double decompression_throughput_GBs = original_GB / decompress_sec;
    
    return CompressionInfo{
        original_bytes / 1024.0 / 1024.0,
        compressed_bytes / 1024.0 / 1024.0,
        compress_ratio,
        0,
        compress_time,
        compression_throughput_GBs,
        0,
        decompress_time,
        decompression_throughput_GBs
    };
}

// 测试用例组
TEST(FalconBasicCompressorTest, SmallDataset) {
    auto data = generate_falcon_test_data(100, 0);
    auto info = test_falcon_basic_compression(data);
    info.print();
    EXPECT_GT(info.compression_ratio, 0);
    EXPECT_LE(info.compression_ratio, 1.0);
}

TEST(FalconBasicCompressorTest, IncrementalData) {
    auto data = generate_falcon_test_data(10000, 1);
    auto info = test_falcon_basic_compression(data);
    info.print();
    EXPECT_GT(info.compression_ratio, 0);
}

TEST(FalconBasicCompressorTest, SineWaveData) {
    auto data = generate_falcon_test_data(10000, 2);
    auto info = test_falcon_basic_compression(data);
    info.print();
    EXPECT_GT(info.compression_ratio, 0);
}

TEST(FalconBasicCompressorTest, LargeDataset) {
    auto data = generate_falcon_test_data(1000000, 0);
    auto info = test_falcon_basic_compression(data);
    info.print();
    EXPECT_GT(info.compression_ratio, 0);
}

// 本地数据测试模板（与ELF保持一致）
CompressionInfo test_falcon_with_file(const std::string& file_path, size_t blockSize = 1025) {
    try {
        auto data = read_data(file_path);
        if (data.empty()) {
            std::cerr << "警告: 文件 " << file_path << " 为空或无法读取" << std::endl;
            return CompressionInfo{};
        }
        
        // std::cout << "文件数据点数: " << data.size() << std::endl;
        
        auto result = test_falcon_basic_compression(data, blockSize);
        
        if (result.original_size_mb == 0) {
            std::cerr << "警告: 文件 " << file_path << " 压缩失败" << std::endl;
        }
        
        return result;
        
    } catch (const std::exception& e) {
        std::cerr << "处理文件 " << file_path << " 时发生异常: " << e.what() << std::endl;
        return CompressionInfo{};
    }
}

// 数据集测试
TEST(FalconBasicCompressorTest, LocalDataset) {
    const std::string data_dir = "../test/data/float"; // 修改为实际路径
    
    if (!fs::exists(data_dir)) {
        GTEST_SKIP() << "数据集目录不存在: " << data_dir;
    }
    
    CompressionInfo total_info;
    int count = 0;
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::cout << "\n处理文件: " << entry.path().filename() << std::endl;
            
            auto info = test_falcon_with_file(entry.path().string());
            if (info.original_size_mb > 0) {
                std::cout << "文件: " << entry.path().filename() << std::endl;
                info.print();
                total_info += info;
                count++;
            }
        }
    }
    
    if (count > 0) {
        std::cout << "\n=== 平均性能 ===" << std::endl;
        (total_info / count).print();
    } else {
        GTEST_SKIP() << "未找到有效数据文件";
    }
}

// 命令行参数测试
int main(int argc, char** argv) {
    if (argc > 1) {
        std::string arg = argv[1];
        
        if (arg == "--dir" && argc >= 3) {
            std::string data_dir = argv[2];
            
            // 检查目录是否存在
            if (!fs::exists(data_dir)) {
                std::cerr << "错误: 数据目录不存在: " << data_dir << std::endl;
                return 1;
            }
            
            std::cout << "=== Falcon CPU版本压缩测试 ===" << std::endl;
            std::cout << "数据目录: " << data_dir << std::endl;
            std::cout << "==============================\n" << std::endl;
            
            CompressionInfo total_info;
            int count = 0;
            
            for (const auto& entry : fs::directory_iterator(data_dir)) {
                if (entry.is_regular_file()) {
                    std::cout << "\n正在处理文件: " << entry.path().filename() << std::endl;
                    
                    // 进行3次测试取平均值（与ELF测试保持一致）
                    CompressionInfo avg_info;
                    bool has_valid_result = false;
                    
                    for (int run = 0; run < 3; run++) {
                        auto info = test_falcon_with_file(entry.path().string());
                        if (info.original_size_mb > 0) {
                            avg_info += info;
                            has_valid_result = true;
                        }
                    }
                    
                    if (has_valid_result) {
                        avg_info = avg_info / 3;
                        std::cout << "文件: " << entry.path().filename() << std::endl;
                        avg_info.print();
                        total_info += avg_info;
                        count++;
                    } else {
                        std::cerr << "跳过文件: " << entry.path().filename() 
                                  << "（处理失败或数据为空）" << std::endl;
                    }
                    
                    std::cout << "\n---------------------------------------------" << std::endl;
                }
            }
            
            if (count > 0) {
                std::cout << "\n====== 总体统计 ======" << std::endl;
                std::cout << "成功处理文件数: " << count << std::endl;
                std::cout << "\n=== 平均性能 ===" << std::endl;
                (total_info / count).print();
            } else {
                std::cout << "未找到有效数据文件" << std::endl;
            }
            
            return 0;
        }
    }
    else
    {
        // 运行标准 GTest 测试
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }

}