#include "elf_star.h"

#include <gtest/gtest.h>
#include <vector>
#include <chrono>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstdint> // 确保 uint8_t 等类型定义
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

// 数据生成模板
std::vector<double> generate_elf_test_data(size_t size, int pattern_type) {
    std::vector<double> data(size);
    const double amplitude = 20.0f;
    
    switch(pattern_type) {
        case 0: // 随机波动数据
            for(size_t i=0; i<size; ++i) {
                data[i] = amplitude * (static_cast<double>(rand())/RAND_MAX - 0.5f);
            }
            break;
            
        case 1: // 线性递增数据
            for(size_t i=0; i<size; ++i) {
                data[i] = 0.0001f + i*0.000001f;
            }
            break;
            
        case 2: // 周期信号+噪声
            for(size_t i=0; i<size; ++i) {
                double base = amplitude * std::sin(i * 0.1f);
                data[i] = base + 0.1f*(static_cast<double>(rand())/RAND_MAX - 0.5f);
            }
            break;
    }
    return data;
}

// 原始压缩测试函数（保持不变）
CompressionInfo test_elf_compression(const std::vector<double>& original, double error_bound) {
    // 压缩测试
    uint8_t* compressed = nullptr;
    auto compress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t compressed_size = elf_star_encode(
        const_cast<double*>(original.data()), 
        original.size(),
        &compressed
    );
    
    auto compress_end = std::chrono::high_resolution_clock::now();
    
    // 解压测试
    std::vector<double> decompressed(original.size());
    auto decompress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t dec_size = elf_star_decode(
        compressed,
        compressed_size,
        decompressed.data()
    );
    
    auto decompress_end = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < original.size()  ; ++i) {
        if(original[i] - decompressed[i] != 0) {
          GTEST_LOG_(INFO) << " " << original[i] << " " << decompressed[i];
        }
      }
    // 性能指标计算
    const double original_bytes = original.size() * sizeof(double);
    const double compress_ratio = compressed_size/original_bytes;
    const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
    const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);

    // 压缩时间转换为秒（1 秒 = 1000 毫秒）
    const double compress_sec = compress_time / 1000.0;
    const double decompress_sec = decompress_time / 1000.0;
    // 压缩吞吐量（GB/s）
    const double compression_throughput_GBs = original_GB / compress_sec;
    const double decompression_throughput_GBs = original_GB / decompress_sec;
    
    free(compressed); // 释放压缩数据内存
    return CompressionInfo{
        original_bytes/1024.0/1024.0,
        compressed_size/1024.0/1024.0,
        compress_ratio,
        0,
        compress_time,
        compression_throughput_GBs,0,decompress_time,decompression_throughput_GBs};
}

// 新增：分块压缩测试函数
CompressionInfo test_elf_compression_with_blocks(const std::vector<double>& original, double error_bound, size_t blockSize) {
    if (blockSize == 0) {
        std::cerr << "错误：blockSize不能为0" << std::endl;
        return CompressionInfo{};
    }
    
    const size_t totalSize = original.size();
    const size_t numBlocks = (totalSize + blockSize - 1) / blockSize; // 向上取整
    
    std::cout << "分块信息: 总数据量=" << totalSize << ", 块大小=" << blockSize 
              << ", 块数量=" << numBlocks << std::endl;
    
    // 累计统计变量
    double total_original_bytes = 0;
    double total_compressed_bytes = 0;
    double total_compress_time = 0;
    double total_decompress_time = 0;
    size_t successful_blocks = 0;
    
    // 存储解压缩结果用于验证
    std::vector<double> all_decompressed;
    all_decompressed.reserve(totalSize);
    
    for (size_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
        const size_t startIdx = blockIdx * blockSize;
        const size_t endIdx = std::min(startIdx + blockSize, totalSize);
        const size_t currentBlockSize = endIdx - startIdx;
        
        // std::cout << "处理块 " << (blockIdx + 1) << "/" << numBlocks 
        //           << " (大小: " << currentBlockSize << ")" << std::endl;
        
        // 创建当前块的数据
        std::vector<double> blockData(original.begin() + startIdx, original.begin() + endIdx);
        
        // 压缩当前块
        uint8_t* compressed = nullptr;
        auto compress_start = std::chrono::high_resolution_clock::now();
        
        ssize_t compressed_size = elf_star_encode(
            blockData.data(), 
            currentBlockSize,
            &compressed
        );
        
        auto compress_end = std::chrono::high_resolution_clock::now();
        
        if (compressed_size <= 0) {
            std::cerr << "块 " << (blockIdx + 1) << " 压缩失败" << std::endl;
            if (compressed) free(compressed);
            continue;
        }
        
        // 解压当前块
        std::vector<double> blockDecompressed(currentBlockSize);
        auto decompress_start = std::chrono::high_resolution_clock::now();
        
        ssize_t dec_size = elf_star_decode(
            compressed,
            compressed_size,
            blockDecompressed.data()
        );
        
        auto decompress_end = std::chrono::high_resolution_clock::now();
        
        if (dec_size != currentBlockSize) {
            std::cerr << "块 " << (blockIdx + 1) << " 解压失败，期望大小: " 
                      << currentBlockSize << ", 实际大小: " << dec_size << std::endl;
            free(compressed);
            continue;
        }
        
        // 验证解压缩结果
        bool block_valid = true;
        for (size_t i = 0; i < currentBlockSize; ++i) {
            if (std::abs(blockData[i] - blockDecompressed[i]) > error_bound) {
                std::cerr << "块 " << (blockIdx + 1) << " 数据验证失败，位置 " << i 
                          << ": 原始=" << blockData[i] << ", 解压=" << blockDecompressed[i] << std::endl;
                block_valid = false;
                break;
            }
        }
        
        if (!block_valid) {
            free(compressed);
            continue;
        }
        
        // 累计统计
        const double block_original_bytes = currentBlockSize * sizeof(double);
        const double block_compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
        const double block_decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
        
        total_original_bytes += block_original_bytes;
        total_compressed_bytes += compressed_size;
        total_compress_time += block_compress_time;
        total_decompress_time += block_decompress_time;
        successful_blocks++;
        
        // 保存解压缩结果
        all_decompressed.insert(all_decompressed.end(), blockDecompressed.begin(), blockDecompressed.end());
        
        // 输出当前块的统计信息
        const double block_ratio = static_cast<double>(compressed_size) / block_original_bytes;
        // std::cout << "  块压缩率: " << block_ratio << ", 压缩时间: " << block_compress_time 
        //           << "ms, 解压时间: " << block_decompress_time << "ms" << std::endl;
        
        free(compressed);
    }
    
    // 计算总体性能指标
    if (successful_blocks == 0) {
        std::cerr << "所有块处理失败" << std::endl;
        return CompressionInfo{};
    }
    
    const double total_compress_ratio = total_compressed_bytes / total_original_bytes;
    const double total_original_GB = total_original_bytes / (1024.0 * 1024.0 * 1024.0);
    const double total_compress_sec = total_compress_time / 1000.0;
    const double total_decompress_sec = total_decompress_time / 1000.0;
    const double compression_throughput_GBs = total_original_GB / total_compress_sec;
    const double decompression_throughput_GBs = total_original_GB / total_decompress_sec;
    
    // std::cout << "\n分块压缩总结:" << std::endl;
    // std::cout << "成功处理块数: " << successful_blocks << "/" << numBlocks << std::endl;
    // std::cout << "总压缩率: " << total_compress_ratio << std::endl;
    // std::cout << "总压缩时间: " << total_compress_time << "ms" << std::endl;
    // std::cout << "总解压时间: " << total_decompress_time << "ms" << std::endl;
    // std::cout << "压缩吞吐量: " << compression_throughput_GBs << " GB/s" << std::endl;
    // std::cout << "解压吞吐量: " << decompression_throughput_GBs << " GB/s" << std::endl;
    
    return CompressionInfo{
        total_original_bytes/1024.0/1024.0,
        total_compressed_bytes/1024.0/1024.0,
        total_compress_ratio,
        0,
        total_compress_time,
        compression_throughput_GBs,
        0,
        total_decompress_time,
        decompression_throughput_GBs
    };
}

// 测试用例组
TEST(ElfCompressorTest, SmallDataset) {
    const std::vector<double> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    test_elf_compression(data, 0.01f);
}

TEST(ElfCompressorTest, IncrementalData) {
    auto data = generate_elf_test_data(1024*256, 1); // 生成262K数据点
    test_elf_compression(data, 0.005f);
}

TEST(ElfCompressorTest, PeriodicWithNoise) {
    auto data = generate_elf_test_data(1024*1024, 2); // 生成1M数据点
    test_elf_compression(data, 0.02f);
}

// 新增：分块测试用例
TEST(ElfCompressorTest, SmallDatasetWithBlocks) {
    const std::vector<double> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    test_elf_compression_with_blocks(data, 0.01f, 2); // 块大小为2
}

TEST(ElfCompressorTest, IncrementalDataWithBlocks) {
    auto data = generate_elf_test_data(1024*256, 1); // 生成262K数据点
    test_elf_compression_with_blocks(data, 0.005f, 1024); // 块大小为1024
}

TEST(ElfCompressorTest, PeriodicWithNoiseBlocks) {
    auto data = generate_elf_test_data(1024*1024, 2); // 生成1M数据点
    test_elf_compression_with_blocks(data, 0.02f, 4096); // 块大小为4096
}

// 添加本地数据测试模板
CompressionInfo test_elf_with_file(const std::string& file_path, double error_bound) {
    auto data = read_data(file_path);
    return test_elf_compression(data, error_bound);
}

// 新增：带分块的文件测试模板
CompressionInfo test_elf_with_file_blocks(const std::string& file_path, double error_bound, size_t blockSize) {
    auto data = read_data(file_path);
    return test_elf_compression_with_blocks(data, error_bound, blockSize);
}

// 新增测试用例组
TEST(ElfCompressorTest, LocalDataset) {
    // const std::string data_dir = "../test/data/big"; // 修改为实际路径
    const std::string data_dir = "../test/data/float"; // 修改为实际路径

    const double error_bound = 0.000; // 根据数据特性调整
    
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.is_regular_file()) {
            std::cout << "\n正在处理文件:  " << entry.path().string() << std::endl;
            test_elf_with_file(entry.path().string(), error_bound);
        }
    }
}

int main(int argc, char *argv[]) {
    
    if (argc < 2) {
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {
        std::string dir_path = argv[2];
        size_t blockSize = 1024*1024; // 默认不分块
        
        // 检查是否指定了块大小
        if (argc >= 5 && std::string(argv[3]) == "--block-size") {
            blockSize = std::stoull(argv[4]);
            std::cout << "使用分块模式，块大小: " << blockSize << std::endl;
        }

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                CompressionInfo ans;
                std::cout << "\n正在处理文件: " << entry.path().string() << std::endl;
                
                for(int i = 0; i < 3; i++) {
                    if (blockSize > 0) {
                        ans += test_elf_with_file_blocks(entry.path().string(), 0, blockSize);
                    } else {
                        ans += test_elf_with_file(entry.path().string(), 0);
                    }
                }
                ans = ans / 3;
                ans.print();
                std::cout << "\n---------------------------------------------" << std::endl;
            }
        }
    }
    else if (arg == "--help") {
        std::cout << "使用方法:" << std::endl;
        std::cout << "  " << argv[0] << " --dir <目录路径> [--block-size <块大小>]" << std::endl;
        std::cout << "  " << argv[0] << " --help" << std::endl;
        std::cout << std::endl;
        std::cout << "参数说明:" << std::endl;
        std::cout << "  --dir <目录路径>        指定包含测试数据文件的目录" << std::endl;
        std::cout << "  --block-size <块大小>   可选，指定分块大小（元素个数）" << std::endl;
        std::cout << "                          如果不指定，则使用整体压缩模式" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  " << argv[0] << " --dir ./test_data" << std::endl;
        std::cout << "  " << argv[0] << " --dir ./test_data --block-size 1024" << std::endl;
        return 0;
    }
    else {
        std::cerr << "未知参数: " << arg << std::endl;
        std::cerr << "使用 --help 查看帮助信息" << std::endl;
        return 1;
    }
    
    return 0;
}

// #include "elf_star.h"

// #include <gtest/gtest.h>
// #include <vector>
// #include <chrono>
// #include <cmath>
// #include <iostream>
// #include <algorithm>
// #include <cstdint> // 确保 uint8_t 等类型定义
// #include "data/dataset_utils.hpp"
// #include <filesystem>
// namespace fs = std::filesystem;
// // 数据生成模板
// std::vector<double> generate_elf_test_data(size_t size, int pattern_type) {
//     std::vector<double> data(size);
//     const double amplitude = 20.0f;
    
//     switch(pattern_type) {
//         case 0: // 随机波动数据
//             for(size_t i=0; i<size; ++i) {
//                 data[i] = amplitude * (static_cast<double>(rand())/RAND_MAX - 0.5f);
//             }
//             break;
            
//         case 1: // 线性递增数据
//             for(size_t i=0; i<size; ++i) {
//                 data[i] = 0.0001f + i*0.000001f;
//             }
//             break;
            
//         case 2: // 周期信号+噪声
//             for(size_t i=0; i<size; ++i) {
//                 double base = amplitude * std::sin(i * 0.1f);
//                 data[i] = base + 0.1f*(static_cast<double>(rand())/RAND_MAX - 0.5f);
//             }
//             break;
//     }
//     return data;
// }

// // 核心测试模板
// CompressionInfo test_elf_compression(const std::vector<double>& original) {
//     // 压缩测试
//     uint8_t* compressed = nullptr;
//     auto compress_start = std::chrono::high_resolution_clock::now();
    
//     ssize_t compressed_size = elf_star_encode(
//         const_cast<double*>(original.data()), 
//         original.size(),
//         &compressed
//     );
    
//     auto compress_end = std::chrono::high_resolution_clock::now();
    
//     // 解压测试
//     std::vector<double> decompressed(original.size());
//     auto decompress_start = std::chrono::high_resolution_clock::now();
    
//     ssize_t dec_size = elf_star_decode(
//         compressed,
//         compressed_size,
//         decompressed.data()
//     );
    
//     auto decompress_end = std::chrono::high_resolution_clock::now();
//     // for (int i = 0; i < original.size()  ; ++i) {
//     //     if(original[i] - decompressed[i] != 0) {
//     //       GTEST_LOG_(INFO) << " " << original[i] << " " << decompressed[i];
//     //     }
//     //   }
//     // 性能指标计算
//     const double original_bytes = original.size() * sizeof(double);
//     const double compress_ratio = compressed_size/original_bytes;
//     const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
//     const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
//     const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);

//     // 压缩时间转换为秒（1 秒 = 1000 毫秒）
//     const double compress_sec = compress_time / 1000.0;
//     const double decompress_sec = decompress_time / 1000.0;
//     // 压缩吞吐量（GB/s）
//     const double compression_throughput_GBs = original_GB / compress_sec;
//     const double decompression_throughput_GBs = original_GB / decompress_sec;
    
//     free(compressed); // 释放压缩数据内存
//     return CompressionInfo{
//         original_bytes/1024.0/1024.0,
//         compressed_size/1024.0/1024.0,
//         compress_ratio,
//         0,
//         compress_time,
//         compression_throughput_GBs,0,decompress_time,decompression_throughput_GBs};
// }

// // 测试用例组
// TEST(ElfCompressorTest, SmallDataset) {
//     const std::vector<double> data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
//     test_elf_compression(data);
// }

// TEST(ElfCompressorTest, IncrementalData) {
//     auto data = generate_elf_test_data(1024*256, 1); // 生成262K数据点
//     test_elf_compression(data);
// }

// TEST(ElfCompressorTest, PeriodicWithNoise) {
//     auto data = generate_elf_test_data(1024*1024, 2); // 生成1M数据点
//     test_elf_compression(data);
// }

// // 添加本地数据测试模板
// CompressionInfo test_elf_with_file(const std::string& file_path, double error_bound) {
//     auto data = read_data(file_path);
//     return test_elf_compression(data);
// }

// // 新增测试用例组
// TEST(ElfCompressorTest, LocalDataset) {
//     // const std::string data_dir = "../test/data/big"; // 修改为实际路径
//     const std::string data_dir = "../test/data/new_tsbs"; // 修改为实际路径

//     const double error_bound = 0.000; // 根据数据特性调整
    
//     for (const auto& entry : fs::directory_iterator(data_dir)) {
//         if (entry.is_regular_file()) {
//             std::cout << "\n正在处理文件:  " << entry.path().string() << std::endl;
//             test_elf_with_file(entry.path().string(), error_bound);
//         }
//     }
// }
// int main(int argc, char *argv[]) {
    
//     std::string arg = argv[1];
    
//     if (arg == "--dir" && argc >= 3) {

//         std::string dir_path = argv[2];

//         // 检查目录是否存在
//         if (!fs::exists(dir_path)) {
//             std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
//             return 1;
//         }
        
//         // const double error_bound = 0.000; // 根据数据特性调整
        
//         for (const auto& entry : fs::directory_iterator(dir_path)) {
//             if (entry.is_regular_file()) {
//                 CompressionInfo ans;
//                 std::cout << "\n正在处理文件: " << entry.path().string() << std::endl;
//                 for(int i=0;i<3;i++)
//                 {
//                     ans+=test_elf_with_file(entry.path().string(), 0);
//                 }
//                 ans=ans/3;
//                 ans.print();
//                 std::cout << "\n---------------------------------------------" << std::endl;
//             }
//         }
//     }
//     else{
//         ::testing::InitGoogleTest(&argc, argv);
//         return RUN_ALL_TESTS();
//     }
// }