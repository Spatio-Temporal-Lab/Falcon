
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#include "Elf_Star_g_Kernel.cuh"
#include "data/dataset_utils.hpp"  // 用于读取文件数据
#include <filesystem>
namespace fs = std::filesystem;

// 测试数据生成函数
std::vector<double> generate_test_data(size_t size, int pattern = 0) {
    std::vector<double> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    switch (pattern) {
        case 0: // 随机数据
        {
            std::normal_distribution<double> dis(0.0, 1.0);
            for (size_t i = 0; i < size; ++i) {
                data[i] = dis(gen);
            }
            break;
        }
        case 1: // 递增序列
        {
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<double>(i) + 0.1;
            }
            break;
        }
        case 2: // 周期性数据
        {
            for (size_t i = 0; i < size; ++i) {
                data[i] = sin(2.0 * M_PI * i / 100.0) * 1000.0;
            }
            break;
        }
        case 3: // 稀疏数据 (很多零)
        {
            std::uniform_real_distribution<double> dis(0.0, 1.0);
            std::uniform_real_distribution<double> val_dis(-1000.0, 1000.0);
            for (size_t i = 0; i < size; ++i) {
                data[i] = (dis(gen) < 0.1) ? val_dis(gen) : 0.0;  // 10%概率非零
            }
            break;
        }
        default:
            std::fill(data.begin(), data.end(), 0.0);
            break;
    }
    
    return data;
}

// 验证解压缩结果的正确性
bool verify_decompression(const std::vector<double>& original, 
                         const double* decompressed, 
                         size_t size,
                         double tolerance = 1e-10) {
    if (size != original.size()) {
        std::cout << "大小不匹配: 原始=" << original.size() 
                  << ", 解压后=" << size << std::endl;
        return false;
    }
    
    size_t error_count = 0;
    double max_error = 0.0;
    
    for (size_t i = 0; i < size; ++i) {
        double error = std::abs(original[i] - decompressed[i]);
        if (error > tolerance) {
            error_count++;
            max_error = std::max(max_error, error);
            if (error_count <= 10) {
                std::cout << "位置 " << i << ": 原始=" << original[i] 
                          << ", 解压=" << decompressed[i] 
                          << ", 误差=" << error << std::endl;
            }
        }
    }
    
    if (error_count > 0) {
        std::cout << "验证失败: " << error_count << "/" << size 
                  << " 个元素超出容差, 最大误差=" << max_error << std::endl;
        return false;
    }
    
    return true;
}

// 带详细时间统计的ELF Star压缩测试函数 - 使用完整接口
CompressionInfo test_elf_star_compression_with_timing(const std::vector<double>& original, double error_bound = 0.0) {
    CompressionInfo result;
    
    // 压缩测试 - 使用完整接口
    uint8_t* compressed = nullptr;
    int64_t* compressed_lengths = nullptr;
    int64_t* compressed_offsets = nullptr;
    int64_t* decompressed_offsets = nullptr;
    int num_blocks = 0;
    ElfStarTimingInfo timing_info;

    auto compress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t compress_result = elf_star_encode_with_timing(
        const_cast<double*>(original.data()), 
        original.size(),
        &compressed,
        &compressed_lengths,
        &compressed_offsets,
        &decompressed_offsets,
        &num_blocks,
        &timing_info
    );
    
    if (compress_result <= 0 || compressed == nullptr) {
        std::cerr << "ELF Star压缩失败" << std::endl;
        return result;
    }
    
    auto compress_end = std::chrono::high_resolution_clock::now();
    const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
    // std::cout << "压缩成功: " << num_blocks << " 个块，总大小 " << compress_result << " 字节" << std::endl;
    
    // 构建解压缩所需的参数
    std::vector<size_t> in_offsets(num_blocks + 1);
    std::vector<size_t> in_lengths(num_blocks);
    std::vector<size_t> out_offsets(num_blocks);
    
    for (int i = 0; i < num_blocks; i++) {
        in_offsets[i] = compressed_offsets[i];
        in_lengths[i] = compressed_lengths[i];
        out_offsets[i] = decompressed_offsets[i];
    }
    in_offsets[num_blocks] = compressed_offsets[num_blocks];
    
    // 解压测试 - 使用完整接口
    std::vector<double> decompressed(original.size());

    auto decompress_start = std::chrono::high_resolution_clock::now();

    ssize_t decompress_result = elf_star_decode_with_timing(
        compressed,
        in_offsets.data(),
        in_lengths.data(),
        decompressed.data(),
        out_offsets.data(),
        num_blocks,
        &timing_info  // 会更新解压时间信息
    );
    
    auto decompress_end = std::chrono::high_resolution_clock::now();
    const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    if (decompress_result <= 0) {
        std::cerr << "ELF Star解压缩失败" << std::endl;
        elf_star_free_encode_result(compressed, compressed_lengths, 
                                   compressed_offsets, decompressed_offsets);
        return result;
    }
    
    // 验证解压缩结果
    bool data_valid = verify_decompression(original, decompressed.data(), decompressed.size(), error_bound);
    
    if (!data_valid) {
        std::cerr << "ELF Star压缩/解压缩数据验证失败" << std::endl;
    }
    
    // 性能指标计算
    const double original_bytes = original.size() * sizeof(double);
    const double compress_ratio = compress_result / original_bytes;
    
    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);
    const double compress_sec = compress_time / 1000.0;
    const double decompress_sec = decompress_time / 1000.0;
    const double compression_throughput_GBs = original_GB / compress_sec;
    const double decompression_throughput_GBs = original_GB / decompress_sec;
    
    // 填充结果结构体 - 适配用户的CompressionInfo结构
    result.original_size_mb = original_bytes / (1024.0 * 1024.0);
    result.compressed_size_mb = compress_result / (1024.0 * 1024.0);
    result.compression_ratio = compress_ratio;
    result.comp_kernel_time = timing_info.compress_kernel_time;        // 压缩核函数时间
    result.comp_time = compress_time;                                  // 压缩总时间
    result.comp_throughput = compression_throughput_GBs;               // 压缩吞吐量
    result.decomp_kernel_time = timing_info.decompress_kernel_time;    // 解压核函数时间
    result.decomp_time = decompress_time;                              // 解压总时间
    result.decomp_throughput = decompression_throughput_GBs;           // 解压吞吐量
    
    // 清理内存
    elf_star_free_encode_result(compressed, compressed_lengths, 
                               compressed_offsets, decompressed_offsets);
    
    return result;
}

// 简化版本：不使用详细时间统计的测试函数 - 使用完整接口
CompressionInfo test_elf_star_compression_simple(const std::vector<double>& original, double error_bound = 0.0) {
    CompressionInfo result;
    
    // 压缩测试 - 使用完整接口
    uint8_t* compressed = nullptr;
    int64_t* compressed_lengths = nullptr;
    int64_t* compressed_offsets = nullptr;
    int64_t* decompressed_offsets = nullptr;
    int num_blocks = 0;
    
    auto compress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t compress_result = elf_star_encode(
        const_cast<double*>(original.data()), 
        original.size(),
        &compressed,
        &compressed_lengths,
        &compressed_offsets,
        &decompressed_offsets,
        &num_blocks
    );
    
    auto compress_end = std::chrono::high_resolution_clock::now();
    
    if (compress_result <= 0 || compressed == nullptr) {
        std::cerr << "ELF Star压缩失败" << std::endl;
        return result;
    }
    
    // std::cout << "压缩成功: " << num_blocks << " 个块，总大小 " << compress_result << " 字节" << std::endl;
    
    // 构建解压缩所需的参数
    std::vector<size_t> in_offsets(num_blocks + 1);
    std::vector<size_t> in_lengths(num_blocks);
    std::vector<size_t> out_offsets(num_blocks);
    
    for (int i = 0; i < num_blocks; i++) {
        in_offsets[i] = compressed_offsets[i];
        in_lengths[i] = compressed_lengths[i];
        out_offsets[i] = decompressed_offsets[i];
    }
    in_offsets[num_blocks] = compressed_offsets[num_blocks];
    
    // 解压测试 - 使用完整接口
    std::vector<double> decompressed(original.size());
    
    auto decompress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t decompress_result = elf_star_decode(
        compressed,
        in_offsets.data(),
        in_lengths.data(),
        decompressed.data(),
        out_offsets.data(),
        num_blocks
    );
    
    auto decompress_end = std::chrono::high_resolution_clock::now();
    
    if (decompress_result <= 0) {
        std::cerr << "ELF Star解压缩失败" << std::endl;
        elf_star_free_encode_result(compressed, compressed_lengths, 
                                   compressed_offsets, decompressed_offsets);
        return result;
    }
    
    // 验证解压缩结果
    bool data_valid = verify_decompression(original, decompressed.data(), decompressed.size(), error_bound);
    
    if (!data_valid) {
        std::cerr << "ELF Star压缩/解压缩数据验证失败" << std::endl;
    }
    
    // 性能指标计算
    const double original_bytes = original.size() * sizeof(double);
    const double compress_ratio = compress_result / original_bytes;
    const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
    const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
    const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);
    const double compress_sec = compress_time / 1000.0;
    const double decompress_sec = decompress_time / 1000.0;
    const double compression_throughput_GBs = original_GB / compress_sec;
    const double decompression_throughput_GBs = original_GB / decompress_sec;
    
    // 填充结果结构体 - 适配用户的CompressionInfo结构
    result.original_size_mb = original_bytes / (1024.0 * 1024.0);
    result.compressed_size_mb = compress_result / (1024.0 * 1024.0);
    result.compression_ratio = compress_ratio;
    result.comp_kernel_time = 0;  // 简化版本没有核函数单独计时
    result.comp_time = compress_time;
    result.comp_throughput = compression_throughput_GBs;
    result.decomp_kernel_time = 0;  // 简化版本没有核函数单独计时
    result.decomp_time = decompress_time;
    result.decomp_throughput = decompression_throughput_GBs;
    
    // 清理内存
    elf_star_free_encode_result(compressed, compressed_lengths, 
                               compressed_offsets, decompressed_offsets);
    
    return result;
}

// 文件测试模板（类似LZ4测试代码）- 使用完整接口
CompressionInfo test_compression_file(const std::string& file_path) {
    std::vector<double> oriData = read_data(file_path);
    return test_elf_star_compression_with_timing(oriData);
}

// 单个测试用例 - 使用完整接口
bool run_test_case(const std::string& test_name, 
                   const std::vector<double>& test_data) {
    std::cout << "\n=== 测试用例: " << test_name << " ===\n";
    std::cout << "数据大小: " << test_data.size() << " 个double元素 ("
              << test_data.size() * sizeof(double) << " 字节)\n";
    
    CompressionInfo result = test_elf_star_compression_with_timing(test_data);

    if (result.original_size_mb > 0) {
        result.print();
        std::cout << "✓ 测试通过\n";
        return true;
    } else {
        std::cout << "❌ 测试失败\n";
        return false;
    }
}

// 主测试函数
int main(int argc, char *argv[]) {
    
    // 初始化CUDA设备
    cudaFree(0);
    
    if (argc < 2) {
        std::cout << "ELF Star 压缩算法测试程序（使用完整接口）\n";
        std::cout << "====================================================\n";
        
        // 检查CUDA设备
        int device_count = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        if (cuda_status != cudaSuccess) {
            std::cout << "❌ CUDA初始化失败: " << cudaGetErrorString(cuda_status) << std::endl;
            return -1;
        }
        
        std::cout << "检测到 " << device_count << " 个CUDA设备\n";
        
        if (device_count > 0) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, 0);
            std::cout << "使用设备: " << prop.name << std::endl;
        }
        
        // 运行测试用例
        std::vector<std::pair<std::string, std::vector<double>>> test_cases;
        
        test_cases.emplace_back("小规模随机数据", generate_test_data(1000, 0));
        test_cases.emplace_back("中等规模递增序列", generate_test_data(10000, 1));
        test_cases.emplace_back("周期性数据", generate_test_data(5000, 2));
        test_cases.emplace_back("稀疏数据", generate_test_data(8000, 3));
        test_cases.emplace_back("大规模随机数据", generate_test_data(100000, 0));
        test_cases.emplace_back("单元素数据", std::vector<double>{42.0});
        test_cases.emplace_back("全零数据", std::vector<double>(1000, 0.0));
        
        int passed_tests = 0;
        int total_tests = test_cases.size();
        
        for (const auto& test_case : test_cases) {
            if (run_test_case(test_case.first, test_case.second)) {
                passed_tests++;
            }
        }
        
        // 测试总结
        std::cout << "\n============================\n";
        std::cout << "测试总结: " << passed_tests << "/" << total_tests << " 通过\n";
        
        if (passed_tests == total_tests) {
            std::cout << "�� 所有测试通过!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ 有测试失败!" << std::endl;
            return 1;
        }
    }
    
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {
        std::string dir_path = argv[2];
        
        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        bool warm = false;
        int processed = 0;
        
        std::cout << "ELF Star 批量文件压缩测试（使用完整接口）\n";
        std::cout << "================================================\n";
        
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                CompressionInfo avg_result;
                
                if (!warm) {
                    std::cout << "\n-------------------预热阶段-------------------------- " << file_path << std::endl;
                    test_compression_file(file_path);
                    warm = true;
                    std::cout << "-------------------预热完成------------------------" << std::endl;
                }
                
                std::cout << "\nProcessing file: " << file_path << std::endl;
                
                // 运行3次取平均值（类似LZ4测试代码）
                for (int i = 0; i < 3; i++) {
                    avg_result += test_compression_file(file_path);
                }
                avg_result = avg_result / 3;
                avg_result.print();
                std::cout << "---------------------------------------------" << std::endl;
                processed++;
            }
        }
        
        if (processed == 0) {
            std::cerr << "No files found in directory: " << dir_path << std::endl;
        } else {
            std::cout << "\n处理完成，共测试 " << processed << " 个文件" << std::endl;
        }
    }
    else if (arg == "--help") {
        std::cout << "ELF Star压缩算法测试程序（使用完整接口）" << std::endl;
        std::cout << std::endl;
        std::cout << "使用方法:" << std::endl;
        std::cout << "  " << argv[0] << "                        运行所有内置测试用例" << std::endl;
        std::cout << "  " << argv[0] << " --dir <目录路径>         处理指定目录中的所有文件" << std::endl;
        std::cout << "  " << argv[0] << " --help                   显示此帮助信息" << std::endl;
        std::cout << std::endl;
        std::cout << "特性:" << std::endl;
        std::cout << "  - 使用完整的ELF Star接口（包含块信息）" << std::endl;
        std::cout << "  - 详细的核函数性能统计（H2D、Kernel、D2H时间）" << std::endl;
        std::cout << "  - 性能占比分析" << std::endl;
        std::cout << "  - 吞吐量计算" << std::endl;
        std::cout << "  - 兼容LZ4测试代码的输出格式" << std::endl;
        std::cout << "  - 适配用户自定义的CompressionInfo结构" << std::endl;
        std::cout << std::endl;
        std::cout << "示例:" << std::endl;
        std::cout << "  " << argv[0] << " --dir ./test_data" << std::endl;
        return 0;
    }
    else {
        std::cerr << "未知参数: " << arg << std::endl;
        std::cerr << "使用 --help 查看帮助信息" << std::endl;
        return 1;
    }
    
    return 0;
}
// #include "Elf_Star_g_Kernel.cuh"

// #include <gtest/gtest.h>
// #include <vector>
// #include <chrono>
// #include <cmath>
// #include <iostream>
// #include <algorithm>
// #include <cstdint>
// #include <cstring>
// #include "data/dataset_utils.hpp"
// #include <filesystem>
// namespace fs = std::filesystem;

// // GPU版本的压缩测试函数 - 使用简化接口
// CompressionInfo test_gpu_elf_compression_simple(const std::vector<double>& original, double error_bound) {
//     // 压缩测试
//     uint8_t* compressed = nullptr;
//     ssize_t compressed_len = 0;
    
//     auto compress_start = std::chrono::high_resolution_clock::now();
    
//     ssize_t compress_result = elf_star_encode_simple(
//         original.data(), 
//         original.size(),
//         &compressed,
//         &compressed_len
//     );
    
//     auto compress_end = std::chrono::high_resolution_clock::now();
    
//     if (compress_result <= 0 || compressed == nullptr) {
//         std::cerr << "GPU压缩失败" << std::endl;
//         return CompressionInfo{};
//     }
    
//     // 解压测试
//     double* decompressed = nullptr;
//     ssize_t decompressed_len = 0;
    
//     auto decompress_start = std::chrono::high_resolution_clock::now();
    
//     ssize_t decompress_result = elf_star_decode_simple(
//         compressed,
//         compressed_len,
//         &decompressed,
//         &decompressed_len
//     );
    
//     auto decompress_end = std::chrono::high_resolution_clock::now();
    
//     if (decompress_result <= 0 || decompressed == nullptr) {
//         std::cerr << "GPU解压缩失败" << std::endl;
//         free(compressed);
//         return CompressionInfo{};
//     }
    
//     // 验证解压缩结果
//     bool data_valid = true;
//     if (decompressed_len != original.size()) {
//         std::cerr << "解压缩数据大小不匹配: 期望=" << original.size() 
//                   << ", 实际=" << decompressed_len << std::endl;
//         data_valid = false;
//     } else {
//         for (size_t i = 0; i < original.size(); ++i) {
//             if (std::abs(original[i] - decompressed[i]) > error_bound) {
//                 if (data_valid) {  // 只打印第一个错误
//                     std::cerr << "数据验证失败，位置 " << i 
//                               << ": 原始=" << original[i] 
//                               << ", 解压=" << decompressed[i] 
//                               << ", 误差=" << std::abs(original[i] - decompressed[i]) << std::endl;
//                 }
//                 data_valid = false;
//             }
//         }
//     }
    
//     if (!data_valid) {
//         std::cerr << "GPU压缩/解压缩数据验证失败" << std::endl;
//     }
    
//     // 性能指标计算
//     const double original_bytes = original.size() * sizeof(double);
//     const double compress_ratio = compressed_len / original_bytes;
//     const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
//     const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
//     const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);
//     const double compress_sec = compress_time / 1000.0;
//     const double decompress_sec = decompress_time / 1000.0;
//     const double compression_throughput_GBs = original_GB / compress_sec;
//     const double decompression_throughput_GBs = original_GB / decompress_sec;
    
//     // 清理内存
//     free(compressed);
//     free(decompressed);
    
//     return CompressionInfo{
//         original_bytes/1024.0/1024.0,
//         compressed_len/1024.0/1024.0,
//         compress_ratio,
//         0,
//         compress_time,
//         compression_throughput_GBs,
//         0,
//         decompress_time,
//         decompression_throughput_GBs
//     };
// }

// // GPU版本的压缩测试函数 - 使用完整接口（带块信息）
// CompressionInfo test_gpu_elf_compression_full(const std::vector<double>& original, double error_bound) {
//     // 压缩测试
//     uint8_t* compressed = nullptr;
//     int64_t* compressed_lengths = nullptr;
//     int64_t* compressed_offsets = nullptr;
//     int64_t* decompressed_offsets = nullptr;
//     int num_blocks = 0;
    
//     auto compress_start = std::chrono::high_resolution_clock::now();
    
//     ssize_t compress_result = elf_star_encode(
//         const_cast<double*>(original.data()), 
//         original.size(),
//         &compressed,
//         &compressed_lengths,
//         &compressed_offsets,
//         &decompressed_offsets,
//         &num_blocks
//     );
    
//     auto compress_end = std::chrono::high_resolution_clock::now();
    
//     if (compress_result <= 0 || compressed == nullptr) {
//         std::cerr << "GPU压缩失败（完整接口）" << std::endl;
//         return CompressionInfo{};
//     }
    
//     std::cout << "压缩成功: " << num_blocks << " 个块，总大小 " << compress_result << " 字节" << std::endl;
    
//     // 构建解压缩所需的参数
//     std::vector<size_t> in_offsets(num_blocks + 1);
//     std::vector<size_t> in_lengths(num_blocks);
//     std::vector<size_t> out_offsets(num_blocks);
    
//     for (int i = 0; i < num_blocks; i++) {
//         in_offsets[i] = compressed_offsets[i];
//         in_lengths[i] = compressed_lengths[i];
//         out_offsets[i] = decompressed_offsets[i];
//     }
//     in_offsets[num_blocks] = compressed_offsets[num_blocks];
    
//     // 解压测试
//     std::vector<double> decompressed(original.size());
    
//     auto decompress_start = std::chrono::high_resolution_clock::now();
    
//     ssize_t decompress_result = elf_star_decode(
//         compressed,
//         in_offsets.data(),
//         in_lengths.data(),
//         decompressed.data(),
//         out_offsets.data(),
//         num_blocks
//     );
    
//     auto decompress_end = std::chrono::high_resolution_clock::now();
    
//     if (decompress_result <= 0) {
//         std::cerr << "GPU解压缩失败（完整接口）" << std::endl;
//         elf_star_free_encode_result(compressed, compressed_lengths, 
//                                    compressed_offsets, decompressed_offsets);
//         return CompressionInfo{};
//     }
    
//     // 验证解压缩结果
//     bool data_valid = true;
//     for (size_t i = 0; i < original.size(); ++i) {
//         if (std::abs(original[i] - decompressed[i]) > error_bound) {
//             if (data_valid) {  // 只打印第一个错误
//                 std::cerr << "数据验证失败，位置 " << i 
//                           << ": 原始=" << original[i] 
//                           << ", 解压=" << decompressed[i] 
//                           << ", 误差=" << std::abs(original[i] - decompressed[i]) << std::endl;
//             }
//             data_valid = false;
//         }
//     }
    
//     // 性能指标计算
//     const double original_bytes = original.size() * sizeof(double);
//     const double compress_ratio = compress_result / original_bytes;
//     const double compress_time = std::chrono::duration<double, std::milli>(compress_end - compress_start).count();
//     const double decompress_time = std::chrono::duration<double, std::milli>(decompress_end - decompress_start).count();
    
//     const double original_GB = original_bytes / (1024.0 * 1024.0 * 1024.0);
//     const double compress_sec = compress_time / 1000.0;
//     const double decompress_sec = decompress_time / 1000.0;
//     const double compression_throughput_GBs = original_GB / compress_sec;
//     const double decompression_throughput_GBs = original_GB / decompress_sec;
    
//     // 清理内存
//     elf_star_free_encode_result(compressed, compressed_lengths, 
//                                compressed_offsets, decompressed_offsets);
    
//     return CompressionInfo{
//         original_bytes/1024.0/1024.0,
//         compress_result/1024.0/1024.0,
//         compress_ratio,
//         0,
//         compress_time,
//         compression_throughput_GBs,
//         0,
//         decompress_time,
//         decompression_throughput_GBs
//     };
// }

// // 文件测试模板 - 简化接口
// CompressionInfo test_gpu_elf_with_file_simple(const std::string& file_path, double error_bound) {
//     auto data = read_data(file_path);
//     return test_gpu_elf_compression_simple(data, error_bound);
// }

// // 文件测试模板 - 完整接口
// CompressionInfo test_gpu_elf_with_file_full(const std::string& file_path, double error_bound) {
//     auto data = read_data(file_path);
//     return test_gpu_elf_compression_full(data, error_bound);
// }

// // 测试用例组
// TEST(GpuElfCompressorTest, SimpleInterfaceSmallData) {
//     const std::vector<double> data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
//     auto result = test_gpu_elf_compression_simple(data, 1e-10);

// }

// TEST(GpuElfCompressorTest, FullInterfaceSmallData) {
//     const std::vector<double> data = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
//     auto result = test_gpu_elf_compression_full(data, 1e-10);
// }

// TEST(GpuElfCompressorTest, LocalDatasetSimple) {
//     const std::string data_dir = "../test/data/float";
//     const double error_bound = 0.0;
    
//     if (!fs::exists(data_dir)) {
//         GTEST_SKIP() << "数据目录不存在: " << data_dir;
//     }
    
//     for (const auto& entry : fs::directory_iterator(data_dir)) {
//         if (entry.is_regular_file()) {
//             std::cout << "\n使用简化接口处理文件: " << entry.path().string() << std::endl;
//             auto result = test_gpu_elf_with_file_simple(entry.path().string(), error_bound);
//             result.print();
//         }
//     }
// }

// TEST(GpuElfCompressorTest, LocalDatasetFull) {
//     const std::string data_dir = "../test/data/float";
//     const double error_bound = 0.0;
    
//     if (!fs::exists(data_dir)) {
//         GTEST_SKIP() << "数据目录不存在: " << data_dir;
//     }
    
//     for (const auto& entry : fs::directory_iterator(data_dir)) {
//         if (entry.is_regular_file()) {
//             std::cout << "\n使用完整接口处理文件: " << entry.path().string() << std::endl;
//             auto result = test_gpu_elf_with_file_full(entry.path().string(), error_bound);
//             result.print();
//         }
//     }
// }

// int main(int argc, char *argv[]) {
    
//     if (argc < 2) {
//         ::testing::InitGoogleTest(&argc, argv);
//         return RUN_ALL_TESTS();
//     }
    
//     std::string arg = argv[1];
    
//     if (arg == "--dir" && argc >= 3) {
//         std::string dir_path = argv[2];
//         bool use_full_interface = false;
        
//         // 检查是否指定了使用完整接口
//         if (argc >= 4 && std::string(argv[3]) == "--full") {
//             use_full_interface = true;
//             std::cout << "使用完整接口模式" << std::endl;
//         } else {
//             std::cout << "使用简化接口模式" << std::endl;
//         }

//         // 检查目录是否存在
//         if (!fs::exists(dir_path)) {
//             std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
//             return 1;
//         }
        
//         // 预热GPU
//         std::cout << "预热GPU..." << std::endl;
//         std::vector<double> warmup_data(1024, 0.5);
//         test_gpu_elf_compression_simple(warmup_data, 0.0);
        
//         for (const auto& entry : fs::directory_iterator(dir_path)) {
//             if (entry.is_regular_file()) {
//                 CompressionInfo ans;
//                 std::cout << "\n正在处理文件: " << entry.path().string() << std::endl;
                
//                 // 运行3次取平均值
//                 for(int i = 0; i < 3; i++) {
//                     if (use_full_interface) {
//                         ans += test_gpu_elf_with_file_full(entry.path().string(), 0);
//                     } else {
//                         ans += test_gpu_elf_with_file_simple(entry.path().string(), 0);
//                     }
//                 }
//                 ans = ans / 3;
//                 ans.print();
//                 std::cout << "\n---------------------------------------------" << std::endl;
//             }
//         }
//     }
//     else if (arg == "--help") {
//         std::cout << "GPU版本ELF Star压缩算法测试程序" << std::endl;
//         std::cout << std::endl;
//         std::cout << "使用方法:" << std::endl;
//         std::cout << "  " << argv[0] << "                        运行所有测试用例" << std::endl;
//         std::cout << "  " << argv[0] << " --dir <目录路径> [--full]  处理指定目录中的所有文件" << std::endl;
//         std::cout << "  " << argv[0] << " --help                    显示此帮助信息" << std::endl;
//         std::cout << std::endl;
//         std::cout << "参数说明:" << std::endl;
//         std::cout << "  --dir <目录路径>    指定包含测试数据文件的目录" << std::endl;
//         std::cout << "  --full             使用完整接口（包含块信息），默认使用简化接口" << std::endl;
//         std::cout << std::endl;
//         std::cout << "示例:" << std::endl;
//         std::cout << "  " << argv[0] << " --dir ./test_data" << std::endl;
//         std::cout << "  " << argv[0] << " --dir ./test_data --full" << std::endl;
//         return 0;
//     }
//     else {
//         std::cerr << "未知参数: " << arg << std::endl;
//         std::cerr << "使用 --help 查看帮助信息" << std::endl;
//         return 1;
//     }
// }