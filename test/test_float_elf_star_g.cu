
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#include "Elf_Star_g_Kernel_32.cuh"
#include "data/dataset_utils.hpp"  // 用于读取文件数据
#include <filesystem>
namespace fs = std::filesystem;

// 测试数据生成函数
std::vector<float> generate_test_data(size_t size, int pattern = 0) {
    std::vector<float> data(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    switch (pattern) {
        case 0: // 随机数据
        {
            std::normal_distribution<float> dis(0.0, 1.0);
            for (size_t i = 0; i < size; ++i) {
                data[i] = dis(gen);
            }
            break;
        }
        case 1: // 递增序列
        {
            for (size_t i = 0; i < size; ++i) {
                data[i] = static_cast<float>(i) + 0.1;
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
            std::uniform_real_distribution<float> dis(0.0, 1.0);
            std::uniform_real_distribution<float> val_dis(-1000.0, 1000.0);
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
bool verify_decompression(const std::vector<float>& original, 
                         const float* decompressed, 
                         size_t size,
                         float tolerance = 1e-10) {
    if (size != original.size()) {
        std::cout << "大小不匹配: 原始=" << original.size() 
                  << ", 解压后=" << size << std::endl;
        return false;
    }
    
    size_t error_count = 0;
    float max_error = 0.0;
    
    for (size_t i = 0; i < size; ++i) {
        float error = std::abs(original[i] - decompressed[i]);
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
CompressionInfo test_elf_star_compression_with_timing_32(const std::vector<float>& original, float error_bound = 1e-7) {
    CompressionInfo result;
    
    // 压缩测试 - 使用完整接口
    uint8_t* compressed = nullptr;
    int64_t* compressed_lengths = nullptr;
    int64_t* compressed_offsets = nullptr;
    int64_t* decompressed_offsets = nullptr;
    int num_blocks = 0;
    ElfStarTimingInfo_32 timing_info;

    auto compress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t compress_result = elf_star_encode_with_timing_32(
        const_cast<float*>(original.data()), 
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
    std::vector<float> decompressed(original.size());

    auto decompress_start = std::chrono::high_resolution_clock::now();

    ssize_t decompress_result = elf_star_decode_with_timing_32(
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
        elf_star_free_encode_result_32(compressed, compressed_lengths, 
                                   compressed_offsets, decompressed_offsets);
        return result;
    }
    
    // 验证解压缩结果
    bool data_valid = verify_decompression(original, decompressed.data(), decompressed.size(), error_bound);
    if (!data_valid) {
        result.original_size_mb=-1;
        std::cerr << "ELF Star压缩/解压缩数据验证失败" << std::endl;
        return result;
    }
    else
    {
        std::cerr << "ELF Star压缩/解压缩数据验证成功!!!!!!!!" << std::endl;
    }
    
    // 性能指标计算
    const double original_bytes = original.size() * sizeof(float);
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
    elf_star_free_encode_result_32(compressed, compressed_lengths, 
                               compressed_offsets, decompressed_offsets);
    
    return result;
}

// 简化版本：不使用详细时间统计的测试函数 - 使用完整接口
CompressionInfo test_elf_star_compression_simple(const std::vector<float>& original, float error_bound = 0.0) {
    CompressionInfo result;
    
    // 压缩测试 - 使用完整接口
    uint8_t* compressed = nullptr;
    int64_t* compressed_lengths = nullptr;
    int64_t* compressed_offsets = nullptr;
    int64_t* decompressed_offsets = nullptr;
    int num_blocks = 0;
    
    auto compress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t compress_result = elf_star_encode_32(
        const_cast<float*>(original.data()), 
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
    std::vector<float> decompressed(original.size());
    
    auto decompress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t decompress_result = elf_star_decode_32(
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
        elf_star_free_encode_result_32(compressed, compressed_lengths, 
                                   compressed_offsets, decompressed_offsets);
        return result;
    }
    
    // 验证解压缩结果
    bool data_valid = verify_decompression(original, decompressed.data(), decompressed.size(), error_bound);
    
    if (!data_valid) {
        std::cerr << "ELF Star压缩/解压缩数据验证失败" << std::endl;
    }
    
    // 性能指标计算
    const double original_bytes = original.size() * sizeof(float);
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
    elf_star_free_encode_result_32(compressed, compressed_lengths, 
                               compressed_offsets, decompressed_offsets);
    
    return result;
}

// 文件测试模板（类似LZ4测试代码）- 使用完整接口
CompressionInfo test_compression_file_32(const std::string& file_path) {
    std::vector<float> oriData = read_data_float(file_path);
    return test_elf_star_compression_with_timing_32(oriData);
}

// 单个测试用例 - 使用完整接口
bool run_test_case(const std::string& test_name, 
                   const std::vector<float>& test_data) {
    std::cout << "\n=== 测试用例: " << test_name << " ===\n";
    std::cout << "数据大小: " << test_data.size() << " 个float元素 ("
              << test_data.size() * sizeof(float) << " 字节)\n";
    
    CompressionInfo result = test_elf_star_compression_with_timing_32(test_data);

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
        std::vector<std::pair<std::string, std::vector<float>>> test_cases;
        
        test_cases.emplace_back("小规模随机数据", generate_test_data(1000, 0));
        test_cases.emplace_back("中等规模递增序列", generate_test_data(10000, 1));
        test_cases.emplace_back("周期性数据", generate_test_data(5000, 2));
        test_cases.emplace_back("稀疏数据", generate_test_data(8000, 3));
        test_cases.emplace_back("大规模随机数据", generate_test_data(100000, 0));
        test_cases.emplace_back("单元素数据", std::vector<float>{42.0});
        test_cases.emplace_back("全零数据", std::vector<float>(1000, 0.0));
        
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
                    test_compression_file_32(file_path);
                    warm = true;
                    std::cout << "-------------------预热完成------------------------" << std::endl;
                }
                
                std::cout << "\nProcessing file: " << file_path << std::endl;
                
                // 运行3次取平均值（类似LZ4测试代码）
                for (int i = 0; i < 3; i++) {
                    avg_result += test_compression_file_32(file_path);
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