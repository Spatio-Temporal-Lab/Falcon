//
// test_elf_star.cu - ELF Star 压缩算法测试程序
// Created by lizhzz on 25-7-20.
//

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>

#include "Elf_Star_g_Kernel.cuh"

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
            // 默认填充为0
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
            if (error_count <= 10) {  // 只显示前10个错误
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

// 单个测试用例
bool run_test_case(const std::string& test_name, 
                   const std::vector<double>& test_data) {
    std::cout << "\n=== 测试用例: " << test_name << " ===\n";
    std::cout << "数据大小: " << test_data.size() << " 个double元素 ("
              << test_data.size() * sizeof(double) << " 字节)\n";
    
    // 1. 压缩测试
    uint8_t* compressed_data = nullptr;
    ssize_t compressed_len = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ssize_t compress_result = elf_star_encode_simple(
        test_data.data(), test_data.size(), 
        &compressed_data, &compressed_len);
    
    auto compress_time = std::chrono::high_resolution_clock::now();
    
    if (compress_result <= 0 || compressed_data == nullptr) {
        std::cout << "❌ 压缩失败!" << std::endl;
        return false;
    }
    
    std::cout << "✓ 压缩成功: " << compress_result << " 字节\n";
    
    // 计算压缩比
    double compression_ratio = static_cast<double>(test_data.size() * sizeof(double)) / compress_result;
    std::cout << "压缩比: " << std::fixed << std::setprecision(2) 
              << compression_ratio << ":1 (" 
              << (1.0 - static_cast<double>(compress_result) / (test_data.size() * sizeof(double))) * 100.0 
              << "% 空间节省)\n";
    
    // 2. 解压缩测试
    double* decompressed_data = nullptr;
    ssize_t decompressed_len = 0;
    
    auto decompress_start = std::chrono::high_resolution_clock::now();
    
    ssize_t decompress_result = elf_star_decode_simple(
        compressed_data, compressed_len,
        &decompressed_data, &decompressed_len);
    
    auto decompress_time = std::chrono::high_resolution_clock::now();
    
    if (decompress_result <= 0 || decompressed_data == nullptr) {
        std::cout << "❌ 解压缩失败!" << std::endl;
        if (compressed_data) free(compressed_data);
        return false;
    }
    
    std::cout << "✓ 解压缩成功: " << decompress_result << " 个元素\n";
    
    // 3. 验证结果正确性
    bool verification_passed = verify_decompression(
        test_data, decompressed_data, decompressed_len);
    
    if (verification_passed) {
        std::cout << "✓ 数据验证通过\n";
    } else {
        std::cout << "❌ 数据验证失败\n";
    }
    
    // 4. 性能统计
    auto compress_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        compress_time - start_time).count();
    auto decompress_duration = std::chrono::duration_cast<std::chrono::microseconds>(
        decompress_time - decompress_start).count();
    
    std::cout << "压缩耗时: " << compress_duration << " μs\n";
    std::cout << "解压缩耗时: " << decompress_duration << " μs\n";
    
    // 清理内存
    if (compressed_data) free(compressed_data);
    if (decompressed_data) free(decompressed_data);
    
    return verification_passed;
}

// 主测试函数
int main() {
    std::cout << "ELF Star 压缩算法测试程序\n";
    std::cout << "============================\n";
    
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
    
    // 测试用例1: 小规模随机数据
    test_cases.emplace_back("小规模随机数据", generate_test_data(1000, 0));
    
    // 测试用例2: 中等规模递增序列
    test_cases.emplace_back("中等规模递增序列", generate_test_data(10000, 1));
    
    // 测试用例3: 周期性数据
    test_cases.emplace_back("周期性数据", generate_test_data(5000, 2));
    
    // 测试用例4: 稀疏数据
    test_cases.emplace_back("稀疏数据", generate_test_data(8000, 3));
    
    // 测试用例5: 大规模数据
    test_cases.emplace_back("大规模随机数据", generate_test_data(100000, 0));
    
    // 测试用例6: 边界情况 - 单元素
    test_cases.emplace_back("单元素数据", std::vector<double>{42.0});
    
    // 测试用例7: 边界情况 - 全零数据
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
        std::cout << "🎉 所有测试通过!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ 有测试失败!" << std::endl;
        return 1;
    }
}