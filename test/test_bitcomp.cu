// #include <gtest/gtest.h>
// #include <fstream>
// #include <vector>
// #include <chrono>
// #include <iostream>
// #include <cassert>
// #include <cuda_runtime.h>
// #include <nvcomp/bitcomp.hpp>
// #include <nvcomp/bitcomp.h>
// #include <nvcomp.h>
// #include <nvcomp.hpp>
// #include "data/dataset_utils.hpp"
// #include <filesystem>

// namespace fs = std::filesystem;

// #define CUDA_CHECK(cond) { gpuAssert((cond), __FILE__, __LINE__); }

// inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
//     if (code != cudaSuccess) {
//         fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort) exit(code);
//     }
// }

// void test_bitcomp_performance(const std::string& file_path) {
//     // 1. 读取数据
//     std::vector<double> oriData = read_data(file_path);
//     const size_t in_bytes = oriData.size() * sizeof(double);
//     const size_t num_elems = oriData.size();

//     // 2. 准备GPU资源
//     cudaStream_t stream;
//     CUDA_CHECK(cudaStreamCreate(&stream));
    
//     double* d_in_data;
//     CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
//     CUDA_CHECK(cudaMemcpyAsync(d_in_data, oriData.data(), in_bytes, 
//                              cudaMemcpyHostToDevice, stream));

//     // 3. 初始化Bitcomp管理器
//     nvcomp::BitcompManager manager(NVCOMP_TYPE_UINT, 0, stream);
//     auto comp_config = manager.configure_compression(in_bytes);

//     // 4. 分配压缩缓冲区
//     uint8_t* d_comp_out;
//     CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

//     // 5. 执行压缩
//     auto compress_start = std::chrono::high_resolution_clock::now();
//     manager.compress(reinterpret_cast<const uint8_t*>(d_in_data),
//                     d_comp_out, comp_config);
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     auto compress_end = std::chrono::high_resolution_clock::now();
    
//     // 6. 获取压缩结果
//     const size_t comp_bytes = manager.get_compressed_output_size(d_comp_out);
//     const double compress_time = std::chrono::duration<double>(compress_end - compress_start).count();
//     const double compress_throughput = (in_bytes / 1e9) / compress_time;

//     // 7. 准备解压
//     auto decomp_config = manager.configure_decompression(d_comp_out);
//     double* d_out_data;
//     CUDA_CHECK(cudaMalloc(&d_out_data, decomp_config.decomp_data_size));
//     CUDA_CHECK(cudaMemsetAsync(d_out_data, 0, decomp_config.decomp_data_size, stream));

//     // 8. 执行解压
//     auto decompress_start = std::chrono::high_resolution_clock::now();
//     manager.decompress(reinterpret_cast<uint8_t*>(d_out_data),
//                       d_comp_out, decomp_config);
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     auto decompress_end = std::chrono::high_resolution_clock::now();
    
//     // 9. 计算性能指标
//     const double decompress_time = std::chrono::duration<double>(decompress_end - decompress_start).count();
//     const double decompress_throughput = (in_bytes / 1e9) / decompress_time;
//     const double compression_ratio = static_cast<double>(comp_bytes) / in_bytes;

//     // 10. 验证数据完整性
//     std::vector<double> decompressed(num_elems);
//     CUDA_CHECK(cudaMemcpy(decompressed.data(), d_out_data, in_bytes, 
//                         cudaMemcpyDeviceToHost));
//     ASSERT_EQ(oriData, decompressed);

//     // 11. 输出结果
//     std::cout << "\n[Bitcomp Performance Report]" << std::endl;
//     std::cout << "Original Size:    " << in_bytes/1e6 << " MB" << std::endl;
//     std::cout << "Compressed Size:  " << comp_bytes/1e6 << " MB" << std::endl;
//     std::cout << "Compression Time: " << compress_time << " sec" << std::endl;
//     std::cout << "Compression Rate: " << compress_throughput << " GB/s" << std::endl;
//     std::cout << "Decompression Time: " << decompress_time << " sec" << std::endl;
//     std::cout << "Decompression Rate: " << decompress_throughput << " GB/s" << std::endl;
//     std::cout << "Compression Ratio: " << compression_ratio << std::endl;

//     // 12. 清理资源
//     CUDA_CHECK(cudaFree(d_in_data));
//     CUDA_CHECK(cudaFree(d_comp_out));
//     CUDA_CHECK(cudaFree(d_out_data));
//     CUDA_CHECK(cudaStreamDestroy(stream));
// }

// /************************ 批量测试实现 ************************/
// void test_bitcomp_batched(const std::string& file_path) {
//     // 1. 数据准备
//     std::vector<double> oriData = read_data(file_path);
//     const size_t total_bytes = oriData.size() * sizeof(double);
//     const size_t chunk_size = 1 << 20; // 1MB chunks
//     const size_t batch_size = (total_bytes + chunk_size - 1) / chunk_size;

//     // 2. 准备批量数据
//     std::vector<void*> h_uncompressed_ptrs(batch_size);
//     std::vector<size_t> h_uncompressed_bytes(batch_size);
    
//     // 3. 分配GPU资源
//     cudaStream_t stream;
//     CUDA_CHECK(cudaStreamCreate(&stream));
    
//     // 批量输入指针
//     void** d_uncompressed_ptrs;
//     size_t* d_uncompressed_bytes;
//     CUDA_CHECK(cudaMalloc(&d_uncompressed_ptrs, batch_size*sizeof(void*)));
//     CUDA_CHECK(cudaMalloc(&d_uncompressed_bytes, batch_size*sizeof(size_t)));

//     // 批量输出指针
//     void** d_compressed_ptrs;
//     size_t* d_compressed_bytes;
//     CUDA_CHECK(cudaMalloc(&d_compressed_ptrs, batch_size*sizeof(void*)));
//     CUDA_CHECK(cudaMalloc(&d_compressed_bytes, batch_size*sizeof(size_t)));

//     // 4. 配置压缩参数
//     nvcompBatchedBitcompFormatOpts opts{nvcompBatchedBitcompDefaultOpts};
//     opts.data_type = NVCOMP_TYPE_UINT;

//     // 5. 执行批量压缩
//     auto compress_start = std::chrono::high_resolution_clock::now();
//     for (size_t i = 0; i < batch_size; ++i) {
//         const size_t offset = i * chunk_size;
//         const size_t bytes = std::min(chunk_size, total_bytes - offset);
        
//         // 分配设备内存
//         CUDA_CHECK(cudaMalloc(&h_uncompressed_ptrs[i], bytes));
//         CUDA_CHECK(cudaMemcpyAsync(h_uncompressed_ptrs[i], 
//                                  oriData.data() + offset/sizeof(double),
//                                  bytes, cudaMemcpyHostToDevice, stream));
//         h_uncompressed_bytes[i] = bytes;
//     }
    
//     // 拷贝元数据到设备
//     CUDA_CHECK(cudaMemcpyAsync(d_uncompressed_ptrs, h_uncompressed_ptrs.data(),
//                              batch_size*sizeof(void*), cudaMemcpyHostToDevice, stream));
//     CUDA_CHECK(cudaMemcpyAsync(d_uncompressed_bytes, h_uncompressed_bytes.data(),
//                              batch_size*sizeof(size_t), cudaMemcpyHostToDevice, stream));
    
//     // 执行压缩
//     nvcompStatus_t comp_res = nvcompBatchedBitcompCompressAsync(
//         d_uncompressed_ptrs,
//         d_uncompressed_bytes,
//         chunk_size,
//         batch_size,
//         nullptr, // temp_ptr
//         0,       // temp_bytes
//         d_compressed_ptrs,
//         d_compressed_bytes,
//         opts,
//         stream);
//     ASSERT_EQ(comp_res, nvcompSuccess);
    
//     CUDA_CHECK(cudaStreamSynchronize(stream));
//     auto compress_end = std::chrono::high_resolution_clock::now();

//     // 6. 性能计算（此处省略解压部分，结构类似）
//     // ... [解压和验证代码与单块测试类似]
// }

// // Google Test 测试用例
// TEST(BitcompTest, PerformanceTest) {
//     std::string dir_path = "../test/data/float";
//     for (const auto& entry : fs::directory_iterator(dir_path)) {
//         if (entry.is_regular_file()) {
//             std::string file_path = entry.path().string();
//             std::cout << "\nTesting file: " << file_path << std::endl;
//             test_bitcomp_performance(file_path);
//             std::cout << "----------------------------------------" << std::endl;
//         }
//     }
// }

// int main(int argc, char** argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }



#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <memory>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>

// nvcomp bitcomp 头文件
#include "nvcomp/bitcomp.hpp"
#include "nvcomp.hpp"
#include "data/dataset_utils.hpp"

namespace fs = std::filesystem;

// 错误检查宏
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            FAIL() << "CUDA error: " << cudaGetErrorString(err); \
        } \
    } while(0)

#define CHECK_NVCOMP(call) \
    do { \
        nvcompStatus_t status = call; \
        if (status != nvcompSuccess) { \
            std::cerr << "nvcomp error at " << __FILE__ << ":" << __LINE__ \
                      << " - Status code: " << status << std::endl; \
            FAIL() << "nvcomp error, status code: " << status; \
        } \
    } while(0)

void comp_Bitcomp(std::vector<double> oriData);
void test_compression(const std::string& file_path);

void test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    comp_Bitcomp(oriData);
}

void comp_Bitcomp(std::vector<double> oriData) {
    std::cout << "Testing nvcomp bitcomp compression..." << std::endl;
    
    // 转换为字节数据
    const size_t data_size = oriData.size() * sizeof(double);
    const uint8_t* input_bytes = reinterpret_cast<const uint8_t*>(oriData.data());
    
    std::cout << "Input size: " << data_size << " bytes (" << oriData.size() << " doubles)" << std::endl;
    
    // 计时变量
    auto start_total = std::chrono::high_resolution_clock::now();
    auto start_kernel =start_total, end_kernel =start_total;
    
    // GPU内存分配
    uint8_t* d_input = nullptr;
    uint8_t* d_compressed = nullptr;
    uint8_t* d_decompressed = nullptr;
    
    // H2D 数据传输 - 开始
    auto start_h2d_comp = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMalloc(&d_input, data_size));
    CHECK_CUDA(cudaMemcpy(d_input, input_bytes, data_size, cudaMemcpyHostToDevice));
    auto end_h2d_comp = std::chrono::high_resolution_clock::now();
    
    try {
        // 1. 准备批处理数据结构
        const size_t batch_size = 1;
        std::vector<const void*> input_ptrs = {d_input};
        std::vector<size_t> input_sizes = {data_size};
        
        // 在GPU上分配指针数组
        const void** d_input_ptrs = nullptr;
        size_t* d_input_sizes = nullptr;
        CHECK_CUDA(cudaMalloc(&d_input_ptrs, batch_size * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_input_sizes, batch_size * sizeof(size_t)));
        CHECK_CUDA(cudaMemcpy(d_input_ptrs, input_ptrs.data(), 
                              batch_size * sizeof(void*), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_input_sizes, input_sizes.data(), 
                              batch_size * sizeof(size_t), cudaMemcpyHostToDevice));
        
        // 2. 获取压缩所需的临时内存大小
        size_t temp_bytes = 0;
        CHECK_NVCOMP(nvcompBatchedBitcompCompressGetTempSize(
            batch_size,
            data_size,
            nvcompBatchedBitcompDefaultOpts,
            &temp_bytes
        ));
        
        // 3. 获取压缩输出的最大大小
        size_t compressed_bytes = 0;
        CHECK_NVCOMP(nvcompBatchedBitcompCompressGetMaxOutputChunkSize(
            data_size,
            nvcompBatchedBitcompDefaultOpts,
            &compressed_bytes
        ));
        
        // 4. 分配临时内存和压缩输出内存
        void* d_temp = nullptr;
        if (temp_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
        }
        CHECK_CUDA(cudaMalloc(&d_compressed, compressed_bytes));
        
        // 分配压缩输出相关的GPU内存
        void** d_compressed_ptrs = nullptr;
        size_t* d_compressed_sizes = nullptr;
        CHECK_CUDA(cudaMalloc(&d_compressed_ptrs, batch_size * sizeof(void*)));
        CHECK_CUDA(cudaMalloc(&d_compressed_sizes, batch_size * sizeof(size_t)));
        
        std::vector<void*> compressed_ptrs = {d_compressed};
        CHECK_CUDA(cudaMemcpy(d_compressed_ptrs, compressed_ptrs.data(),
                              batch_size * sizeof(void*), cudaMemcpyHostToDevice));
        
        // 5. 执行压缩 - 核函数计时
        start_kernel = std::chrono::high_resolution_clock::now();
        CHECK_NVCOMP(nvcompBatchedBitcompCompressAsync(
            d_input_ptrs,
            d_input_sizes,
            data_size,
            batch_size,
            d_temp,
            temp_bytes,
            d_compressed_ptrs,
            d_compressed_sizes,
            nvcompBatchedBitcompDefaultOpts,
            0
        ));
        CHECK_CUDA(cudaStreamSynchronize(0));
        end_kernel = std::chrono::high_resolution_clock::now();
        
        // 获取实际压缩大小
        std::vector<size_t> actual_compressed_sizes(batch_size);
        CHECK_CUDA(cudaMemcpy(actual_compressed_sizes.data(), d_compressed_sizes,
                              batch_size * sizeof(size_t), cudaMemcpyDeviceToHost));
        
        size_t actual_compressed_bytes = actual_compressed_sizes[0];
        
        // D2H 压缩数据传输
        auto start_d2h_comp = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> compressed_data(actual_compressed_bytes);
        CHECK_CUDA(cudaMemcpy(compressed_data.data(), d_compressed, 
                              actual_compressed_bytes, cudaMemcpyDeviceToHost));
        auto end_d2h_comp = std::chrono::high_resolution_clock::now();
        
        // 计算压缩时间
        double compression_kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
        double compression_h2d_time = std::chrono::duration<double, std::milli>(end_h2d_comp - start_h2d_comp).count();
        double compression_d2h_time = std::chrono::duration<double, std::milli>(end_d2h_comp - start_d2h_comp).count();
        double compression_total_time = compression_h2d_time + compression_kernel_time + compression_d2h_time;
        
        std::cout << "=== 压缩结果 ===" << std::endl;
        std::cout << "压缩后大小: " << actual_compressed_bytes << " bytes" << std::endl;
        std::cout << "压缩率: " << (double)actual_compressed_bytes / data_size << std::endl;
        std::cout << "压缩核函数时间: " << compression_kernel_time << " ms" << std::endl;
        std::cout << "压缩H2D时间: " << compression_h2d_time << " ms" << std::endl;
        std::cout << "压缩D2H时间: " << compression_d2h_time << " ms" << std::endl;
        std::cout << "压缩总时间: " << compression_total_time << " ms" << std::endl;
        
        // ==================== 解压缩流程 ====================
        
        // H2D 解压缩数据传输
        auto start_h2d_decomp = std::chrono::high_resolution_clock::now();
        // 重新分配压缩数据到GPU (模拟实际场景)
        if (d_compressed) cudaFree(d_compressed);
        CHECK_CUDA(cudaMalloc(&d_compressed, actual_compressed_bytes));
        CHECK_CUDA(cudaMemcpy(d_compressed, compressed_data.data(), 
                              actual_compressed_bytes, cudaMemcpyHostToDevice));
        
        // 更新压缩数据指针
        compressed_ptrs = {d_compressed};
        CHECK_CUDA(cudaMemcpy(d_compressed_ptrs, compressed_ptrs.data(),
                              batch_size * sizeof(void*), cudaMemcpyHostToDevice));
        
        // 更新压缩大小
        std::vector<size_t> compressed_sizes = {actual_compressed_bytes};
        CHECK_CUDA(cudaMemcpy(d_compressed_sizes, compressed_sizes.data(),
                              batch_size * sizeof(size_t), cudaMemcpyHostToDevice));
        auto end_h2d_decomp = std::chrono::high_resolution_clock::now();
        
        // 6. 获取解压缩所需的临时内存大小
        size_t decomp_temp_bytes = 0;
        CHECK_NVCOMP(nvcompBatchedBitcompDecompressGetTempSize(
            batch_size,
            data_size,
            &decomp_temp_bytes
        ));
        
        // 7. 获取原始数据大小
        size_t* d_decomp_sizes = nullptr;
        CHECK_CUDA(cudaMalloc(&d_decomp_sizes, batch_size * sizeof(size_t)));
        
        CHECK_NVCOMP(nvcompBatchedBitcompGetDecompressSizeAsync(
            d_compressed_ptrs,
            d_compressed_sizes,
            d_decomp_sizes,
            batch_size,
            0
        ));
        CHECK_CUDA(cudaStreamSynchronize(0));
        
        std::vector<size_t> decomp_sizes(batch_size);
        CHECK_CUDA(cudaMemcpy(decomp_sizes.data(), d_decomp_sizes,
                              batch_size * sizeof(size_t), cudaMemcpyDeviceToHost));
        
        size_t decomp_bytes = decomp_sizes[0];
        
        // 8. 分配解压缩内存
        void* d_decomp_temp = nullptr;
        if (decomp_temp_bytes > 0) {
            CHECK_CUDA(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));
        }
        CHECK_CUDA(cudaMalloc(&d_decompressed, decomp_bytes));
        
        void** d_decompressed_ptrs = nullptr;
        CHECK_CUDA(cudaMalloc(&d_decompressed_ptrs, batch_size * sizeof(void*)));
        
        std::vector<void*> decompressed_ptrs = {d_decompressed};
        CHECK_CUDA(cudaMemcpy(d_decompressed_ptrs, decompressed_ptrs.data(),
                              batch_size * sizeof(void*), cudaMemcpyHostToDevice));
        
        // 9. 分配状态数组
        nvcompStatus_t* d_statuses = nullptr;
        CHECK_CUDA(cudaMalloc(&d_statuses, batch_size * sizeof(nvcompStatus_t)));
        
        // 10. 执行解压缩 - 核函数计时
        start_kernel = std::chrono::high_resolution_clock::now();
        CHECK_NVCOMP(nvcompBatchedBitcompDecompressAsync(
            d_compressed_ptrs,
            d_compressed_sizes,
            d_decomp_sizes,
            d_decomp_sizes,
            batch_size,
            d_decomp_temp,
            decomp_temp_bytes,
            d_decompressed_ptrs,
            d_statuses,
            0
        ));
        CHECK_CUDA(cudaStreamSynchronize(0));
        end_kernel = std::chrono::high_resolution_clock::now();
        
        // 检查解压缩状态
        std::vector<nvcompStatus_t> statuses(batch_size);
        CHECK_CUDA(cudaMemcpy(statuses.data(), d_statuses,
                              batch_size * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
        
        ASSERT_EQ(statuses[0], nvcompSuccess) << "Decompression failed with status: " << statuses[0];
        
        // D2H 解压缩数据传输
        auto start_d2h_decomp = std::chrono::high_resolution_clock::now();
        std::vector<uint8_t> output_data(decomp_bytes);
        CHECK_CUDA(cudaMemcpy(output_data.data(), d_decompressed, 
                              decomp_bytes, cudaMemcpyDeviceToHost));
        auto end_d2h_decomp = std::chrono::high_resolution_clock::now();
        
        auto end_total = std::chrono::high_resolution_clock::now();
        
        // 计算解压缩时间
        double decompression_kernel_time = std::chrono::duration<double, std::milli>(end_kernel - start_kernel).count();
        double decompression_h2d_time = std::chrono::duration<double, std::milli>(end_h2d_decomp - start_h2d_decomp).count();
        double decompression_d2h_time = std::chrono::duration<double, std::milli>(end_d2h_decomp - start_d2h_decomp).count();
        double decompression_total_time = decompression_h2d_time + decompression_kernel_time + decompression_d2h_time;
        double total_time = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        
        std::cout << "=== 解压缩结果 ===" << std::endl;
        std::cout << "解压缩后大小: " << decomp_bytes << " bytes" << std::endl;
        std::cout << "解压缩核函数时间: " << decompression_kernel_time << " ms" << std::endl;
        std::cout << "解压缩H2D时间: " << decompression_h2d_time << " ms" << std::endl;
        std::cout << "解压缩D2H时间: " << decompression_d2h_time << " ms" << std::endl;
        std::cout << "解压缩总时间: " << decompression_total_time << " ms" << std::endl;
        std::cout << "=== 总体结果 ===" << std::endl;
        std::cout << "完整流程总时间: " << total_time << " ms" << std::endl;
        
        // 11. 验证数据
        bool success = (decomp_bytes == data_size) && 
                      (memcmp(input_bytes, output_data.data(), data_size) == 0);
        
        ASSERT_TRUE(success) << "Data verification failed!";
        
        if (success) {
            std::cout << "✓ 压缩和解压缩验证成功!" << std::endl;
        }
        
        // 清理GPU内存
        if (d_temp) cudaFree(d_temp);
        if (d_decomp_temp) cudaFree(d_decomp_temp);
        if (d_input_ptrs) cudaFree(d_input_ptrs);
        if (d_input_sizes) cudaFree(d_input_sizes);
        if (d_compressed_ptrs) cudaFree(d_compressed_ptrs);
        if (d_compressed_sizes) cudaFree(d_compressed_sizes);
        if (d_decomp_sizes) cudaFree(d_decomp_sizes);
        if (d_decompressed_ptrs) cudaFree(d_decompressed_ptrs);
        if (d_statuses) cudaFree(d_statuses);
        
    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
    
    // 清理主要GPU内存
    if (d_input) cudaFree(d_input);
    if (d_compressed) cudaFree(d_compressed);
    if (d_decompressed) cudaFree(d_decompressed);
}

// Google Test 测试用例
TEST(BitcompCompressorTest, CompressionDecompression) {
    // 读取数据并测试压缩和解压
    std::string dir_path = "../test/data/float"; 
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_compression(file_path);
            std::cout << "==============================================" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}