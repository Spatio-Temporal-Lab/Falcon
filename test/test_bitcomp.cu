#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/bitcomp.h>
#include <nvcomp.h>
#include <nvcomp.hpp>
#include "data/dataset_utils.hpp"
#include <filesystem>

namespace fs = std::filesystem;

#define CUDA_CHECK(cond) { gpuAssert((cond), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

void test_bitcomp_performance(const std::string& file_path) {
    // 1. 读取数据
    std::vector<double> oriData = read_data(file_path);
    const size_t in_bytes = oriData.size() * sizeof(double);
    const size_t num_elems = oriData.size();

    // 2. 准备GPU资源
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    double* d_in_data;
    CUDA_CHECK(cudaMalloc(&d_in_data, in_bytes));
    CUDA_CHECK(cudaMemcpyAsync(d_in_data, oriData.data(), in_bytes, 
                             cudaMemcpyHostToDevice, stream));

    // 3. 初始化Bitcomp管理器
    nvcomp::BitcompManager manager(NVCOMP_TYPE_UINT, 0, stream);
    auto comp_config = manager.configure_compression(in_bytes);

    // 4. 分配压缩缓冲区
    uint8_t* d_comp_out;
    CUDA_CHECK(cudaMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

    // 5. 执行压缩
    auto compress_start = std::chrono::high_resolution_clock::now();
    manager.compress(reinterpret_cast<const uint8_t*>(d_in_data),
                    d_comp_out, comp_config);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto compress_end = std::chrono::high_resolution_clock::now();
    
    // 6. 获取压缩结果
    const size_t comp_bytes = manager.get_compressed_output_size(d_comp_out);
    const double compress_time = std::chrono::duration<double>(compress_end - compress_start).count();
    const double compress_throughput = (in_bytes / 1e9) / compress_time;

    // 7. 准备解压
    auto decomp_config = manager.configure_decompression(d_comp_out);
    double* d_out_data;
    CUDA_CHECK(cudaMalloc(&d_out_data, decomp_config.decomp_data_size));
    CUDA_CHECK(cudaMemsetAsync(d_out_data, 0, decomp_config.decomp_data_size, stream));

    // 8. 执行解压
    auto decompress_start = std::chrono::high_resolution_clock::now();
    manager.decompress(reinterpret_cast<uint8_t*>(d_out_data),
                      d_comp_out, decomp_config);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto decompress_end = std::chrono::high_resolution_clock::now();
    
    // 9. 计算性能指标
    const double decompress_time = std::chrono::duration<double>(decompress_end - decompress_start).count();
    const double decompress_throughput = (in_bytes / 1e9) / decompress_time;
    const double compression_ratio = static_cast<double>(comp_bytes) / in_bytes;

    // 10. 验证数据完整性
    std::vector<double> decompressed(num_elems);
    CUDA_CHECK(cudaMemcpy(decompressed.data(), d_out_data, in_bytes, 
                        cudaMemcpyDeviceToHost));
    ASSERT_EQ(oriData, decompressed);

    // 11. 输出结果
    std::cout << "\n[Bitcomp Performance Report]" << std::endl;
    std::cout << "Original Size:    " << in_bytes/1e6 << " MB" << std::endl;
    std::cout << "Compressed Size:  " << comp_bytes/1e6 << " MB" << std::endl;
    std::cout << "Compression Time: " << compress_time << " sec" << std::endl;
    std::cout << "Compression Rate: " << compress_throughput << " GB/s" << std::endl;
    std::cout << "Decompression Time: " << decompress_time << " sec" << std::endl;
    std::cout << "Decompression Rate: " << decompress_throughput << " GB/s" << std::endl;
    std::cout << "Compression Ratio: " << compression_ratio << std::endl;

    // 12. 清理资源
    CUDA_CHECK(cudaFree(d_in_data));
    CUDA_CHECK(cudaFree(d_comp_out));
    CUDA_CHECK(cudaFree(d_out_data));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

/************************ 批量测试实现 ************************/
void test_bitcomp_batched(const std::string& file_path) {
    // 1. 数据准备
    std::vector<double> oriData = read_data(file_path);
    const size_t total_bytes = oriData.size() * sizeof(double);
    const size_t chunk_size = 1 << 20; // 1MB chunks
    const size_t batch_size = (total_bytes + chunk_size - 1) / chunk_size;

    // 2. 准备批量数据
    std::vector<void*> h_uncompressed_ptrs(batch_size);
    std::vector<size_t> h_uncompressed_bytes(batch_size);
    
    // 3. 分配GPU资源
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // 批量输入指针
    void** d_uncompressed_ptrs;
    size_t* d_uncompressed_bytes;
    CUDA_CHECK(cudaMalloc(&d_uncompressed_ptrs, batch_size*sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_uncompressed_bytes, batch_size*sizeof(size_t)));

    // 批量输出指针
    void** d_compressed_ptrs;
    size_t* d_compressed_bytes;
    CUDA_CHECK(cudaMalloc(&d_compressed_ptrs, batch_size*sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_compressed_bytes, batch_size*sizeof(size_t)));

    // 4. 配置压缩参数
    nvcompBatchedBitcompFormatOpts opts{nvcompBatchedBitcompDefaultOpts};
    opts.data_type = NVCOMP_TYPE_UINT;

    // 5. 执行批量压缩
    auto compress_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < batch_size; ++i) {
        const size_t offset = i * chunk_size;
        const size_t bytes = std::min(chunk_size, total_bytes - offset);
        
        // 分配设备内存
        CUDA_CHECK(cudaMalloc(&h_uncompressed_ptrs[i], bytes));
        CUDA_CHECK(cudaMemcpyAsync(h_uncompressed_ptrs[i], 
                                 oriData.data() + offset/sizeof(double),
                                 bytes, cudaMemcpyHostToDevice, stream));
        h_uncompressed_bytes[i] = bytes;
    }
    
    // 拷贝元数据到设备
    CUDA_CHECK(cudaMemcpyAsync(d_uncompressed_ptrs, h_uncompressed_ptrs.data(),
                             batch_size*sizeof(void*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_uncompressed_bytes, h_uncompressed_bytes.data(),
                             batch_size*sizeof(size_t), cudaMemcpyHostToDevice, stream));
    
    // 执行压缩
    nvcompStatus_t comp_res = nvcompBatchedBitcompCompressAsync(
        d_uncompressed_ptrs,
        d_uncompressed_bytes,
        chunk_size,
        batch_size,
        nullptr, // temp_ptr
        0,       // temp_bytes
        d_compressed_ptrs,
        d_compressed_bytes,
        opts,
        stream);
    ASSERT_EQ(comp_res, nvcompSuccess);
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto compress_end = std::chrono::high_resolution_clock::now();

    // 6. 性能计算（此处省略解压部分，结构类似）
    // ... [解压和验证代码与单块测试类似]
}

// Google Test 测试用例
TEST(BitcompTest, PerformanceTest) {
    std::string dir_path = "../test/data/float";
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "\nTesting file: " << file_path << std::endl;
            test_bitcomp_performance(file_path);
            std::cout << "----------------------------------------" << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}