
#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <nvcomp/lz4.hpp>  // 引入 nvcomp LZ4 库
#include <cassert>
#include <nvcomp/lz4.h>
#include <cuda_runtime.h>
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;
CompressionInfo comp_LZ4(std::vector<double> oriData);
CompressionInfo test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path);
    return comp_LZ4(oriData);
}

CompressionInfo comp_LZ4(std::vector<double> oriData)
{
    // 获取数据大小
    size_t in_bytes = oriData.size() * sizeof(double);

    // 将数据复制到char类型的数组中，因为LZ4压缩处理的是字节数组
    char* input_data = reinterpret_cast<char*>(oriData.data());
    
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建CUDA事件用于精确计时
    cudaEvent_t start_event, end_event;
    cudaEvent_t compress_h2d_start, compress_h2d_end;
    cudaEvent_t compress_kernel_start, compress_kernel_end;
    cudaEvent_t compress_d2h_start, compress_d2h_end;
    cudaEvent_t decompress_h2d_start, decompress_h2d_end;
    cudaEvent_t decompress_kernel_start, decompress_kernel_end;
    cudaEvent_t decompress_d2h_start, decompress_d2h_end;

    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);
    cudaEventCreate(&compress_h2d_start);
    cudaEventCreate(&compress_h2d_end);
    cudaEventCreate(&compress_kernel_start);
    cudaEventCreate(&compress_kernel_end);
    cudaEventCreate(&compress_d2h_start);
    cudaEventCreate(&compress_d2h_end);
    cudaEventCreate(&decompress_h2d_start);
    cudaEventCreate(&decompress_h2d_end);
    cudaEventCreate(&decompress_kernel_start);
    cudaEventCreate(&decompress_kernel_end);
    cudaEventCreate(&decompress_d2h_start);
    cudaEventCreate(&decompress_d2h_end);

    // 计算块大小
    const size_t chunk_size = 65536;
    const size_t batch_size = (in_bytes + chunk_size - 1) / chunk_size;

    // 分配CUDA内存
    char* device_input_data;
    cudaMalloc(&device_input_data, in_bytes);

    size_t* host_uncompressed_bytes;
    cudaMallocHost(&host_uncompressed_bytes, sizeof(size_t) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        if (i + 1 < batch_size) {
            host_uncompressed_bytes[i] = chunk_size;
        } else {
            host_uncompressed_bytes[i] = in_bytes - (chunk_size * i);
        }
    }

    // 设置每个块的指针
    void** host_uncompressed_ptrs;
    cudaMallocHost(&host_uncompressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        host_uncompressed_ptrs[i] = device_input_data + chunk_size * i;
    }

    size_t* device_uncompressed_bytes;
    void** device_uncompressed_ptrs;
    cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
    cudaMalloc(&device_uncompressed_ptrs, sizeof(void*) * batch_size);

    // 分配临时空间
    size_t temp_bytes;
    nvcompBatchedLZ4CompressGetTempSize(batch_size, chunk_size, nvcompBatchedLZ4DefaultOpts, &temp_bytes);
    void* device_temp_ptr;
    cudaMalloc(&device_temp_ptr, temp_bytes);

    // 获取每个块的最大输出大小
    size_t max_out_bytes;
    nvcompBatchedLZ4CompressGetMaxOutputChunkSize(chunk_size, nvcompBatchedLZ4DefaultOpts, &max_out_bytes);

    // 分配压缩后的输出空间
    void** host_compressed_ptrs;
    cudaMallocHost(&host_compressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        cudaMalloc(&host_compressed_ptrs[i], max_out_bytes);
    }

    void** device_compressed_ptrs;
    cudaMalloc(&device_compressed_ptrs, sizeof(void*) * batch_size);

    size_t* device_compressed_bytes;
    cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);

    // ======================== 压缩阶段 ========================
    // std::cout << "=== Compression Phase ===" << std::endl;
    
    // 记录整个压缩阶段开始
    cudaEventRecord(start_event, stream);
    
    // H2D: Host to Device 数据传输
    cudaEventRecord(compress_h2d_start, stream);
    
    cudaMemcpyAsync(device_input_data, input_data, in_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_ptrs, host_uncompressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    
    cudaEventRecord(compress_h2d_end, stream);

    // KERNEL: 压缩核函数执行
    cudaEventRecord(compress_kernel_start, stream);
    
    // 压缩数据
    nvcompStatus_t comp_res = nvcompBatchedLZ4CompressAsync(
        device_uncompressed_ptrs,
        device_uncompressed_bytes,
        chunk_size,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        device_compressed_ptrs,
        device_compressed_bytes,
        nvcompBatchedLZ4DefaultOpts,
        stream);
    cudaDeviceSynchronize();
    if (comp_res != nvcompSuccess) {
        std::cerr << "Compression failed!" << std::endl;
        assert(comp_res == nvcompSuccess);
    }
    
    cudaEventRecord(compress_kernel_end, stream);

    // D2H: Device to Host 压缩结果传输
    cudaEventRecord(compress_d2h_start, stream);
    
    size_t* host_compressed_bytes = new size_t[batch_size];
    cudaMemcpyAsync(host_compressed_bytes, device_compressed_bytes, 
                   sizeof(size_t) * batch_size, cudaMemcpyDeviceToHost, stream);
    
    // 将压缩后的数据传回主机
    char** host_compressed_data = new char*[batch_size];
    for (size_t i = 0; i < batch_size; ++i) {
        host_compressed_data[i] = new char[max_out_bytes];
        cudaMemcpyAsync(host_compressed_data[i], host_compressed_ptrs[i], 
                       max_out_bytes, cudaMemcpyDeviceToHost, stream);
    }
    
    cudaEventRecord(compress_d2h_end, stream);
    
    // 等待压缩阶段完成并计算时间
    cudaStreamSynchronize(stream);
    
    float compress_h2d_time, compress_kernel_time, compress_d2h_time;
    cudaEventElapsedTime(&compress_h2d_time, compress_h2d_start, compress_h2d_end);
    cudaEventElapsedTime(&compress_kernel_time, compress_kernel_start, compress_kernel_end);
    cudaEventElapsedTime(&compress_d2h_time, compress_d2h_start, compress_d2h_end);
    
    float total_compress_time = compress_h2d_time + compress_kernel_time + compress_d2h_time;


    // 计算压缩率
    size_t total_compressed = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        total_compressed += host_compressed_bytes[i];
    }
    double compression_ratio = total_compressed / static_cast<double>(in_bytes);

    // ======================== 解压阶段 ========================
    // std::cout << "\n=== Decompression Phase ===" << std::endl;
    
    // 重新分配设备内存用于解压
    char* device_output_data;
    cudaMalloc(&device_output_data, in_bytes);
    
    // 重新设置解压输出指针
    void** host_decompressed_ptrs;
    cudaMallocHost(&host_decompressed_ptrs, sizeof(void*) * batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        host_decompressed_ptrs[i] = device_output_data + chunk_size * i;
    }
    
    void** device_decompressed_ptrs;
    cudaMalloc(&device_decompressed_ptrs, sizeof(void*) * batch_size);

    // 分配解压临时缓冲区
    size_t decomp_temp_bytes;
    nvcompBatchedLZ4DecompressGetTempSize(batch_size, chunk_size, &decomp_temp_bytes);
    void* device_decomp_temp;
    cudaMalloc(&device_decomp_temp, decomp_temp_bytes);

    nvcompStatus_t* device_statuses;
    cudaMalloc(&device_statuses, sizeof(nvcompStatus_t) * batch_size);

    size_t* device_actual_uncompressed_bytes;
    cudaMalloc(&device_actual_uncompressed_bytes, sizeof(size_t) * batch_size);

    // H2D: 将压缩数据传回设备
    cudaEventRecord(decompress_h2d_start, stream);
    
    // 重新分配设备压缩数据空间并传输
    for (size_t i = 0; i < batch_size; ++i) {
        cudaMemcpyAsync(host_compressed_ptrs[i], host_compressed_data[i], 
                       host_compressed_bytes[i], cudaMemcpyHostToDevice, stream);
    }
    
    cudaMemcpyAsync(device_compressed_bytes, host_compressed_bytes, 
                   sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_compressed_ptrs, host_compressed_ptrs, 
                   sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_decompressed_ptrs, host_decompressed_ptrs, 
                   sizeof(void*) * batch_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_uncompressed_bytes, host_uncompressed_bytes, 
                   sizeof(size_t) * batch_size, cudaMemcpyHostToDevice, stream);
    
    cudaEventRecord(decompress_h2d_end, stream);

    // KERNEL: 解压核函数执行
    cudaEventRecord(decompress_kernel_start, stream);
    
    // 解压数据
    nvcompStatus_t decomp_res = nvcompBatchedLZ4DecompressAsync(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        device_actual_uncompressed_bytes,
        batch_size,
        device_decomp_temp,
        decomp_temp_bytes,
        device_decompressed_ptrs,
        device_statuses,
        stream);
    cudaDeviceSynchronize();
    if (decomp_res != nvcompSuccess) {
        std::cerr << "Decompression failed!" << std::endl;
        assert(decomp_res == nvcompSuccess);
    }

    cudaEventRecord(decompress_kernel_end, stream);

    // D2H: 解压结果传回主机
    cudaEventRecord(decompress_d2h_start, stream);
    
    char* reconstructed = new char[in_bytes];
    cudaMemcpyAsync(reconstructed, device_output_data, in_bytes, 
                   cudaMemcpyDeviceToHost, stream);
    
    cudaEventRecord(decompress_d2h_end, stream);
    cudaEventRecord(end_event, stream);

    // 等待所有操作完成并计算时间
    cudaStreamSynchronize(stream);
    
    float decompress_h2d_time, decompress_kernel_time, decompress_d2h_time;
    cudaEventElapsedTime(&decompress_h2d_time, decompress_h2d_start, decompress_h2d_end);
    cudaEventElapsedTime(&decompress_kernel_time, decompress_kernel_start, decompress_kernel_end);
    cudaEventElapsedTime(&decompress_d2h_time, decompress_d2h_start, decompress_d2h_end);
    
    float total_decompress_time = decompress_h2d_time + decompress_kernel_time + decompress_d2h_time;
    float total_overall_time;
    cudaEventElapsedTime(&total_overall_time, start_event, end_event);
    // 吞吐量计算
    double data_size_gb = in_bytes / (1024.0 * 1024.0 * 1024.0);
    double compress_throughput = data_size_gb / (total_compress_time / 1000.0);
    double decompress_throughput = data_size_gb / (total_decompress_time / 1000.0);

    CompressionInfo a{
        in_bytes/1024.0/1024.0,
        total_compressed/1024.0/1024.0,
        compression_ratio,
        compress_kernel_time,
        total_compress_time,
        compress_throughput,
        decompress_kernel_time,
        total_decompress_time,
        decompress_throughput

    };

    // 数据校验
    if (memcmp(input_data, reconstructed, in_bytes) != 0) {
        std::cout << "Data mismatch!" << std::endl;
    } else {
        // std::cout << "\nData verification: PASSED" << std::endl;
    }

    // 清理资源
    delete[] host_compressed_bytes;
    for (size_t i = 0; i < batch_size; ++i) {
        delete[] host_compressed_data[i];
    }
    delete[] host_compressed_data;
    delete[] reconstructed;

    // 清理CUDA事件
    cudaEventDestroy(start_event);
    cudaEventDestroy(end_event);
    cudaEventDestroy(compress_h2d_start);
    cudaEventDestroy(compress_h2d_end);
    cudaEventDestroy(compress_kernel_start);
    cudaEventDestroy(compress_kernel_end);
    cudaEventDestroy(compress_d2h_start);
    cudaEventDestroy(compress_d2h_end);
    cudaEventDestroy(decompress_h2d_start);
    cudaEventDestroy(decompress_h2d_end);
    cudaEventDestroy(decompress_kernel_start);
    cudaEventDestroy(decompress_kernel_end);
    cudaEventDestroy(decompress_d2h_start);
    cudaEventDestroy(decompress_d2h_end);

    // 清理CUDA资源
    cudaStreamSynchronize(stream);
    cudaFree(device_input_data);
    cudaFree(device_output_data);
    cudaFree(device_uncompressed_bytes);
    cudaFree(device_uncompressed_ptrs);
    cudaFree(device_temp_ptr);
    cudaFree(device_compressed_ptrs);
    cudaFree(device_compressed_bytes);
    cudaFree(device_decomp_temp);
    cudaFree(device_statuses);
    cudaFree(device_actual_uncompressed_bytes);
    cudaFree(device_decompressed_ptrs);
    
    cudaFreeHost(host_uncompressed_bytes);
    cudaFreeHost(host_uncompressed_ptrs);
    cudaFreeHost(host_compressed_ptrs);
    cudaFreeHost(host_decompressed_ptrs);
    
    cudaStreamDestroy(stream);
    return a;
}

// Google Test 测试用例
TEST(LZ4CompressorTest, CompressionDecompression) {
    // 读取数据并测试压缩和解压
    // std::string dir_path = "../test/data/big"; 
    std::string dir_path = "../test/data/mew_tsbs"; 
    bool warmup = 0;

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            if(!warmup)
            {
                warmup=1;
                std::cout << "====================warmup==========================" << std::endl;
                test_compression(file_path);
                std::cout << "====================warmup_end=========================" << std::endl;
            }
            std::cout << "正在处理文件: " << file_path << std::endl;
            test_compression(file_path);
            std::cout << "==============================================" << std::endl;
        }
    }
}

int main(int argc, char *argv[]) {
    
    cudaFree(0);
    std::string arg = argv[1];
    
    if (arg == "--dir" && argc >= 3) {

        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        bool warm=0;
        int processed = 0;
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                CompressionInfo a;
                if(!warm)
                {
                    // std::cout << "\n-------------------warm-------------------------- " << file_path << std::endl;
                    test_compression(file_path);
                    warm=1;
                    // std::cout << "-------------------warm_end------------------------" << std::endl;
                }
                std::cout << "\nProcessing file: " << file_path << std::endl;
                for(int i=0;i<3;i++)
                {
                    cudaDeviceReset();
                    a+=test_compression(file_path);
                }
                a=a/3;
                a.print();
                std::cout << "---------------------------------------------" << std::endl;
                processed++;
            }
        }
        
        if (processed == 0) {
            std::cerr << "No files found in directory: " << dir_path << std::endl;
        }
    }
    else{
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
}