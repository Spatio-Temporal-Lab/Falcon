#include <gtest/gtest.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <iostream>
#include <cassert>
#include <cuda_runtime.h>
#include <nvcomp/gdeflate.hpp>
#include "data/dataset_utils.hpp"
#include <filesystem>
namespace fs = std::filesystem;

#define avg_times 3 

CompressionInfo test_compression(const std::string& file_path) {
    // 读取数据
    std::vector<float> oriData = read_data_float(file_path);
    size_t in_bytes = oriData.size() * sizeof(float);
    if (in_bytes == 0) {
        std::cerr << "Error: Empty file or read failure: " << file_path << std::endl;
        return CompressionInfo{};
    }

    // 创建CUDA流和事件
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaEvent_t start_event, end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    // 分配设备内存
    uint8_t* device_input_data = nullptr;
    uint8_t* device_compressed_data = nullptr;
    uint8_t* device_decompressed_data = nullptr;
    
    cudaError_t err = cudaMalloc(&device_input_data, in_bytes);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device_input_data: " << cudaGetErrorString(err) << std::endl;
        return CompressionInfo{};
    }

    // 创建 GDeflate 管理器
    const size_t chunk_size = 65536;
    nvcompBatchedGdeflateOpts_t format_opts = {0};
    nvcomp::GdeflateManager manager{chunk_size, format_opts, stream};

    // 配置压缩
    nvcomp::CompressionConfig comp_config = manager.configure_compression(in_bytes);
    err = cudaMalloc(&device_compressed_data, comp_config.max_compressed_buffer_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device_compressed_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input_data);
        return CompressionInfo{};
    }

    // =========================== 压缩流程测量 ===========================
    
    // 1. 测量H2D传输时间（压缩）
    auto start_h2d_compress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(device_input_data, oriData.data(), in_bytes, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_h2d_compress = std::chrono::high_resolution_clock::now();
    
    float h2d_compress_time_ms;
    cudaEventElapsedTime(&h2d_compress_time_ms, start_event, end_event);
    double h2d_compress_time = std::chrono::duration<double>(end_h2d_compress - start_h2d_compress).count();

    // 2. 测量压缩核函数时间
    auto start_compress_kernel = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    manager.compress(device_input_data, device_compressed_data, comp_config);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_compress_kernel = std::chrono::high_resolution_clock::now();
    
    float compress_kernel_time_ms;
    cudaEventElapsedTime(&compress_kernel_time_ms, start_event, end_event);
    double compress_kernel_time = std::chrono::duration<double>(end_compress_kernel - start_compress_kernel).count();

    // 获取压缩后大小
    size_t comp_out_bytes = manager.get_compressed_output_size(device_compressed_data);

    // 3. 测量D2H传输时间（压缩结果，可选）
    std::vector<uint8_t> compressed_host_data(comp_out_bytes);
    auto start_d2h_compress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(compressed_host_data.data(), device_compressed_data, comp_out_bytes, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_d2h_compress = std::chrono::high_resolution_clock::now();
    
    float d2h_compress_time_ms;
    cudaEventElapsedTime(&d2h_compress_time_ms, start_event, end_event);
    double d2h_compress_time = std::chrono::duration<double>(end_d2h_compress - start_d2h_compress).count();

    // 压缩总时间
    double total_compress_time = h2d_compress_time + compress_kernel_time + d2h_compress_time;

    // =========================== 解压流程测量 ===========================
    
    // 配置解压
    nvcomp::DecompressionConfig decomp_config = manager.configure_decompression(device_compressed_data);
    err = cudaMalloc(&device_decompressed_data, decomp_config.decomp_data_size);
    if (err != cudaSuccess) {
        std::cerr << "Error allocating device_decompressed_data: " << cudaGetErrorString(err) << std::endl;
        cudaFree(device_input_data);
        cudaFree(device_compressed_data);
        return CompressionInfo{};
    }
    
    // 1. 测量H2D传输时间（解压，如果需要从主机传输压缩数据）
    auto start_h2d_decompress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(device_compressed_data, compressed_host_data.data(), comp_out_bytes, cudaMemcpyHostToDevice, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_h2d_decompress = std::chrono::high_resolution_clock::now();
    
    float h2d_decompress_time_ms;
    cudaEventElapsedTime(&h2d_decompress_time_ms, start_event, end_event);
    double h2d_decompress_time = std::chrono::duration<double>(end_h2d_decompress - start_h2d_decompress).count();

    // 2. 测量解压核函数时间
    auto start_decompress_kernel = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    manager.decompress(device_decompressed_data, device_compressed_data, decomp_config);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_decompress_kernel = std::chrono::high_resolution_clock::now();
    
    float decompress_kernel_time_ms;
    cudaEventElapsedTime(&decompress_kernel_time_ms, start_event, end_event);
    double decompress_kernel_time = std::chrono::duration<double>(end_decompress_kernel - start_decompress_kernel).count();

    // 3. 测量D2H传输时间（解压结果）
    std::vector<float> decompressedData(oriData.size());
    auto start_d2h_decompress = std::chrono::high_resolution_clock::now();
    cudaEventRecord(start_event, stream);
    cudaMemcpyAsync(decompressedData.data(), device_decompressed_data, in_bytes, cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(end_event, stream);
    cudaStreamSynchronize(stream);
    auto end_d2h_decompress = std::chrono::high_resolution_clock::now();
    
    float d2h_decompress_time_ms;
    cudaEventElapsedTime(&d2h_decompress_time_ms, start_event, end_event);
    double d2h_decompress_time = std::chrono::duration<double>(end_d2h_decompress - start_d2h_decompress).count();

    // 解压总时间
    double total_decompress_time = h2d_decompress_time + decompress_kernel_time + d2h_decompress_time;

    // =========================== 数据验证 ===========================
    const float tolerance = 1e-9;
    bool valid = true;
    for (size_t i = 0; i < oriData.size(); ++i) {
        if (std::abs(oriData[i] - decompressedData[i]) > tolerance) {
            valid = false;
            std::cerr << "Mismatch at position " << i 
                      << ": original=" << oriData[i] 
                      << ", decompressed=" << decompressedData[i] 
                      << std::endl;
            break;
        }
    }

    if (!valid) {
        std::cerr << "Data mismatch detected in " << file_path << "!" << std::endl;
        // FAIL() << "Data validation failed for file: " << file_path;
    } else {
        std::cout << "Decompression validated successfully." << std::endl;
    }

    // =========================== 性能计算 ===========================
    double compression_ratio = static_cast<double>(comp_out_bytes) / in_bytes; // 压缩率（小于1）
    double data_size_gb = in_bytes / (1024.0 * 1024.0 * 1024.0);
    double compress_throughput = data_size_gb / (total_compress_time );
    double decompress_throughput = data_size_gb / (total_decompress_time);


    // =========================== 结果输出 ===========================
        CompressionInfo ans{
            in_bytes/1024.0/1024.0,
            static_cast<double>(comp_out_bytes) /1024.0/1024.0,
            compression_ratio,
            compress_kernel_time * 1000,
            total_compress_time * 1000 ,
            compress_throughput,
            decompress_kernel_time * 1000 ,
            total_decompress_time * 1000,
            decompress_throughput};

//     std::cout << "File: " << fs::path(file_path).filename() << std::endl;
// //    std::cout << "Compression Ratio: " << compression_ratio << std::endl;
//     printf("Compression Ratio: %0.3f\n",compression_ratio);
//     std::cout << "Total Compress Time: " << total_compress_time * 1000 << " ms" << std::endl;
//     std::cout << "Compress Kernel Time: " << compress_kernel_time * 1000 << " ms" << std::endl;
//     std::cout << "Total Decompress Time: " << total_decompress_time * 1000 << " ms" << std::endl;
//     std::cout << "Decompress Kernel Time: " << decompress_kernel_time * 1000 << " ms" << std::endl;
//     std::cout << "Data Size: " << data_size_gb << " GB" << std::endl;
//     std::cout << "Compression Throughput: " << compress_throughput << " GB/s" << std::endl;
//     std::cout << "Decompression Throughput: " << decompress_throughput << " GB/s" << std::endl;
    // =========================== 清理资源 ===========================
    
    if (device_input_data) {
        err = cudaFree(device_input_data);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing device_input_data: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    if (device_compressed_data) {
        err = cudaFree(device_compressed_data);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing device_compressed_data: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    if (device_decompressed_data) {
        err = cudaFree(device_decompressed_data);
        if (err != cudaSuccess) {
            std::cerr << "Error freeing device_decompressed_data: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    // err = cudaEventDestroy(start_event);
    // if (err != cudaSuccess) {
    //     std::cerr << "Error destroying start_event: " << cudaGetErrorString(err) << std::endl;
    // }
    
    // err = cudaEventDestroy(end_event);
    // if (err != cudaSuccess) {
    //     std::cerr << "Error destroying end_event: " << cudaGetErrorString(err) << std::endl;
    // }
    
    // err = cudaStreamDestroy(stream);
    // if (err != cudaSuccess) {
    //     std::cerr << "Error destroying stream: " << cudaGetErrorString(err) << std::endl;
    // }
    return ans;
}
// Google Test 测试用例
TEST(GDeflateCompressorTest, CompressionDecompression) {
    std::string dir_path = "../test/data/new_tsbs";
    if (!fs::exists(dir_path)) {
        GTEST_SKIP() << "Data directory not found: " << dir_path;
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