// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <cassert>
// #include <nvcomp.h>
// #include "nvcomp/bitcomp.h"
// #include "nvcomp/bitcomp.hpp"
// #include <filesystem>
// #include "data/dataset_utils.hpp"
// namespace fs = std::filesystem;
// using namespace nvcomp;

// void test_compression(const std::string& file_path) {
//     // 读取数据
//     std::vector<double> oriData = read_data(file_path);

//     // 获取原始数据的字节数
//     size_t in_bytes = oriData.size() * sizeof(double);

//     // 将原始数据复制到设备内存
//     char* device_input_data;
//     cudaMalloc(&device_input_data, in_bytes);
//     cudaMemcpy(device_input_data, oriData.data(), in_bytes, cudaMemcpyHostToDevice);

//     // 获取压缩和解压缩所需的流
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);

//     // 初始化压缩相关的参数
//     nvcompBatchedBitcompFormatOpts format_opts = {};
//     format_opts.data_type = NVCOMP_TYPE_DOUBLE;
//     format_opts.algorithm_type = NVCOMP_ALGORITHM_BITCOMP;

//     // 设置最大块大小和批量大小
//     size_t batch_size = 1;
//     size_t chunk_size = in_bytes;

//     // 为压缩数据分配空间
//     void* device_compressed_data;
//     size_t* device_compressed_bytes;
//     size_t max_compressed_size;
//     nvcompBatchedBitcompCompressGetMaxOutputChunkSize(chunk_size, format_opts, &max_compressed_size);
//     cudaMalloc(&device_compressed_data, max_compressed_size);
//     cudaMalloc(&device_compressed_bytes, sizeof(size_t) * batch_size);

//     // 压缩数据并记录时间
//     auto start_compress = std::chrono::high_resolution_clock::now();
//     nvcompStatus_t compress_status = nvcompBatchedBitcompCompressAsync(
//         &device_input_data, 
//         &in_bytes, 
//         0, 
//         batch_size, 
//         nullptr, 
//         0, 
//         &device_compressed_data, 
//         device_compressed_bytes, 
//         format_opts, 
//         stream
//     );
//     cudaStreamSynchronize(stream);
//     auto end_compress = std::chrono::high_resolution_clock::now();

//     // 计算压缩时间
//     std::chrono::duration<double> compress_duration = end_compress - start_compress;
//     std::cout << "Compression time: " << compress_duration.count() << " seconds." << std::endl;

//     // 获取解压后的数据空间
//     size_t* device_uncompressed_bytes;
//     void* device_decompressed_data;
//     cudaMalloc(&device_uncompressed_bytes, sizeof(size_t) * batch_size);
//     cudaMalloc(&device_decompressed_data, in_bytes);

//     // 解压缩数据并记录时间
//     auto start_decompress = std::chrono::high_resolution_clock::now();
//     nvcompStatus_t decompress_status = nvcompBatchedBitcompDecompressAsync(
//         &device_compressed_data, 
//         nullptr, 
//         device_uncompressed_bytes, 
//         &in_bytes, 
//         batch_size, 
//         nullptr, 
//         0, 
//         &device_decompressed_data, 
//         nullptr, 
//         stream
//     );
//     cudaStreamSynchronize(stream);
//     auto end_decompress = std::chrono::high_resolution_clock::now();

//     // 计算解压时间
//     std::chrono::duration<double> decompress_duration = end_decompress - start_decompress;
//     std::cout << "Decompression time: " << decompress_duration.count() << " seconds." << std::endl;

//     // 获取压缩后的数据大小
//     size_t compressed_size;
//     cudaMemcpy(&compressed_size, device_compressed_bytes, sizeof(size_t), cudaMemcpyDeviceToHost);

//     // 计算压缩率
//     double compression_ratio = static_cast<double>(compressed_size) / in_bytes;
//     std::cout << "Compression ratio: " << compression_ratio << std::endl;

//     // 验证解压缩数据
//     std::vector<double> decompressed_data(oriData.size());
//     cudaMemcpy(decompressed_data.data(), device_decompressed_data, in_bytes, cudaMemcpyDeviceToHost);
//     for (size_t i = 0; i < oriData.size(); ++i) {
//         assert(oriData[i] == decompressed_data[i] && "Data mismatch after decompression!");
//     }
//     std::cout << "Decompression verified successfully!" << std::endl;

//     // 清理内存
//     cudaFree(device_input_data);
//     cudaFree(device_compressed_data);
//     cudaFree(device_compressed_bytes);
//     cudaFree(device_uncompressed_bytes);
//     cudaFree(device_decompressed_data);
//     cudaStreamDestroy(stream);
// }


// // Google Test 测试用例
// TEST(LZ4CompressorTest, CompressionDecompression) {
//     // 读取数据并测试压缩和解压
//     std::string dir_path = "../test/data/float"; 
//     for (const auto& entry : fs::directory_iterator(dir_path)) {
//         if (entry.is_regular_file()) {
//             std::string file_path = entry.path().string();
//             std::cout << "正在处理文件: " << file_path << std::endl;
//             test_compression(file_path);
//             std::cout << "---------------------------------------------" << std::endl;
//         }
//     }
// }

// int main(int argc, char** argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }

// #include <nvcomp/cascaded.h>
// #include <cuda_runtime.h>
// #include <iostream>
// #include <vector>
// #include <cassert>

// #define CHECK_CUDA(call) {\
//     cudaError_t err = call; \
//     if (err != cudaSuccess) { \
//         std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
//         exit(err); \
//     } \
// }

// int main()
// {
//     // 要压缩的数据
//     const size_t data_size = 1024;
//     std::vector<int> input_data(data_size);
//     for (size_t i = 0; i < data_size; ++i) {
//         input_data[i] = static_cast<int>(i % 100);
//     }

//     // 分配 GPU 内存并拷贝数据
//     int* d_input_data;
//     CHECK_CUDA(cudaMalloc(&d_input_data, data_size * sizeof(int)));
//     CHECK_CUDA(cudaMemcpy(d_input_data, input_data.data(), data_size * sizeof(int), cudaMemcpyHostToDevice));

//     // 配置 Cascaded 压缩参数
//     nvcompCascadedFormatOpts options = {};
//     options.num_RLEs = 1;
//     options.num_deltas = 1;
//     options.use_bp = 1;

//     // 获取临时缓冲区大小
//     size_t temp_bytes;
//     nvcompStatus_t status = nvcompCascadedCompressGetTempSize(
//         data_size * sizeof(int),
//         options,
//         &temp_bytes);
//     assert(status == nvcompSuccess);

//     // 分配临时缓冲区
//     void* d_temp;
//     CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));

//     // 获取压缩后数据的最大大小
//     size_t max_compressed_size;
//     status = nvcompCascadedCompressGetOutputSize(
//         d_input_data,
//         data_size * sizeof(int),
//         options,
//         d_temp,
//         temp_bytes,
//         &max_compressed_size);
//     assert(status == nvcompSuccess);

//     // 分配压缩后数据的存储空间
//     void* d_compressed_data;
//     CHECK_CUDA(cudaMalloc(&d_compressed_data, max_compressed_size));

//     // 开始压缩
//     size_t compressed_size;
//     status = nvcompCascadedCompressAsync(
//         d_input_data,
//         data_size * sizeof(int),
//         d_temp,
//         temp_bytes,
//         d_compressed_data,
//         &compressed_size,
//         options,
//         0);  // 使用默认 CUDA Stream
//     assert(status == nvcompSuccess);
//     CHECK_CUDA(cudaStreamSynchronize(0));

//     std::cout << "Compression successful! Compressed size: " << compressed_size << " bytes" << std::endl;

//     // 解压缩
//     // 获取解压缩后数据大小
//     size_t decompressed_size;
//     status = nvcompCascadedDecompressGetOutputSize(
//         d_compressed_data,
//         compressed_size,
//         &decompressed_size,
//         0);
//     assert(status == nvcompSuccess);

//     // 分配解压缩后的内存
//     int* d_output_data;
//     CHECK_CUDA(cudaMalloc(&d_output_data, decompressed_size));

//     // 获取解压缩需要的临时缓冲区大小
//     size_t temp_bytes_decomp;
//     status = nvcompCascadedDecompressGetTempSize(
//         d_compressed_data,
//         compressed_size,
//         &temp_bytes_decomp);
//     assert(status == nvcompSuccess);

//     // 分配解压缩临时缓冲区
//     void* d_temp_decomp;
//     CHECK_CUDA(cudaMalloc(&d_temp_decomp, temp_bytes_decomp));

//     // 执行解压缩
//     status = nvcompCascadedDecompressAsync(
//         d_compressed_data,
//         compressed_size,
//         d_temp_decomp,
//         temp_bytes_decomp,
//         d_output_data,
//         decompressed_size,
//         0);
//     assert(status == nvcompSuccess);
//     CHECK_CUDA(cudaStreamSynchronize(0));

//     // 验证解压后的数据
//     std::vector<int> output_data(data_size);
//     CHECK_CUDA(cudaMemcpy(output_data.data(), d_output_data, data_size * sizeof(int), cudaMemcpyDeviceToHost));

//     bool is_correct = (input_data == output_data);
//     std::cout << "Decompression " << (is_correct ? "successful!" : "failed!") << std::endl;

//     // 清理内存
//     cudaFree(d_input_data);
//     cudaFree(d_compressed_data);
//     cudaFree(d_output_data);
//     cudaFree(d_temp);
//     cudaFree(d_temp_decomp);

//     return is_correct ? 0 : 1;
// }
