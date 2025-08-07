// #include <stdio.h>
// #include <stdlib.h>
// #include <unistd.h>
// #include <math.h>
// #include <cuda_runtime.h>
// #include <stdint.h>
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <random>
// #include <fstream>
// #include "data/dataset_utils.hpp"
// #include "GDFCompressor.cuh"
// #include "GDFDecompressor.cuh"
// #include <thread>
// namespace fs = std::filesystem;

// std::string title=""; 
// #define NUM_STREAMS 16 // 使用16个CUDA Stream实现流水线
// #define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
// {
//     if (code != cudaSuccess)
//     {
//         fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
//         if (abort) exit(code);
//     }
// }

// cudaEvent_t global_start_event, global_end_event;//全局开始和结束事件

// // 高精度计时器
// class Timer {
// private:
//     std::chrono::time_point<std::chrono::high_resolution_clock> start;
// public:
//     void Start() { start = std::chrono::high_resolution_clock::now(); }
//     float Elapsed() {
//         auto end = std::chrono::high_resolution_clock::now();
//         return std::chrono::duration<float, std::milli>(end - start).count();
//     }
// };

// //内存池
// class MemoryPool
// {
// public:
//     MemoryPool(size_t i_poolSize, size_t chunkSize) : poolSize(i_poolSize), chunkSize(chunkSize)
//     {
//         cudaError_t err = cudaMalloc(&pool, poolSize);
//         if (err != cudaSuccess) {
//             std::cerr << "CUDA malloc failed: " << cudaGetErrorString(err) << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         freeList = (void**)malloc(poolSize / chunkSize * sizeof(void*)); // 用于管理空闲块
//         if (!freeList) {
//             std::cerr << "Failed to allocate freeList!" << std::endl;
//             exit(EXIT_FAILURE);
//         }

//         freeBlockCount = poolSize / chunkSize;
        
//         // 初始化空闲块列表
//         for (size_t i = 0; i < freeBlockCount; ++i)
//         {
//             freeList[i] = (void*)((char*)pool + i * chunkSize);
//         }
        
//         // 初始化块状态跟踪
//         usedBlocks = (bool*)calloc(poolSize / chunkSize, sizeof(bool));
//         if (!usedBlocks) {
//             std::cerr << "Failed to allocate usedBlocks!" << std::endl;
//             exit(EXIT_FAILURE);
//         }
//     }
    
//     void reset() {
//         // 确保所有 GPU 操作已完成
//         cudaDeviceSynchronize();
//         // 重置内存池但要检查所有块是否已归还
//         if (freeBlockCount != poolSize / chunkSize) {
//             std::cerr << "Warning: Memory pool reset while " 
//                       << (poolSize / chunkSize - freeBlockCount) 
//                       << " blocks still in use!" << std::endl;
//         }
        
//         freeBlockCount = poolSize / chunkSize;
//         for (size_t i = 0; i < freeBlockCount; ++i) {
//             freeList[i] = (char*)pool + i * chunkSize;
//             usedBlocks[i] = false;
//         }
//     }
    
    
//     // 从内存池中获取内存
//     void* allocate()
//     {
//         if (freeBlockCount == 0)
//         {
//             std::cerr << "Error: No available memory blocks in the pool!" << std::endl;
//             std::cerr << "Error: No available memory blocks in the pool!" << std::endl;
//             return nullptr;
//         }
        
        
//         void* block = freeList[--freeBlockCount]; // 获取一个空闲块
        
//         // 标记为已使用
//         size_t blockIndex = ((char*)block - (char*)pool) / chunkSize;
//         usedBlocks[blockIndex] = true;
        
        
//         // 标记为已使用
//         size_t blockIndex = ((char*)block - (char*)pool) / chunkSize;
//         usedBlocks[blockIndex] = true;
        
//         return block;
//     }

//     // 归还内存块
//     void deallocate(void* block)
//     {
//         if (!block) return;
        
//         // 确认地址在合法范围内
//         if (block < pool || block >= (void*)((char*)pool + poolSize)) {
//             std::cerr << "Error: Trying to deallocate invalid memory address!" << std::endl;
//         if (!block) return;
        
//         // 确认地址在合法范围内
//         if (block < pool || block >= (void*)((char*)pool + poolSize)) {
//             std::cerr << "Error: Trying to deallocate invalid memory address!" << std::endl;
//             return;
//         }
        
//         // 计算块索引并检查是否已经释放
//         size_t blockIndex = ((char*)block - (char*)pool) / chunkSize;
//         if (!usedBlocks[blockIndex]) {
//             std::cerr << "Error: Double free detected for block " << blockIndex << std::endl;
//             return;
//         }
        
//         // 标记为空闲并添加到可用列表
//         usedBlocks[blockIndex] = false;
//         freeList[freeBlockCount++] = block;
//     }

//     // 检查内存池状态
//     void printStatus() {
//         std::cout << "Memory Pool Status: " 
//                   << freeBlockCount << "/" << (poolSize / chunkSize) 
//                   << " blocks free (" 
//                   << (100.0 * freeBlockCount / (poolSize / chunkSize)) 
//                   << "%)" << std::endl;
        
//         // 计算块索引并检查是否已经释放
//         size_t blockIndex = ((char*)block - (char*)pool) / chunkSize;
//         if (!usedBlocks[blockIndex]) {
//             std::cerr << "Error: Double free detected for block " << blockIndex << std::endl;
//             return;
//         }
        
//         // 标记为空闲并添加到可用列表
//         usedBlocks[blockIndex] = false;
//         freeList[freeBlockCount++] = block;
//     }

//     // 检查内存池状态
//     void printStatus() {
//         std::cout << "Memory Pool Status: " 
//                   << freeBlockCount << "/" << (poolSize / chunkSize) 
//                   << " blocks free (" 
//                   << (100.0 * freeBlockCount / (poolSize / chunkSize)) 
//                   << "%)" << std::endl;
//     }

//     ~MemoryPool()
//     {
//         if (pool) cudaFree(pool);
//         if (freeList) free(freeList);
//         if (usedBlocks) free(usedBlocks);
//         if (pool) cudaFree(pool);
//         if (freeList) free(freeList);
//         if (usedBlocks) free(usedBlocks);
//     }

// private:
//     void* pool{}; // 内存池的起始地址
//     void** freeList; // 空闲块链表
//     bool* usedBlocks; // 跟踪每个块的使用情况
//     bool* usedBlocks; // 跟踪每个块的使用情况
//     size_t poolSize; // 内存池的大小
//     size_t chunkSize; // 每个内存块的大小
//     size_t freeBlockCount; // 当前空闲块的数量
// };

// // 流水线每个阶段的计时事件
// struct StreamTiming {
//     cudaEvent_t start_event, h2d_event, comp_event, d2h_event, end_event;
//     float begin_time = 0;
//     float h2d_time = 0;
//     float comp_time = 0;
//     float d2h_time = 0;
//     float total_time = 0;
//     float end_time = 0;
// };

// // 流水线总体性能分析
// struct PipelineAnalysis {
//     float total_h2d = 0;    //总年
//     float total_comp = 0;   //
//     float total_d2h = 0;    //
//     float end_time = 0;
//     float sequential_time = 0;
//     float comp_level = 0;
//     float comp_level = 0;
//     float speedup = 0;
//     float avg_h2d = 0;    // 平均H2D时间
//     float avg_comp = 0;   // 平均计算时间
//     float avg_d2h = 0;    // 平均D2H时间
//     float total_size = 0; // 数据总大小
//     size_t chunk_size = 0; // 记录块大小
// };

// // 流水线操作类，负责管理单个数据块的处理

// class PipelineOperator {
// private:
//     int index;
//     cudaStream_t stream;
//     StreamTiming compression_timing;   // For compression timing
//     StreamTiming decompression_timing; // For decompression timing
    
// public:
//     cudaEvent_t get_comp_finished_event() const {
//         return compression_timing.comp_event;
//     }
//     cudaEvent_t get_decomp_finished_event() const {
//         return decompression_timing.comp_event;
//     }
//     PipelineOperator() {
//         // 创建流和事件
//         cudaCheckError(cudaStreamCreate(&stream));
        
//         // Compression events
//         cudaCheckError(cudaEventCreate(&compression_timing.start_event));
//         cudaCheckError(cudaEventCreate(&compression_timing.h2d_event));
//         cudaCheckError(cudaEventCreate(&compression_timing.comp_event));
//         cudaCheckError(cudaEventCreate(&compression_timing.d2h_event));
//         cudaCheckError(cudaEventCreate(&compression_timing.end_event));
        
//         // Decompression events
//         cudaCheckError(cudaEventCreate(&decompression_timing.start_event));
//         cudaCheckError(cudaEventCreate(&decompression_timing.h2d_event));
//         cudaCheckError(cudaEventCreate(&decompression_timing.comp_event)); // Will be used for decompression operation
//         cudaCheckError(cudaEventCreate(&decompression_timing.d2h_event));
//         cudaCheckError(cudaEventCreate(&decompression_timing.end_event));
//     }
    
//     ~PipelineOperator() {
//         // 销毁流和事件
//         cudaStreamDestroy(stream);
        
//         // Destroy compression events
//         cudaEventDestroy(compression_timing.start_event);
//         cudaEventDestroy(compression_timing.h2d_event);
//         cudaEventDestroy(compression_timing.comp_event);
//         cudaEventDestroy(compression_timing.d2h_event);
//         cudaEventDestroy(compression_timing.end_event);
        
//         // Destroy decompression events
//         cudaEventDestroy(decompression_timing.start_event);
//         cudaEventDestroy(decompression_timing.h2d_event);
//         cudaEventDestroy(decompression_timing.comp_event);
//         cudaEventDestroy(decompression_timing.d2h_event);
//         cudaEventDestroy(decompression_timing.end_event);
//     }
    
//     void set_index(int idx) { index = idx; }
//     int get_index() const { return index; }
//     cudaStream_t get_stream() const { return stream; }
    
//     // 执行一次异步操作（H2D -> Compression -> D2H） , 存在问题cmpSize会不更新，显存问题再说
//     /**
//      *
//      * @param h_data HOST 原始数据
//      * @param h_cmpBytes HOST 压缩后的数据
//      * @param cmpSize 废弃参数
//      * @param d_data DEVICE 原始数据
//      * @param d_cmpBytes DEVICE 压缩后的数据
//      * @param chunkEle 数据数量
//      * @param offset 数据缓冲区偏移
//      */
//     void process_chunk(
//         double *h_data, unsigned char *h_cmpBytes, unsigned int *cmpSize,
//         double *d_data, unsigned char *d_cmpBytes,
//         size_t chunkEle, size_t offset,
//         cudaEvent_t prev_comp_event = nullptr,  // 前一个流的压缩完成事件
//         size_t* global_cmp_offset = nullptr     // 全局压缩偏移量指针
//     ) {
//         // 记录开始事件
//         cudaCheckError(cudaEventRecord(compression_timing.start_event, stream));

//         // H2D - 这个可以立即开始，不需要等待
//         cudaCheckError(cudaMemcpyAsync(d_data, h_data + offset, chunkEle * sizeof(double),
//             cudaMemcpyHostToDevice, stream));
//         cudaCheckError(cudaEventRecord(compression_timing.h2d_event, stream));
        
//         // 如果有前一个流，等待其压缩完成
//         if (prev_comp_event != nullptr) {
//             cudaCheckError(cudaStreamWaitEvent(stream, prev_comp_event, 0));
//         }
        
//         // 执行压缩计算
//         GDFCompressor::GDFC_compress_stream(d_data, d_cmpBytes, cmpSize, chunkEle, stream);
//         cudaCheckError(cudaEventRecord(compression_timing.comp_event, stream));
        
        
//         // 计算当前流的D2H偏移量（需要同步获取压缩大小）
//         cudaStreamSynchronize(stream);  // 确保压缩完成，cmpSize已更新
//         size_t current_compressed_size = (*cmpSize + 7) / 8;
        
//         size_t d2h_offset = 0;
//         if (global_cmp_offset != nullptr) {
//             d2h_offset = *global_cmp_offset;
//             *global_cmp_offset += current_compressed_size;  // 更新全局偏移
//         }
        
//         std::cout << "Stream " << index << " - CompSize: " << current_compressed_size 
//                   << ", D2H offset: " << d2h_offset << std::endl;
//         // D2H - 使用正确的偏移量

//         cudaCheckError(cudaMemcpyAsync(
//             h_cmpBytes + d2h_offset,  // 使用计算得到的正确偏移
//             d_cmpBytes, 
//             current_compressed_size,  // 使用实际的压缩大小
//             cudaMemcpyDeviceToHost,
//             stream
//         ));
//         cudaCheckError(cudaEventRecord(compression_timing.d2h_event, stream));
        
//         cudaCheckError(cudaEventRecord(compression_timing.end_event, stream));
//         cudaStreamSynchronize(stream);  // 确保压缩完成，cmpSize已更新
//     }
//     // void process_chunk(
//     //     double* h_data, unsigned char* h_cmpBytes, size_t& cmpSize,
//     //     double* d_data, unsigned char* d_cmpBytes,
//     //     size_t chunkEle, size_t offset) {
        
//     //     // 记录开始事件
//     //     cudaCheckError(cudaEventRecord(compression_timing.start_event, stream));
        
//     //     // 主机到设备的拷贝 (H2D)
//     //     cudaCheckError(cudaMemcpyAsync(d_data, h_data + offset, chunkEle * sizeof(double), 
//     //                                   cudaMemcpyHostToDevice, stream));
//     //     cudaCheckError(cudaEventRecord(compression_timing.h2d_event, stream));
//     //     // cudaStreamSynchronize(stream);

//     //     // 执行压缩计算
//     //     size_t chunkCmpSize = 0;
//     //     GDFCompressor::GDFC_compress(d_data, d_cmpBytes, chunkEle, &chunkCmpSize, stream);

//     //     cudaCheckError(cudaEventRecord(compression_timing.comp_event, stream));
//     //     // cudaStreamSynchronize(stream);
        
//     //     // 设备到主机的拷贝 (D2H) - 压缩结果
//     //     cudaCheckError(cudaMemcpyAsync(h_cmpBytes + cmpSize, d_cmpBytes, 
//     //                                   chunkEle * sizeof(double),//chunkCmpSize * sizeof(unsigned char), 
//     //                                   cudaMemcpyDeviceToHost, stream));

             
//     //     cudaCheckError(cudaEventRecord(compression_timing.d2h_event, stream));

//     //     // 记录结束事件
//     //     cudaCheckError(cudaEventRecord(compression_timing.end_event, stream));

//     //     cudaStreamSynchronize(stream);
//     //     // 更新总压缩大小并返回当前块的压缩大小
//     //     cmpSize += chunkCmpSize;
//     // }

//     // 新增：执行解压操作 (H2D -> Decompression -> D2H)
//     void decompress_chunk(
//         double* h_decData, unsigned char* h_cmpBytes, 
//         double* d_decData, unsigned char* d_cmpBytes,
//         size_t chunkEle, size_t cmpSize, size_t cmpOffset, size_t decOffset) {
        
//         // 确保参数有效
//         if (chunkEle == 0 || cmpSize == 0) {
//             std::cerr << "Error: Invalid chunk parameters for decompression. chunkEle=" 
//                     << chunkEle << ", cmpSize=" << cmpSize << std::endl;
//             return;
//         }
        
//         // 记录开始事件
//         cudaCheckError(cudaEventRecord(decompression_timing.start_event, stream));
        
//         // 主机到设备的拷贝 (H2D) - 压缩数据
//         // std::cout << "复制压缩数据 " << cmpSize << " 字节, 从偏移 " << cmpOffset << std::endl;
        
        
//         // 在重要操作之前添加同步点
//         cudaStreamSynchronize(stream);
//         // HTD
//         cudaCheckError(cudaMemcpyAsync(d_cmpBytes, h_cmpBytes + cmpOffset, 
//                                     cmpSize, cudaMemcpyHostToDevice, stream));
//                                     cmpSize, cudaMemcpyHostToDevice, stream));
//         cudaCheckError(cudaEventRecord(decompression_timing.h2d_event, stream));

//         // 添加内存拷贝后的同步点，确保数据已经完全复制到GPU
//         cudaStreamSynchronize(stream);
        
//         // 执行解压计算
//         // std::cout << "解压数据块: 原始大小=" << chunkEle << "元素, 压缩大小=" << cmpSize << "字节" << std::endl;
//         GDFDecompressor GDFC;
        
//         // 检查解压参数有效性
//         if (d_decData == nullptr || d_cmpBytes == nullptr) {
//             std::cerr << "错误: 无效的设备内存指针用于解压" << std::endl;
//             return;
//         }
        
//         // 使用try-catch捕获可能的解压错误
//         try {
//             GDFC.GDFC_decompress(d_decData, d_cmpBytes, chunkEle, cmpSize, stream);
//         }
//         catch (const std::exception& e) {
//             std::cerr << "解压发生异常: " << e.what() << std::endl;
//             return;
//         }
        
//         cudaCheckError(cudaEventRecord(decompression_timing.comp_event, stream));
        
//         // 解压后同步，确保解压已完成
//         cudaStreamSynchronize(stream);
        
//         // 设备到主机的拷贝 (D2H) - 解压结果
//         // std::cout << "复制解压数据 " << chunkEle << " 元素, 到偏移 " << decOffset << std::endl;
//         cudaError_t err = cudaMemcpyAsync(h_decData + decOffset, d_decData, 
//             chunkEle * sizeof(double),
//             cudaMemcpyDeviceToHost, stream);
//         if (err != cudaSuccess) {
//             std::cerr << "CUDA Error during D2H copy: " << cudaGetErrorString(err) 
//             << "\nOffset: " << decOffset << ", Elements: " << chunkEle 
//             << ", Device ptr: " << d_decData << ", Host ptr: " << (h_decData + decOffset) << std::endl;
//             return;
//         }
//         //decData指针传输错误
//         cudaCheckError(cudaEventRecord(decompression_timing.d2h_event, stream));
        
//         // 记录结束事件
//         cudaCheckError(cudaEventRecord(decompression_timing.end_event, stream));
        
//         // 最终同步，确保数据已经回到主机
//         cudaStreamSynchronize(stream);
//     }
    

//     // 计算并返回压缩阶段的时间
//     StreamTiming calculate_compression_timing() {
//         float tmp;
//         // 计算在全局的起始时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, global_start_event, compression_timing.start_event));
//         compression_timing.begin_time = tmp;

//         // 计算H2D时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, compression_timing.start_event, compression_timing.h2d_event));
//         compression_timing.h2d_time = tmp;
        
//         // 计算压缩时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, compression_timing.h2d_event, compression_timing.comp_event));
//         compression_timing.comp_time = tmp;
        
//         // 计算D2H时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, compression_timing.comp_event, compression_timing.d2h_event));
//         compression_timing.d2h_time = tmp;
        
//         // 计算总时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, compression_timing.start_event, compression_timing.end_event));
//         compression_timing.total_time = tmp;
        
//         // 计算在全局中结束时间
//         compression_timing.end_time = compression_timing.total_time + compression_timing.begin_time;
//         return compression_timing;
//     }

//     // 计算并返回解压阶段的时间
//     StreamTiming calculate_decompression_timing() {
//         float tmp;
//         // 计算在全局的起始时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, global_start_event, decompression_timing.start_event));
//         decompression_timing.begin_time = tmp;

//         // 计算H2D时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, decompression_timing.start_event, decompression_timing.h2d_event));
//         decompression_timing.h2d_time = tmp;
        
//         // 计算解压时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, decompression_timing.h2d_event, decompression_timing.comp_event));
//         decompression_timing.comp_time = tmp;
        
//         // 计算D2H时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, decompression_timing.comp_event, decompression_timing.d2h_event));
//         decompression_timing.d2h_time = tmp;
        
//         // 计算总时间
//         cudaCheckError(cudaEventElapsedTime(&tmp, decompression_timing.start_event, decompression_timing.end_event));
//         decompression_timing.total_time = tmp;
        
//         // 计算在全局中结束时间
//         decompression_timing.end_time = decompression_timing.total_time + decompression_timing.begin_time;
//         return decompression_timing;
//     }
// };

// // 定义一个新的结构体来保存每个数据块的压缩信息
// struct CompressionInfo {
//     size_t original_offset;  // 原始数据的偏移
//     size_t compressed_offset; // 压缩数据的偏移
//     size_t original_size;    // 原始数据的大小（以元素计）
//     size_t compressed_size;  // 压缩数据的大小（以字节计）
// };

// // 添加用于解压缩验证的新结构体
// struct PipelineVerification {
//     float compression_time = 0;
//     float decompression_time = 0;
//     float compression_ratio = 0;
//     bool all_blocks_verified = true;
//     int total_blocks = 0;
//     int verified_blocks = 0;
//     double max_error = 0.0;
//     double average_error = 0.0;
// };

// // 验证数据块解压正确性的函数
// bool verify_chunk(
//     const double* original,
//     const double* decompressed,
//     size_t size,
//     double& max_error,
//     double& avg_error) {
    
//     if (!original || !decompressed) {
//         std::cerr << "验证错误: 无效的数据指针" << std::endl;
//         return false;
//     }
    
// bool verify_chunk(
//     const double* original,
//     const double* decompressed,
//     size_t size,
//     double& max_error,
//     double& avg_error) {
    
//     if (!original || !decompressed) {
//         std::cerr << "验证错误: 无效的数据指针" << std::endl;
//         return false;
//     }
    
//     max_error = 0.0;
//     double total_error = 0.0;
//     bool is_valid = true;
    
//     // 为了调试，总是打印前几个元素的值
//     const int debug_count = 5;
//     for (int i = 0; i < std::min(debug_count, (int)size); i++) {
//         std::cout << "original[" << i << "]:" << original[i] 
//                   << " , decompressed[" << i << "]: " << decompressed[i] << std::endl;
        
//         if (std::isnan(decompressed[i]) || std::isinf(decompressed[i])) {
//             std::cout << "  警告: 元素 " << i << " 包含无效值" << std::endl;
//             is_valid = false;
//         }
//     }
    
//     // 完整验证
//     for (size_t i = 0; i < size; i++) {
//         double err = std::abs(original[i] - decompressed[i]);
//         max_error = std::max(max_error, err);
//         total_error += err;
        
//         // 检查错误是否超过阈值
//         const double error_threshold = 1e-6;
//         if (err > error_threshold) {
//             // 如果错误超过阈值且尚未打印过这个元素，则打印详细信息
//             if (i >= debug_count && is_valid && i % 1000000 == 0) { // 只打印部分以避免大量输出
//                 std::cout << "错误超过阈值: original[" << i << "]=" << original[i] 
//                           << ", decompressed[" << i << "]=" << decompressed[i] 
//                           << ", err=" << err << std::endl;
//             }
//             is_valid = false;
//         }
//     }
    
//     avg_error = total_error / size;
//     return is_valid;
//     bool is_valid = true;
    
//     // 为了调试，总是打印前几个元素的值
//     const int debug_count = 5;
//     for (int i = 0; i < std::min(debug_count, (int)size); i++) {
//         std::cout << "original[" << i << "]:" << original[i] 
//                   << " , decompressed[" << i << "]: " << decompressed[i] << std::endl;
        
//         if (std::isnan(decompressed[i]) || std::isinf(decompressed[i])) {
//             std::cout << "  警告: 元素 " << i << " 包含无效值" << std::endl;
//             is_valid = false;
//         }
//     }
    
//     // 完整验证
//     for (size_t i = 0; i < size; i++) {
//         double err = std::abs(original[i] - decompressed[i]);
//         max_error = std::max(max_error, err);
//         total_error += err;
        
//         // 检查错误是否超过阈值
//         const double error_threshold = 1e-6;
//         if (err > error_threshold) {
//             // 如果错误超过阈值且尚未打印过这个元素，则打印详细信息
//             if (i >= debug_count && is_valid && i % 1000000 == 0) { // 只打印部分以避免大量输出
//                 std::cout << "错误超过阈值: original[" << i << "]=" << original[i] 
//                           << ", decompressed[" << i << "]=" << decompressed[i] 
//                           << ", err=" << err << std::endl;
//             }
//             is_valid = false;
//         }
//     }
    
//     avg_error = total_error / size;
//     return is_valid;
// }

// // 分析流水线性能
// PipelineAnalysis analyze_pipeline(const std::vector<StreamTiming>& timings, size_t chunkSize, float comp_level,bool print_results = true) {
// PipelineAnalysis analyze_pipeline(const std::vector<StreamTiming>& timings, size_t chunkSize, float comp_level,bool print_results = true) {
//     PipelineAnalysis analysis;
//     analysis.chunk_size = chunkSize;
//     analysis.comp_level =comp_level;
//     analysis.comp_level =comp_level;
//     if (timings.empty()) {
//         return analysis;
//     }
    
//     // 计算各阶段总时间和最大总时间
//     for (const auto& timing : timings) {
//         analysis.total_h2d += timing.h2d_time;
//         analysis.total_comp += timing.comp_time;
//         analysis.total_d2h += timing.d2h_time;
//         analysis.end_time = std::max(analysis.end_time, timing.end_time);
//     }
    
//     // 计算平均时间
//     analysis.avg_h2d = analysis.total_h2d / timings.size();
//     analysis.avg_comp = analysis.total_comp / timings.size();
//     analysis.avg_d2h = analysis.total_d2h / timings.size();
    
//     // 计算顺序执行理论时间和加速比
//     analysis.sequential_time = analysis.total_h2d + analysis.total_comp + analysis.total_d2h;
//     analysis.speedup = analysis.sequential_time / analysis.end_time;
    
//     if (print_results) {
//         // 打印流水线分析结果
//         printf("\n===== 流水线执行分析 (块大小: %zu 元素) =====\n", chunkSize);
//         printf("各阶段总时间:\n");
//         printf("- H2D总时间: %.2f ms (平均: %.2f ms/块)\n", analysis.total_h2d, analysis.avg_h2d);
//         printf("- 压缩计算总时间: %.2f ms (平均: %.2f ms/块)\n", analysis.total_comp, analysis.avg_comp);
//         printf("- D2H总时间: %.2f ms (平均: %.2f ms/块)\n", analysis.total_d2h, analysis.avg_d2h);
//         printf("- 顺序执行理论时间: %.2f ms\n", analysis.sequential_time);
//         printf("- 实际并行执行时间: %.2f ms\n", analysis.end_time);
//         printf("- 流水线加速比: %.2fx\n", analysis.speedup);
//     }
    
//     return analysis;
// }

// // 可视化流水线执行时间线
// void visualize_timeline(const std::vector<StreamTiming>& timings) {
//     printf("\n===== 流水线时间线可视化 =====\n");
//     printf("图例: |H2D|->|Comp|->|D2H|\n\n");
    
//     // 查找最长的执行时间
//     float max_time = 0;
//     for (const auto& timing : timings) {
//         max_time = std::max(max_time, timing.total_time);
//     }
    
//     const int TIME_SCALE = 2; // 每毫秒显示的字符数
//     float h2d_ending=0;
//     // 为每个流/操作显示时间线
//     for (size_t i = 0; i < timings.size(); i++) {
//         const auto& timing = timings[i];
        
//         // 计算各阶段的开始和结束位置
//         float h2d_start = timing.begin_time;
//         float h2d_end = timing.h2d_time;
//         float comp_start = h2d_end;
//         float comp_end = comp_start + timing.comp_time;
//         float d2h_start = comp_end;
//         float d2h_end = d2h_start + timing.d2h_time;
        
//         // 打印操作ID和流ID
//         printf("操作%2zu(流%zu): ", i, i % NUM_STREAMS);
        
//         //绘制全局前偏移
//         int beg_len = std::max(1, (int)(h2d_start * TIME_SCALE));
//         for (int j = 0; j < beg_len; j+=2) printf(" ");
//         // 绘制H2D
//         printf("|H2D|");
//         int h2d_len = std::max(1, (int)(timing.h2d_time * TIME_SCALE));
//         for (int j = 0; j < h2d_len; j+=2) printf("=");
        
//         // 绘制Comp
//         printf("|Comp|");
//         int comp_len = std::max(1, (int)(timing.comp_time * TIME_SCALE));
//         for (int j = 0; j < comp_len; j+=2) printf("~");
        
//         // 绘制D2H
//         printf("|D2H|");
//         int d2h_len = std::max(1, (int)(timing.d2h_time * TIME_SCALE));
//         for (int j = 0; j < d2h_len; j+=2) printf("-");
        
//         printf(" (%.2f ms)(cont :%0.2f ms)\n", timing.total_time,h2d_start-h2d_ending);
//         h2d_ending=h2d_end+h2d_start;
//     }
// }

// // 生成随机数据的函数
// std::vector<double> generate_test_data(size_t nbEle, int pattern_type = 0) {
//     std::vector<double> data(nbEle);
    
//     std::random_device rd;
//     std::mt19937 gen(rd());
    
//     switch(pattern_type) {
//         case 0: { // 随机数据
//             std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
//             for (size_t i = 0; i < nbEle; ++i) {
//                 data[i] = dist(gen);
//             }
//             break;
//         }
//         case 1: { // 线性增长数据
//             for (size_t i = 0; i < nbEle; ++i) {
//                 data[i] = static_cast<double>(i) * 0.01;
//             }
//             break;
//         }
//         case 2: { // 正弦波数据
//             for (size_t i = 0; i < nbEle; ++i) {
//                 data[i] = 1000.0 * sin(0.01 * i);
//             }
//             break;
//         }
//         case 3: { // 多步阶数据
//             int step_size = nbEle / 10;
//             for (size_t i = 0; i < nbEle; ++i) {
//                 data[i] = static_cast<double>((i / step_size) * 100);
//             }
//             break;
//         }
//         default: {
//             std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
//             for (size_t i = 0; i < nbEle; ++i) {
//                 data[i] = dist(gen);
//             }
//         }
//     }
    
//     return data;
// }

// // 数据预处理函数
// struct ProcessedData {
//     double* oriData;
//     unsigned char* cmpBytes;
//     unsigned int *cmpSize;
//     size_t nbEle;
// };

// // 准备数据函数，支持文件和生成数据两种模式
// ProcessedData prepare_data(const std::string &source_path = "", size_t generate_size = 0, int pattern_type = 0) {
//     ProcessedData result;
//     std::vector<double> data;

//     // 决定数据来源
//     if (generate_size > 0) {
//         // 生成指定大小的数据
//         printf("生成 %zu 个元素的测试数据 (模式: %d)\n", generate_size, pattern_type);
//         data = generate_test_data(generate_size, pattern_type);
//         result.nbEle = generate_size;
//     } else if (!source_path.empty()) {
//         // 从文件读取数据
//         printf("从文件加载数据: %s\n", source_path.c_str());
//         data = read_data(source_path);
//         result.nbEle = data.size();
//     } else {
//         printf("错误: 未指定数据源\n");
//         result.nbEle = 0;
//         result.oriData = nullptr;
//         result.cmpBytes = nullptr;
//         return result;
//     }

//     // 分配固定内存
//     cudaCheckError(cudaHostAlloc(&result.oriData, result.nbEle * sizeof(double), cudaHostAllocDefault));
//     cudaCheckError(cudaHostAlloc((void**)&result.cmpBytes, result.nbEle * sizeof(double) * 2, cudaHostAllocDefault));
//     cudaCheckError(cudaHostAlloc((void**)&result.cmpSize, sizeof(unsigned int), cudaHostAllocDefault));

//     // 将数据拷贝到固定内存
//     #pragma omp parallel for
//     for (size_t i = 0; i < result.nbEle; ++i) {
//         result.oriData[i] = data[i];
//     }

//     return result;
// }

// // 清理资源函数
// void cleanup_data(ProcessedData& data) {
//     if (data.oriData != nullptr) {
//         cudaFreeHost(data.oriData);
//         data.oriData = nullptr;
//     }
    
//     if (data.cmpBytes != nullptr) {
//         cudaFreeHost(data.cmpBytes);
//         data.cmpBytes = nullptr;
//     }
// }

// // 设置GPU内存池函数
// size_t setup_gpu_memory_pool(size_t nbEle, size_t &chunkSize) {
//     // 检查GPU可用内存
//     size_t freeMem, totalMem;
//     cudaMemGetInfo(&freeMem, &totalMem);
//     std::cout << "GPU可用内存: " << freeMem / (1024 * 1024) << " MB / 总内存: "
//             << totalMem / (1024 * 1024) << " MB" << std::endl;

//     // 设置内存池大小
//     size_t poolSize = freeMem * 0.4;
//     poolSize = (poolSize + 1024 * 2 * sizeof(double) - 1) & ~(1024 * 2 * sizeof(double) - 1); // 向上对齐
//     chunkSize = poolSize * 0.5 * 0.5 / sizeof(double);

//     // 设置分块大小为2的幂，以优化对齐
//     int temp = 2;
//     while (temp * 2 < chunkSize) {
//         temp *= 2;
//     }
//     chunkSize = temp;//调整
    
//     return poolSize;
// }

// // 执行压缩流程函数
// PipelineAnalysis execute_pipeline(ProcessedData& data, size_t chunkSize, size_t poolSize, bool visualize = false) {
//     // 创建时间线记录事件
//     cudaEventCreate(&global_start_event);
//     cudaEventCreate(&global_end_event);

//     // 创建内存池
//     MemoryPool ori_data_pool(poolSize, chunkSize * sizeof(double)); // 原始数据显存池
//     MemoryPool cmp_bytes_pool(poolSize, chunkSize * sizeof(double)); // 压缩结果显存池

//     std::cout << "数据总大小: " << data.nbEle * sizeof(double) / (1024 * 1024) << " MB" << std::endl;
//     std::cout << "块大小: " << chunkSize * sizeof(double) / (1024 * 1024) << " MB" << std::endl;

//     // 创建流水线操作器
//     std::vector<PipelineOperator> operators(NUM_STREAMS);

//     // 创建流水线分析对象
//     std::vector<StreamTiming> timings;

//     // 启动计时器
//     Timer timer;
//     timer.Start();

//     // 执行流水线处理
//     size_t processedEle = 0;
//     int chunkIndex = 0;
//     unsigned int* cmpSize = data.cmpSize;
//     *cmpSize = 0;
//     unsigned int* locCmpSize;
//     cudaCheckError(cudaHostAlloc((void**)&locCmpSize, sizeof(unsigned int) * NUM_STREAMS, cudaHostAllocDefault));
//     size_t global_cmp_offset = 0;  // 全局压缩偏移量
//     while (processedEle < data.nbEle) {
//         if (processedEle == 0) {
//             cudaEventRecord(global_start_event); //全局时间记录
//         }
//         // 计算当前块的大小
//         size_t chunkEle = (data.nbEle - processedEle) > chunkSize ? chunkSize : (data.nbEle - processedEle);
//         if (chunkEle == 0) break;
        
//         // 选择当前流
//         int streamIdx = chunkIndex % NUM_STREAMS;
        
//         // 分配GPU内存
//         double* d_oriData = (double*)ori_data_pool.allocate();
//         unsigned char* d_cmpBytes = (unsigned char*)cmp_bytes_pool.allocate();
        
//         if (!d_oriData || !d_cmpBytes) {
//             std::cerr << "内存池分配失败!" << std::endl;
//             break;
//         }
        
//         std::cout << "处理数据块 " << chunkIndex << " (流 " << streamIdx << "): " 
//                   << processedEle << " 到 " << (processedEle + chunkEle) 
//                   << " 块大小: " << chunkEle << std::endl;
//         // 设置当前块的索引
//         operators[streamIdx].set_index(chunkIndex);
        
//         // 处理当前数据块（异步操作）
//         // operators[streamIdx].process_chunk(
//         //     data.oriData, data.cmpBytes, cmpSize,
//         //     d_oriData, d_cmpBytes,
//         //     chunkEle, processedEle
//         // );

//         cudaEvent_t prev_comp_event = nullptr;
//         if (chunkIndex > 0) {
//             int prevStreamIdx = (chunkIndex - 1) % NUM_STREAMS;
//             prev_comp_event = operators[prevStreamIdx].get_comp_finished_event();
//         }

//         operators[streamIdx].process_chunk(
//             data.oriData, data.cmpBytes, locCmpSize + streamIdx,
//             d_oriData, d_cmpBytes,
//             chunkEle, processedEle,
//             prev_comp_event,     // 传入前一个流的压缩完成事件
//             &global_cmp_offset   // 传入全局偏移量
//         );
        
//         cudaCheckError(cudaStreamSynchronize(operators[streamIdx].get_stream()));
//         //释放显存
//         ori_data_pool.deallocate(d_oriData);
//         cmp_bytes_pool.deallocate(d_cmpBytes);

//         // 处理下一个数据块
//         processedEle += chunkEle;
//         chunkIndex++;
//         *cmpSize = global_cmp_offset;  // 更新总压缩大小
//         // 若已经处理了NUM_STREAMS个数据块，则需要等待最早的流完成
//         if (chunkIndex >= NUM_STREAMS && chunkIndex % NUM_STREAMS == 0) {
//             // 等待每个流完成一轮操作
//             for (int i = 0; i < NUM_STREAMS; i++) {
//                 cudaCheckError(cudaStreamSynchronize(operators[i].get_stream()));
//                 // *cmpSize += (*(locCmpSize + i)+7)/8;
//                 // 计算并记录时间
//                 StreamTiming timing = operators[i].calculate_compression_timing();
//                 timings.push_back(timing);
                
//             }
//         }
//     }
    
//     // 等待所有未完成的流操作
//     for (int i = 0; i < NUM_STREAMS; i++) {
//         cudaCheckError(cudaStreamSynchronize(operators[i].get_stream()));
//         // *cmpSize += (*(locCmpSize + i)+7)/8;
//         // 收集最后一批操作的时间数据
//         if ((chunkIndex - 1) % NUM_STREAMS >= i) {
//             StreamTiming timing = operators[i].calculate_compression_timing();
//             timings.push_back(timing);
//         }
//     }
//     cudaEventRecord(global_end_event);

//     // 记录总执行时间
//     float totalTime = timer.Elapsed();
//     float tmp;
//     cudaEventElapsedTime(&tmp, global_start_event, global_end_event);

//     // 分析流水线性能
//     PipelineAnalysis analysis = analyze_pipeline(timings, chunkSize,static_cast<double>(data.nbEle * sizeof(double)) / cmpSize, false);
//     analysis.total_size = data.nbEle * sizeof(double) / (1024*1024);
//     // 可视化时间线
//     if (visualize) {
//         visualize_timeline(timings);
        
//         // 分析流水线性能
//         analyze_pipeline(timings, chunkSize, static_cast<double>(data.nbEle * sizeof(double)) / cmpSize,true);
        
//         // 输出统计信息
//         printf("\n===== 压缩统计 =====\n");
//         printf("压缩大小: %lu 字节\n", *cmpSize);
//         printf("原始大小: %lu 字节\n", data.nbEle * sizeof(double));
//         printf("压缩比: %.2f\n", static_cast<double>(data.nbEle * sizeof(double)) / *cmpSize);
        
//         printf("\n===== 性能统计 =====\n");
//         printf("端到端总时间: %.2f ms (事件记录：%.2f ms)\n", totalTime,tmp);
//         printf("压缩吞吐量: %.2f GB/s\n", 
//             (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0));
//         printf("流水线效率: %.2f%%\n", analysis.speedup * 100.0 / NUM_STREAMS);
//     }
    
//     // 清理事件
//     cudaEventDestroy(global_start_event);
//     cudaEventDestroy(global_end_event);
    
//     return analysis;
// }

// // 为可视化解压缩时间线添加新函数
// void visualize_decompression_timeline(const std::vector<StreamTiming> &timings) {
//     printf("\n===== 解压缩流水线时间线可视化 =====\n");
//     printf("图例: |H2D|->|Decomp|->|D2H|\n\n");

//     // 查找最长的执行时间
//     float max_time = 0;
//     for (const auto &timing: timings) {
//         max_time = std::max(max_time, timing.total_time);
//     }

//     const int TIME_SCALE = 2; // 每毫秒显示的字符数
//     float h2d_ending = 0;

//     // 为每个流/操作显示时间线
//     for (size_t i = 0; i < timings.size(); i++) {
//         const auto &timing = timings[i];

//         // 计算各阶段的开始和结束位置
//         float h2d_start = timing.begin_time;
//         float h2d_end = timing.h2d_time;
//         float decomp_start = h2d_end;
//         float decomp_end = decomp_start + timing.comp_time;
//         float d2h_start = decomp_end;
//         float d2h_end = d2h_start + timing.d2h_time;

//         // 打印操作ID和流ID
//         printf("操作%2zu(流%zu): ", i, i % NUM_STREAMS);

//         //绘制全局前偏移
//         int beg_len = std::max(1, (int) (h2d_start * TIME_SCALE));
//         for (int j = 0; j < beg_len; j += 2) printf(" ");

//         // 绘制H2D
//         printf("|H2D|");
//         int h2d_len = std::max(1, (int) (timing.h2d_time * TIME_SCALE));
//         for (int j = 0; j < h2d_len; j += 2) printf("=");

//         // 绘制Decomp
//         printf("|Decomp|");
//         int decomp_len = std::max(1, (int) (timing.comp_time * TIME_SCALE));
//         for (int j = 0; j < decomp_len; j += 2) printf("~");

//         // 绘制D2H
//         printf("|D2H|");
//         int d2h_len = std::max(1, (int) (timing.d2h_time * TIME_SCALE));
//         for (int j = 0; j < d2h_len; j += 2) printf("-");

//         printf(" (%.2f ms)(cont: %.2f ms)\n", timing.total_time, h2d_start - h2d_ending);
//         h2d_ending = h2d_end + h2d_start;
//     }
// }

// // 修改执行流水线函数，增加解压和验证步骤
// PipelineVerification execute_pipeline_with_verification(ProcessedData& data, size_t chunkSize, size_t poolSize, bool visualize = false) {
//     // 创建时间线记录事件
//     cudaEventCreate(&global_start_event);  
//     cudaEventCreate(&global_end_event);  
    
//     // 创建内存池
//     MemoryPool ori_data_pool(poolSize, chunkSize * sizeof(double)); // 原始数据显存池
//     MemoryPool cmp_bytes_pool(poolSize, chunkSize * sizeof(double)); // 压缩结果显存池
    
//     std::cout << "数据总大小: " << data.nbEle * sizeof(double) / (1024*1024) << " MB" << std::endl;
//     std::cout << "块大小: " << chunkSize * sizeof(double) / (1024*1024) << " MB" << std::endl;
    
//     // 创建流水线操作器
//     std::vector<PipelineOperator> operators(NUM_STREAMS);
    
//     // 创建流水线分析对象
//     std::vector<StreamTiming> comp_timings;
//     std::vector<StreamTiming> decomp_timings;
    
//     // 用于存储每个块的压缩信息
//     std::vector<CompressionInfo> compression_infos;
    
//     // 分配解压缩结果的内存（固定内存以便高效传输）
//     double* decompressed_data;
//     cudaCheckError(cudaHostAlloc(&decompressed_data, data.nbEle * sizeof(double), cudaHostAllocDefault));
//     cudaCheckError(cudaHostAlloc(&decompressed_data, data.nbEle * sizeof(double), cudaHostAllocDefault));
    
//     // 启动计时器
//     Timer timer;
//     timer.Start();
    
//     // 执行压缩流水线处理
//     size_t processedEle = 0;
//     int chunkIndex = 0;
//     size_t global_cmp_offset = 0;  // 全局压缩偏移量
//     unsigned int *cmpSize = data.cmpSize;
//     *cmpSize = 0;
//     unsigned int* locCmpSize;
//     cudaCheckError(cudaHostAlloc((void**)&locCmpSize, sizeof(unsigned int) * NUM_STREAMS, cudaHostAllocDefault));
    
//     // 阶段1: 压缩处理
//     std::cout << "\n===== 阶段1: 执行压缩流水线 =====\n";

//     cudaEventRecord(global_start_event);  //全局时间记录

//     while (processedEle < data.nbEle) {

//     cudaEventRecord(global_start_event);  //全局时间记录

//     while (processedEle < data.nbEle) {
//         // 计算当前块的大小
//         size_t chunkEle = (data.nbEle - processedEle) > chunkSize ? chunkSize : (data.nbEle - processedEle);
//         if (chunkEle == 0) break;
        
//         // 选择当前流
//         int streamIdx = chunkIndex % NUM_STREAMS;
        
//         // 分配GPU内存
//         double* d_oriData = (double*)ori_data_pool.allocate();
//         unsigned char* d_cmpBytes = (unsigned char*)cmp_bytes_pool.allocate();
        
//         if (!d_oriData || !d_cmpBytes) {
//             std::cerr << "内存池分配失败!" << std::endl;
//             break;
//         }

//         // printf("d_oriData = %p\n", d_oriData);
//         // printf("d_cmpBytes = %p\n", d_cmpBytes);

//         // std::cout << "处理数据块 " << chunkIndex << " (流 " << streamIdx << "): " 
//         //           << processedEle << " 到 " << (processedEle + chunkEle) 
//         //           << " 块大小: " << chunkEle << std::endl;
        
//         // 设置当前块的索引
//         operators[streamIdx].set_index(chunkIndex);
        
//         // 确定前一个流的压缩完成事件
//         cudaEvent_t prev_comp_event = nullptr;
//         if (chunkIndex > 0) {
//             int prevStreamIdx = (chunkIndex - 1) % NUM_STREAMS;
//             prev_comp_event = operators[prevStreamIdx].get_comp_finished_event();
//         }
        
        
//         // 处理当前数据块
//         operators[streamIdx].process_chunk(
//             data.oriData, data.cmpBytes, locCmpSize + streamIdx,
//             d_oriData, d_cmpBytes,
//             chunkEle, processedEle,
//             prev_comp_event,     // 传入前一个流的压缩完成事件
//             &global_cmp_offset   // 传入全局偏移量
//         );
        
//         // 记录压缩信息
//         CompressionInfo info;
//         info.original_offset = processedEle;
//         info.original_size = chunkEle;
        
//         // 等待当前流完成以获取准确的压缩大小
//         cudaCheckError(cudaStreamSynchronize(operators[streamIdx].get_stream()));
        
//         info.compressed_size = (*(locCmpSize + streamIdx) + 7) / 8;
//         info.compressed_offset = global_cmp_offset - info.compressed_size; // 之前已经更新了偏移
        
//         compression_infos.push_back(info);

//         // std::cout << "  Compressed " << chunkEle << " elements to " << info.compressed_size 
//         //     << " bytes (ratio: " << static_cast<double>(chunkEle * sizeof(double)) / info.compressed_size << ")" << std::endl;
        
//         cudaStreamSynchronize(operators[streamIdx].get_stream());
//         //释放显存
//         // ori_data_pool.deallocate(d_oriData);
//         // cmp_bytes_pool.deallocate(d_cmpBytes);
        
//         // printf("data.cmpBytes:%d,%d\n",data.cmpBytes[(current_cmp_offset+7)/8],data.cmpBytes[(current_cmp_offset+7)/8+1]);
        
//         // 处理下一个数据块
//         processedEle += chunkEle;
//         chunkIndex++;
//         *cmpSize = global_cmp_offset;  // 更新总压缩大小

//         // 若已经处理了NUM_STREAMS个数据块，则需要等待最早的流完成
//         if (chunkIndex >= NUM_STREAMS && chunkIndex % NUM_STREAMS == 0) {
//             // 等待每个流完成一轮操作
//             for (int i = 0; i < NUM_STREAMS; i++) {
//                 cudaCheckError(cudaStreamSynchronize(operators[i].get_stream()));
                
//                 // 计算并记录时间
//                 StreamTiming timing = operators[i].calculate_compression_timing();
//                 comp_timings.push_back(timing);
//             }
//         }
//     }
    
//     // 等待所有未完成的流操作
//     for (int i = 0; i < NUM_STREAMS; i++) {
//         cudaCheckError(cudaStreamSynchronize(operators[i].get_stream()));
        
//         // 收集最后一批操作的时间数据
//         if ((chunkIndex - 1) % NUM_STREAMS >= i) {
//             StreamTiming timing = operators[i].calculate_compression_timing();
//             comp_timings.push_back(timing);
//         }
//     }
//     cudaEventRecord(global_end_event);
//     cudaEventSynchronize(global_end_event);

//     cudaEventRecord(global_end_event);
//     cudaEventSynchronize(global_end_event);

//     float compression_time = timer.Elapsed();
//     std::cout << "压缩阶段完成，用时: " << compression_time << " ms" << std::endl;
//     std::cout << "压缩大小: " << *cmpSize << " 字节，原始大小: " << data.nbEle * sizeof(double) << " 字节" << std::endl;
//     std::cout << "压缩比: " << static_cast<double>(data.nbEle * sizeof(double)) / *cmpSize << std::endl;
    
//     // 重置计时器开始解压阶段
//     timer.Start();
    
//     // 阶段2: 解压处理
//     std::cout << "\n===== 阶段2: 执行解压流水线 =====\n";
    
//     // 使用和压缩相同的内存池！！！！！！不能重置，不然会有内存碎片
//     // ori_data_pool = MemoryPool(poolSize, chunkSize * sizeof(double)); // 重置内存池
//     // cmp_bytes_pool = MemoryPool(poolSize, chunkSize * sizeof(double));
//     // 为解压阶段创建新的内存池
//     // 重要提示：不要创建新对象，只需释放并重用现有的内存池
//     // 这能确保避免出现内存碎片问题 
    
//     //创建新对象会导致内存不够用！！
    
//     //释放然后新建
//     cmp_bytes_pool.reset();
//     ori_data_pool.reset();
//     // MemoryPool dec_data_pool(poolSize, chunkSize * sizeof(double)); // 原始数据显存池

//     // 使用和压缩相同的内存池！！！！！！不能重置，不然会有内存碎片
//     // ori_data_pool = MemoryPool(poolSize, chunkSize * sizeof(double)); // 重置内存池
//     // cmp_bytes_pool = MemoryPool(poolSize, chunkSize * sizeof(double));
//     // 为解压阶段创建新的内存池
//     // 重要提示：不要创建新对象，只需释放并重用现有的内存池
//     // 这能确保避免出现内存碎片问题 
    
//     //创建新对象会导致内存不够用！！
    
//     //释放然后新建
//     cmp_bytes_pool.reset();
//     ori_data_pool.reset();
//     // MemoryPool dec_data_pool(poolSize, chunkSize * sizeof(double)); // 原始数据显存池

//     // 为解压过程创建新的流
//     cudaEvent_t decomp_start, decomp_end;
//     cudaEventCreate(&decomp_start);
//     cudaEventCreate(&decomp_end);
//     cudaEventRecord(decomp_start);
    
//     // 处理每个压缩块
//     for (size_t i = 0; i < compression_infos.size(); ++i) {
//         const auto& info = compression_infos[i];
//         int streamIdx = i % NUM_STREAMS;
        
//         // Validate compressed data size
//         if (info.compressed_size <= 0) {
//             std::cerr << "Error: Invalid compressed size for chunk " << i 
//                      << ": " << info.compressed_size << " bytes" << std::endl;
//             continue;
//         }
        
//         // std::cout << "解压数据块 " << i << " (流 " << streamIdx << "): 原始偏移 " 
//         //           << info.original_offset << ", 压缩偏移 " << info.compressed_offset 
//         //           << ", 大小: " << info.original_size << " 元素, 压缩大小: " 
//         //           << info.compressed_size << " 字节" << std::endl;
        
//         // 分配GPU内存


//         double* d_decData = (double*)ori_data_pool.allocate();
//         // printf("d_decData\n");
//         // printf("d_decData\n");
//         unsigned char* d_cmpBytes = (unsigned char*)cmp_bytes_pool.allocate();
//         // printf("d_cmpBytes:\n");
//         // printf("d_cmpBytes:\n");
//         if (!d_decData || !d_cmpBytes) {
//             std::cerr << "解压阶段内存池分配失败!" << std::endl;
//             break;
//         }

//         // printf("d_decData = %p\n", d_decData);
//         // printf("d_cmpBytes = %p\n", d_cmpBytes);

//         // printf("d_decData = %p\n", d_decData);
//         // printf("d_cmpBytes = %p\n", d_cmpBytes);
        
//         // 设置当前块的索引
//         operators[streamIdx].set_index(i);
        
//         // 执行解压（异步操作）
//         operators[streamIdx].decompress_chunk(
//             decompressed_data, //这个是空的。       输出结果
//             data.cmpBytes,  // 有初始值，压缩后     输入
//             d_decData,      // 它的初始值就是ori    中间
//             d_cmpBytes,     // 有初始值            中间
//             decompressed_data, //这个是空的。       输出结果
//             data.cmpBytes,  // 有初始值，压缩后     输入
//             d_decData,      // 它的初始值就是ori    中间
//             d_cmpBytes,     // 有初始值            中间
//             info.original_size, info.compressed_size, info.compressed_offset, info.original_offset
//         );

//         cudaStreamSynchronize(operators[streamIdx].get_stream());
//         //释放显存
//         // 每处理NUM_STREAMS个块，等待最早的流完成
//         if ((i + 1) >= NUM_STREAMS && (i + 1) % NUM_STREAMS == 0) {
//             for (int j = 0; j < NUM_STREAMS; j++) {
//                 cudaCheckError(cudaStreamSynchronize(operators[j].get_stream()));
                
//                 // 计算并记录解压时间
//                 StreamTiming timing = operators[j].calculate_decompression_timing();
//                 decomp_timings.push_back(timing);
                
//                 // 这里可以释放内存池资源
//                 ori_data_pool.deallocate(d_decData);
//                 cmp_bytes_pool.deallocate(d_cmpBytes);
//                 ori_data_pool.deallocate(d_decData);
//                 cmp_bytes_pool.deallocate(d_cmpBytes);
//             }
//         }
//     }
    
//     // 等待所有未完成的解压操作
//     for (int i = 0; i < NUM_STREAMS; i++) {
//         cudaCheckError(cudaStreamSynchronize(operators[i].get_stream()));
        
//         // 收集最后一批解压操作的时间数据
//         if ((compression_infos.size() - 1) % NUM_STREAMS >= i) {
//             StreamTiming timing = operators[i].calculate_decompression_timing();
//             decomp_timings.push_back(timing);
//         }
        
        
//     }
    
//     cudaEventRecord(decomp_end);
//     cudaEventSynchronize(decomp_end);
    
//     float decompression_time = timer.Elapsed();
//     std::cout << "解压阶段完成，用时: " << decompression_time << " ms" << std::endl;
   
   
//     // 阶段3: 验证解压结果
//     std::cout << "\n===== 阶段3: 验证解压结果 =====\n";
    
//     bool all_verified = true;
//     double max_error = 0.0;
//     double total_error = 0.0;
//     int blocks_verified = 0;
    
//     for (const auto& info : compression_infos) {
//         double chunk_max_error = 0.0;
//         double chunk_avg_error = 0.0;
        
//         bool chunk_verified = verify_chunk(
//             data.oriData + info.original_offset,
//             decompressed_data + info.original_offset,
//             info.original_size,
//             chunk_max_error,
//             chunk_avg_error
//         );
        
//         if (chunk_verified) {
//             blocks_verified++;
//         } else {
//             all_verified = false;
//         }
        
//         max_error = std::max(max_error, chunk_max_error);
//         total_error += (chunk_avg_error * info.original_size);
        
//         std::cout << "数据块 " << (&info - &compression_infos[0]) << " 验证: " 
//                   << (chunk_verified ? "通过" : "失败") 
//                   << " ( Size : "<<info.original_size<<", 最大误差: " << chunk_max_error << ", 平均误差: " << chunk_avg_error << ")" << std::endl;
//                   << " ( Size : "<<info.original_size<<", 最大误差: " << chunk_max_error << ", 平均误差: " << chunk_avg_error << ")" << std::endl;
//     }
    
//     double avg_error = total_error / data.nbEle;
    
//     std::cout << "验证结果: " << (all_verified ? "所有数据块都完全匹配" : "存在不匹配的数据块") << std::endl;
//     std::cout << "验证块数: " << blocks_verified << "/" << compression_infos.size() << std::endl;
//     std::cout << "最大误差: " << max_error << ", 平均误差: " << avg_error << std::endl;
    
//     // 如果需要可视化
//     if (visualize) {
//         // 可视化压缩阶段时间线
//         visualize_timeline(comp_timings);
        
//         // 可视化解压阶段时间线
//         visualize_decompression_timeline(decomp_timings);
        
//         // 分析压缩流水线性能
//         analyze_pipeline(comp_timings, chunkSize,static_cast<double>(data.nbEle * sizeof(double)) / cmpSize, true);
        
//         // 分析解压流水线性能
//         PipelineAnalysis decomp_analysis = analyze_pipeline(decomp_timings, chunkSize,static_cast<double>(data.nbEle * sizeof(double)) / cmpSize, false);
        
//         printf("\n===== 解压流水线执行分析 (块大小: %zu 元素) =====\n", chunkSize);
//         printf("各阶段总时间:\n");
//         printf("- H2D总时间: %.2f ms (平均: %.2f ms/块)\n", decomp_analysis.total_h2d, decomp_analysis.avg_h2d);
//         printf("- 解压计算总时间: %.2f ms (平均: %.2f ms/块)\n", decomp_analysis.total_comp, decomp_analysis.avg_comp);
//         printf("- D2H总时间: %.2f ms (平均: %.2f ms/块)\n", decomp_analysis.total_d2h, decomp_analysis.avg_d2h);
//         printf("- 顺序执行理论时间: %.2f ms\n", decomp_analysis.sequential_time);
//         printf("- 实际并行执行时间: %.2f ms\n", decomp_analysis.end_time);
//         printf("- 流水线加速比: %.2fx\n", decomp_analysis.speedup);
        
//         printf("\n===== 压缩率和验证统计 =====\n");
//         printf("压缩大小: %lu 字节\n", *cmpSize);
//         printf("原始大小: %lu 字节\n", data.nbEle * sizeof(double));
//         printf("压缩比: %.2f\n", static_cast<double>(data.nbEle * sizeof(double)) / *cmpSize);
//         printf("压缩用时: %.2f ms\n", compression_time);
//         printf("解压用时: %.2f ms\n", decompression_time);
//         printf("压缩/解压速度比: %.2f\n", compression_time / decompression_time);
//         printf("压缩/解压速度比: %.2f\n", compression_time / decompression_time);
//         printf("验证结果: %s\n", all_verified ? "所有数据块完全匹配" : "存在不匹配的数据块");
//         if (!all_verified) {
//             printf("最大误差: %e\n", max_error);
//             printf("平均误差: %e\n", avg_error);
//         }
//     }
    
//     // 创建并填充验证结果结构体
//     PipelineVerification verification;
//     verification.compression_time = compression_time;
//     verification.decompression_time = decompression_time;
//     verification.compression_ratio = static_cast<double>(data.nbEle * sizeof(double)) / *cmpSize;
//     verification.all_blocks_verified = all_verified;
//     verification.total_blocks = compression_infos.size();
//     verification.verified_blocks = blocks_verified;
//     verification.max_error = max_error;
//     verification.average_error = avg_error;
    
//     // 释放内存
//     cudaFreeHost(decompressed_data);
    
//     // 清理事件
//     cudaEventDestroy(global_start_event);
//     cudaEventDestroy(global_end_event);
//     cudaEventDestroy(decomp_start);
//     cudaEventDestroy(decomp_end);
//     return verification;
// }


// // 输出块大小与运行时间关系的CSV文件
// void output_blocksize_timing_csv(const std::vector<PipelineAnalysis> &results, const std::string &filename) {
//     std::ofstream csv_file(filename, std::ios::app);
//     if (!csv_file.is_open()) {
//         std::cerr << "无法创建CSV文件: " << filename << std::endl;
//         return;
//     }
//     bool write_header = csv_file.tellp() == 0;
//     if (write_header) {
//     // 写入CSV头
//         csv_file << title <<"\n数据量(MB),块大小(KB),平均H2D时间(ms),平均压缩时间(ms),平均D2H时间(ms),加速比,总时间,吞吐量比例,压缩率\n";
//     }
//     else
//     {
//         csv_file << "\n" << title <<"\n数据量(MB),块大小(KB),平均H2D时间(ms),平均压缩时间(ms),平均D2H时间(ms),加速比,总时间,吞吐量比例,压缩率\n";
//     }

//     // 写入每个块大小的数据
//     for (const auto& result : results) {
//         csv_file << result.total_size << "," 
//                  << (result.chunk_size * sizeof(double) / 1024) << "," 
//                  << result.avg_h2d << "," 
//                  << result.avg_comp << "," 
//                  << result.avg_d2h << "," 
//                  << result.speedup << "," 
//                  << result.end_time << ","
//                  << (result.avg_comp > 0 ? (result.avg_h2d + result.avg_d2h) / result.avg_comp : 0) <<","
//                  << result.comp_level <<"\n";
//     }

//     csv_file.close();
//     std::cout << "已将块大小与运行时间关系保存到 " << filename << std::endl;
// }

// // 可视化块大小与阶段时间的关系
// void visualize_stage_timing_relationship(const std::vector<PipelineAnalysis>& results) {
//     printf("\n===== 块大小与运行时间关系分析 =====\n");
//     printf("总大小(MB) \t块大小(KB) \tH2D(ms) \tComp(ms) \tD2H(ms) \t比例(IO/Comp) \t总时间(ms) \t加速比 \t压缩率 \n");
//     printf("---------------------------------------------------------------------------\n");
    
//     for (const auto& result : results) {
//         float io_comp_ratio = result.avg_comp > 0 ? (result.avg_h2d + result.avg_d2h) / result.avg_comp : 0;
        
//         printf("%8.2f \t %8.2f \t%7.2f \t%7.2f \t%7.2f \t%7.2f \t%7.2f \t%7.2f  \t%7.2f\n", 
//                result.total_size,
//                (result.chunk_size * sizeof(double) / 1024.0), 
//                result.avg_h2d, 
//                result.avg_comp, 
//                result.avg_d2h,
//                io_comp_ratio,
//                result.end_time,
//                result.speedup,
//                result.comp_level);
//     }
// }

// // 从文件测试多个块大小
// int test_multiple_blocksizes(const std::string& file_path, const std::vector<size_t>& block_sizes_kb) {
//     printf("=================================================\n");
//     printf("=======Testing GDFC with Different Block Sizes===\n");
//     printf("=================================================\n");
    
//     // 准备数据
//     ProcessedData data = prepare_data(file_path);
//     if (data.nbEle == 0) {
//         printf("错误: 无法读取文件数据\n");
//         return 1;
//     }
    
//     // 检查GPU可用内存
//     size_t freeMem, totalMem;
//     cudaMemGetInfo(&freeMem, &totalMem);
//     size_t poolSize = freeMem * 0.4;
//     poolSize = (poolSize + 1024*2*sizeof(double)-1) & ~(1024*2*sizeof(double)-1);  // 向上对齐
    
//     std::vector<PipelineAnalysis> results;
    
//     // 对每个块大小进行测试
//     for (size_t block_size_kb : block_sizes_kb) {
//         // 将KB转换为元素数量 (double = 8 bytes)
//         size_t chunkSize = (block_size_kb * 1024) / sizeof(double);
        
//         printf("\n[测试块大小: %zu KB (%zu 元素)]\n", block_size_kb, chunkSize);
        
//         // 执行压缩并获取结果
//         PipelineAnalysis result = execute_pipeline(data, chunkSize, poolSize, true);
//         results.push_back(result);
        
//         // 稍微延迟一下，让GPU冷却
//         std::this_thread::sleep_for(std::chrono::milliseconds(500));
//     }
    
//     // 可视化块大小与阶段时间的关系
//     visualize_stage_timing_relationship(results);
    
//     // 输出CSV数据以便外部绘图分析
//     output_blocksize_timing_csv(results, "block_size_timing_analysis.csv");
    
//     // 清理资源
//     cleanup_data(data);
    
//     return 0;
// }

// // 生成数据测试多个块大小
// int test_multiple_blocksizes_generated(size_t data_size_mb, const std::vector<size_t>& block_sizes_kb, int pattern_type = 0) {
//     printf("=================================================\n");
//     printf("=======Testing GDFC with Different Block Sizes===\n");
//     printf("=================================================\n");
    
//     // 将MB转换为元素数量 (每个元素是double类型，8字节)
//     size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
    
//     // 准备数据
//     ProcessedData data = prepare_data("", nbEle, pattern_type);
//     if (data.nbEle == 0) {
//         printf("错误: 无法生成测试数据\n");
//         return 1;
//     }
    
//     // 检查GPU可用内存
//     size_t freeMem, totalMem;
//     cudaMemGetInfo(&freeMem, &totalMem);
//     size_t poolSize = freeMem * 0.4;
//     poolSize = (poolSize + 1024*2*sizeof(double)-1) & ~(1024*2*sizeof(double)-1);  // 向上对齐
    
//     std::vector<PipelineAnalysis> results;
    
//     // 对每个块大小进行测试
//     for (size_t block_size_kb : block_sizes_kb) {
//         // 将KB转换为元素数量 (double = 8 bytes)
//         size_t chunkSize = (block_size_kb * 1024) / sizeof(double);
        
//         printf("\n[测试块大小: %zu KB (%zu 元素)]\n", block_size_kb, chunkSize);
        
//         // 执行压缩并获取结果
//         PipelineAnalysis result = execute_pipeline(data, chunkSize, poolSize, true);
//         results.push_back(result);
        
//         // 稍微延迟一下，让GPU冷却
//         std::this_thread::sleep_for(std::chrono::milliseconds(500));
//     }
    
//     // 可视化块大小与阶段时间的关系
//     visualize_stage_timing_relationship(results);
    
//     // 输出CSV数据以便外部绘图分析
//     output_blocksize_timing_csv(results, "block_size_timing_analysis_generated.csv");
    
//     // 清理资源
//     cleanup_data(data);
    
//     return 0;
// }

// // 生成二次方增长的块大小序列
// std::vector<size_t> generate_power2_blocksizes(size_t min_kb, size_t max_kb) {
//     std::vector<size_t> sizes;
//     for (size_t size = min_kb; size <= max_kb; size *= 2) {
//         sizes.push_back(size);
//     }
//     return sizes;
// }

// // 生成线性增长的块大小序列
// std::vector<size_t> generate_linear_blocksizes(size_t min_kb, size_t max_kb, size_t step_kb) {
//     std::vector<size_t> sizes;
//     for (size_t size = min_kb; size <= max_kb; size += step_kb) {
//         sizes.push_back(size);
//     }
//     return sizes;
// }

// // 主要测试函数 - 支持文件路径或生成数据
// int test(const std::string &file_path = "", size_t data_size_mb = 0, int pattern_type = 0) {
//     if (!file_path.empty()) {
//         size_t chunkSize;
//         ProcessedData data = prepare_data(file_path);
//         size_t poolSize = setup_gpu_memory_pool(data.nbEle, chunkSize);
//         PipelineVerification verification = execute_pipeline_with_verification(data, chunkSize, poolSize, true);
//         cleanup_data(data);
//         return 0;
//     } else if (data_size_mb > 0) {
//         size_t chunkSize;
//         size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
//         ProcessedData data = prepare_data("", nbEle, pattern_type);
//         size_t poolSize = setup_gpu_memory_pool(data.nbEle, chunkSize);
//         PipelineVerification verification = execute_pipeline_with_verification(data, chunkSize, poolSize, true);
//         cleanup_data(data);
//         return 0;
//     } else {
//         printf("错误: 必须提供文件路径或数据生成参数\n");
//         return 1;
//     }
// }

// int main(int argc, char *argv[])
// {
//     cudaSetDevice(0);
//     cudaSetDevice(0);
//     if (argc < 2) {
//         printf("使用方法:\n");
//         printf("  %s --file <file_path> : 从文件测试\n", argv[0]);
//         printf("  %s --dir <directory_path> : 测试目录中所有文件\n", argv[0]);
//         printf("  %s --generate <size_in_mb> [pattern_type] : 生成数据测试\n", argv[0]);
//         printf("    pattern_type: 0=随机数据, 1=线性增长, 2=正弦波, 3=阶梯\n");
//         printf("  %s --analyze-blocks <file_path> : 分析不同块大小的性能\n", argv[0]);
//         printf("  %s --analyze-blocks-gen <size_in_mb> [pattern_type] : 使用生成数据分析不同块大小的性能\n", argv[0]);
//         return 1;
//     }
    
//     std::string arg = argv[1];
    
//     if (arg == "--file" && argc >= 3) {
//         std::string file_path = argv[2];
//         test(file_path);
//     }
//     else if (arg == "--dir" && argc >= 3) {
//         std::string dir_path = argv[2];
        
//         // 检查目录是否存在
//         if (!fs::exists(dir_path)) {
//             std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
//             return 1;
//         }
        
//         for (const auto& entry : fs::directory_iterator(dir_path)) {
//             if (entry.is_regular_file()) {
//                 std::string file_path = entry.path().string();
//                 std::cout << "正在处理文件: " << file_path << std::endl;
//                 test(file_path);
//                 std::cout << "---------------------------------------------" << std::endl;
//             }
//         }
//     }
//     else if (arg == "--generate" && argc >= 3) {
//         size_t data_size_mb = std::stoul(argv[2]);
//         int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;
        
//         test("", data_size_mb, pattern_type);
//     }
//     else if (arg == "--analyze-blocks" && argc >= 3) {
//         std::string file_path = argv[2];
//         title = "analyze-blocks " + file_path;
//         // 创建不同大小的块序列
//         // 从64KB到64MB，以二次方增长
//         // std::vector<size_t> block_sizes = generate_power2_blocksizes(64*1024/4, 8*65536);
//         std::vector<size_t> block_sizes = generate_linear_blocksizes(16*1024,3*64*1024,1024*4);
//         // std::vector<size_t> block_sizes = generate_power2_blocksizes(64*1024/4, 8*65536);
//         std::vector<size_t> block_sizes = generate_linear_blocksizes(16*1024,3*64*1024,1024*4);
//         test_multiple_blocksizes(file_path, block_sizes);
//     }
//     else if (arg == "--analyze-blocks-gen" && argc >= 3) {
//         size_t data_size_mb = std::stoul(argv[2]);
//         int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;
        
//         std::string pattern_str = (argc >= 4) ? argv[3] : "0";
//         title = std::string("analyze-blocks-gen ") + argv[2] + " " + pattern_str;
        
//         // 创建不同大小的块序列
//         // 从64KB到64MB，以二次方增长
//         // std::vector<size_t> block_sizes = generate_power2_blocksizes(64*1024/4, 8*65536);
//         // std::vector<size_t> block_sizes = generate_linear_blocksizes(64*1024/4,64*1024,1024*16);
//         std::vector<size_t> block_sizes = generate_linear_blocksizes(16*1024,3*64*1024,1024*4);

//         // std::vector<size_t> block_sizes = generate_power2_blocksizes(64*1024/4, 8*65536);
//         // std::vector<size_t> block_sizes = generate_linear_blocksizes(64*1024/4,64*1024,1024*16);
//         std::vector<size_t> block_sizes = generate_linear_blocksizes(16*1024,3*64*1024,1024*4);

//         test_multiple_blocksizes_generated(data_size_mb, block_sizes, pattern_type);
//     }
//     else if (arg == "--analyze-blocks-custom" && argc >= 3) {
//         std::string file_path = argv[2];
        
//         // 使用自定义块大小序列
//         std::vector<size_t> block_sizes;
        
//         // 从命令行参数解析块大小
//         for (int i = 3; i < argc; i++) {
//             block_sizes.push_back(std::stoul(argv[i]));
//         }
        
//         if (block_sizes.empty()) {
//             std::cerr << "错误: 需要至少一个块大小参数" << std::endl;
//             return 1;
//         }
        
//         test_multiple_blocksizes(file_path, block_sizes);
//     }
//     else {
//         printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
//         return 1;
//     }
    
//     return 0;
// }
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include "data/dataset_utils.hpp"
#include "GDFCompressor.cuh"
#include "GDFDecompressor.cuh"
#include <thread>
namespace fs = std::filesystem;

std::string title = "";
#define NUM_STREAMS 16 // 使用16个CUDA Stream实现流水线
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

cudaEvent_t global_start_event, global_end_event; //全局开始和结束事件

// 高精度计时器
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;

public:
    void Start() { start = std::chrono::high_resolution_clock::now(); }

    float Elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(end - start).count();
    }
};

// 流水线总体性能分析
struct PipelineAnalysis {
    float total_size = 0; // 数据总大小
    float compression_ratio = 0;
    float comp_time=0;
    float decomp_time=0;
    float comp_throughout=0;
    float decomp_throughout=0;
    
    float kernal_comp = 0; // 平均计算时间
    float kernal_decomp = 0;

    float init_time = 0;
    float sequential_time = 0;
    float end_time = 0;
    float total_h2d = 0; //总年
    float total_comp = 0; //
    float total_d2h = 0; //
    float speedup = 0;
    float avg_h2d = 0; // 平均H2D时间
    float avg_comp = 0; // 平均计算时间
    float avg_d2h = 0; // 平均D2H时间

    float total_compressed_size = 0; //压缩后总大小
    size_t chunk_size = 0; // 记录块大小
};


// 生成随机数据的函数
std::vector<double> generate_test_data(size_t nbEle, int pattern_type = 0) {
    std::vector<double> data(nbEle);

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (pattern_type) {
        case 0: {
            // 随机数据
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
            break;
        }
        case 1: {
            // 线性增长数据
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<double>(i) * 0.01;
            }
            break;
        }
        case 2: {
            // 正弦波数据
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = 1000.0 * sin(0.01 * i);
            }
            break;
        }
        case 3: {
            // 多步阶数据
            int step_size = nbEle / 10;
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<double>((i / step_size) * 100);
            }
            break;
        }
        default: {
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
        }
    }

    return data;
}

// 数据预处理函数
struct ProcessedData {
    double *oriData;
    unsigned char *cmpBytes;
    unsigned int *cmpSize;
    double *decData;
    size_t nbEle;
};

// 准备数据函数，支持文件和生成数据两种模式
ProcessedData prepare_data(const std::string &source_path = "", size_t generate_size = 0, int pattern_type = 0) {
    ProcessedData result;
    std::vector<double> data;

    // 决定数据来源
    if (generate_size > 0) {
        // 生成指定大小的数据
        printf("生成 %zu 个元素的测试数据 (模式: %d)\n", generate_size, pattern_type);
        data = generate_test_data(generate_size, pattern_type);
        result.nbEle = generate_size;
    } else if (!source_path.empty()) {
        // 从文件读取数据
        printf("从文件加载数据: %s\n", source_path.c_str());
        data = read_data(source_path);
        result.nbEle = data.size();
    } else {
        printf("错误: 未指定数据源\n");
        result.nbEle = 0;
        result.oriData = nullptr;
        result.cmpBytes = nullptr;
        return result;
    }
if(        result.nbEle <= 0)
{
    printf("wrong");
}
    // 分配固定内存
    cudaCheckError(cudaHostAlloc(&result.oriData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpBytes, result.nbEle * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpSize, sizeof(unsigned int), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&result.decData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    // 将数据拷贝到固定内存
#pragma omp parallel for
    for (size_t i = 0; i < result.nbEle; ++i) {
        result.oriData[i] = data[i];
    }

    return result;
}

// 清理资源函数
void cleanup_data(ProcessedData &data) {
    if (data.oriData != nullptr) {
        cudaFreeHost(data.oriData);
        data.oriData = nullptr;
    }

    if (data.cmpBytes != nullptr) {
        cudaFreeHost(data.cmpBytes);
        data.cmpBytes = nullptr;
    }

        if (data.decData != nullptr) {
        cudaFreeHost(data.decData);
        data.decData = nullptr;
    }
}

enum Stage { IDLE, SIZE_PENDING, DATA_PENDING };

// 压缩结果结构体
struct CompressionResult {
    PipelineAnalysis analysis;
    std::vector<size_t> chunkSizes;        // 每个chunk的压缩大小（字节）
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                    // 总chunk数量
    // CompressionInfo cmpInfo;
};

// 执行压缩流程函数
// CompressionResult execute_pipeline(ProcessedData &data, size_t chunkSize, bool visualize = false) {
//     cudaDeviceSynchronize();
//     // 创建时间线记录事件
//     cudaEventCreate(&global_start_event);
//     cudaEventCreate(&global_end_event);
//     //初始化
//     cudaEvent_t init_start_event,init_end_event;
//     cudaEventCreate(&init_start_event);
//     cudaEventCreate(&init_end_event);

//     //初始化
//     cudaDeviceSynchronize();//测量前确保稳定
//     cudaEventRecord(init_start_event);

//     // 主机侧内存分配
//     unsigned int *locCmpSize;
//     cudaCheckError(cudaHostAlloc((void**)&locCmpSize, sizeof(unsigned int) * NUM_STREAMS, cudaHostAllocDefault));
//     unsigned int *h_cmp_offset;
//     cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * NUM_STREAMS + 1, cudaHostAllocDefault));

//     // 初始化偏移量数组
//     h_cmp_offset[0] = 0;

//     bool of_rd[NUM_STREAMS]={0};//内存偏移准备完成信号
//     of_rd[0]=true;

//     // 新增：记录每个chunk信息的数组
//     size_t *chunkElementCounts;
//     cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, sizeof(size_t) * NUM_STREAMS, cudaHostAllocDefault));
//     // 计算总chunk数量
//     size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
//     if(totalChunks==1)
//     // if(1)
//     {
//         chunkSize=data.nbEle;
//     }
//     // 用于存储每个chunk的压缩信息
//     std::vector<size_t> chunkSizes;
//     std::vector<size_t> chunkElementCountsVec;
//     chunkSizes.reserve(totalChunks);
//     chunkElementCountsVec.reserve(totalChunks);

//     // ---------- 创建流池和事件 ----------
//     cudaStream_t streams[NUM_STREAMS];
//     cudaEvent_t evSize[NUM_STREAMS]; // size 拷贝完成
//     cudaEvent_t evData[NUM_STREAMS]; // 数据拷贝完成
//     Stage stage[NUM_STREAMS]; // 每流状态

//     for (int i = 0; i < NUM_STREAMS; ++i) {
//         cudaCheckError(cudaStreamCreate(&streams[i]));
//         cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
//         cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
//         stage[i] = IDLE;
//     }

//     // ---------- 为每个流分配固定设备缓冲 ----------
//     double *d_in[NUM_STREAMS];
//     unsigned char *d_out[NUM_STREAMS];
//     for (int i = 0; i < NUM_STREAMS; ++i) {
//         cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
//         cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double)));
//     }


//     cudaEventRecord(init_end_event);
//     // 等待所有操作完成
//     cudaEventSynchronize(init_end_event);

//     // ---------- 主循环：轮叫 stream 直到数据处理完 ----------
//     size_t processedEle = 0; // 已完成元素数
//     int active = 0;
//     size_t totalCmpSize = 0; // 总压缩大小
//     size_t completedChunks = 0; // 已完成的chunk数量

//     // 记录全局开始时间
//     cudaDeviceSynchronize();//测量前确保稳定
//     cudaEventRecord(global_start_event);
//     // for(int s=0;s<1;s++)
//     // {
//     //     size_t todo = std::min(chunkSize, data.nbEle - processedEle);
//     //     // 记录当前流处理的元素数量
//     //     chunkElementCounts[s] = todo;
//     //     cudaCheckError(cudaMemcpyAsync(
//     //         d_in[s],
//     //         data.oriData,
//     //         todo * sizeof(double), // 修正：使用实际元素数
//     //         cudaMemcpyHostToDevice,
//     //         streams[s]));

//     //     // 调用你的压缩接口（内部处理所有临时内存）
//     //     GDFCompressor::GDFC_compress_stream(
//     //         d_in[s],
//     //         d_out[s],
//     //         &locCmpSize[s],  // 直接传递压缩大小指针
//     //         todo,
//     //         streams[s]);
//     //     // cudaStreamSynchronize(streams[s]);
//     //     unsigned int compressedBits = locCmpSize[s];
//     //     unsigned int compressedBytes = (compressedBits + 7) / 8;
//     //     // // 记录当前chunk的压缩信息
//     //     // chunkSizes.push_back(compressedBytes);
//     //     // chunkElementCountsVec.push_back(chunkElementCounts[s]);
//     //     // totalCmpSize += compressedBytes;
//     //     // 异步 D→H 拷贝结果
//     //     cudaCheckError(cudaMemcpyAsync(
//     //         data.cmpBytes + h_cmp_offset[s],
//     //         d_out[s],
//     //         compressedBytes,  // 使用实际压缩大小
//     //         cudaMemcpyDeviceToHost,
//     //         streams[s]));

//     // }
//     // totalCmpSize += (locCmpSize[0]+7)/8;
//     while (processedEle < data.nbEle || active > 0) {
//         for (int s = 0; s < NUM_STREAMS; ++s) {
//             switch (stage[s]) {
//                 case IDLE:
//                     if (processedEle < data.nbEle ) {
//                         // 计算本批次元素数
//                         size_t todo = std::min(chunkSize, data.nbEle - processedEle);
//                         // 记录当前流处理的元素数量
//                         chunkElementCounts[s] = todo;

//                         // 异步 H→D 拷贝
//                         cudaCheckError(cudaMemcpyAsync(
//                             d_in[s],
//                             data.oriData + processedEle,
//                             todo * sizeof(double), // 修正：使用实际元素数
//                             cudaMemcpyHostToDevice,
//                             streams[s]));

//                         // 调用你的压缩接口（内部处理所有临时内存）
//                         GDFCompressor::GDFC_compress_stream(
//                             d_in[s],
//                             d_out[s],
//                             &locCmpSize[s],  // 直接传递压缩大小指针
//                             todo,
//                             streams[s]);
//                         // cudaStreamSynchronize(streams[s]);
//                         // 记录尺寸事件
//                         cudaCheckError(cudaEventRecord(evSize[s], streams[s]));

//                         stage[s] = SIZE_PENDING;
//                         active += 1;
//                         processedEle += todo;
//                     }
//                     break;

//                 case SIZE_PENDING:
//                     // 查询尺寸已拷回？&&偏移已经准备好
//                     if (cudaEventQuery(evSize[s]) == cudaSuccess && of_rd[s]) {
//                         // 计算压缩后的字节大小
//                         unsigned int compressedBits = locCmpSize[s];
//                         unsigned int compressedBytes = (compressedBits + 7) / 8;

//                         // 异步 D→H 拷贝结果
//                         cudaCheckError(cudaMemcpyAsync(
//                             data.cmpBytes + h_cmp_offset[s],
//                             d_out[s],
//                             compressedBytes,  // 使用实际压缩大小
//                             cudaMemcpyDeviceToHost,
//                             streams[s]));

//                         // 更新偏移量

//                         h_cmp_offset[(s + 1)%NUM_STREAMS] = h_cmp_offset[s] + compressedBytes;
//                         chunkSizes.push_back(compressedBytes);
//                         chunkElementCountsVec.push_back(chunkElementCounts[s]);
//                         of_rd[s]=0;//准备好的偏移已经被使用了
//                         of_rd[(s + 1)%NUM_STREAMS]=1;//给下一个准备好了偏移
//                         cudaCheckError(cudaEventRecord(evData[s], streams[s]));
//                         stage[s] = DATA_PENDING;
//                     }
//                     break;

//                 case DATA_PENDING:
//                     if (cudaEventQuery(evData[s]) == cudaSuccess) {
//                         // 累计压缩大小
//                         unsigned int compressedBytes = (locCmpSize[s] + 7) / 8;
//                         totalCmpSize += compressedBytes;

//                         // 记录当前chunk的压缩信息
//                         completedChunks++;
//                         stage[s] = IDLE;
//                         active -= 1;
//                         // cudaEventRecord(evData[s], streams[s]);
//                     }
//                     break;
//             }
//         }
//     }

//     // 等待所有流完成
//     for (int i = 0; i < NUM_STREAMS; i++) {
//         cudaCheckError(cudaStreamSynchronize(streams[i]));
//         cudaCheckError(cudaEventSynchronize(evData[i]));
//     }
//     // 记录全局结束时间
//     cudaEventRecord(global_end_event);

//     // 等待所有操作完成
//     cudaEventSynchronize(global_end_event);

//     // 计算总时间
//     float totalTime;
//     cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
//     float initTime;
//     cudaEventElapsedTime(&initTime, init_start_event, init_end_event);
//     // 计算压缩比
//     double compressionRatio = static_cast<double>(data.nbEle * sizeof(double)) / totalCmpSize;

//     // 创建分析结果
//     PipelineAnalysis analysis;
//     analysis.compression_ratio = compressionRatio;
//     analysis.total_compressed_size = totalCmpSize;
//     analysis.total_size = data.nbEle * sizeof(double)/1024/1024;
//     analysis.end_time = totalTime;
//     analysis.chunk_size = chunkSize;

//     *data.cmpSize=totalCmpSize;

//     if (visualize) {
//         std::cout << "===== 压缩统计 =====" << std::endl;
//         std::cout << "原始大小: " << data.nbEle * sizeof(double) << " 字节 ("
//                   << data.nbEle * sizeof(double) / (1024 * 1024) << " MB)" << std::endl;
//         std::cout << "压缩后大小: " << totalCmpSize << " 字节 ("
//                   << totalCmpSize / (1024 * 1024) << " MB)" << std::endl;
//         std::cout << "压缩比: " << compressionRatio << std::endl;
//         std::cout << "总chunk数: " << completedChunks << std::endl;

//         std::cout << "总执行时间: " << totalTime << " ms( " << initTime <<") "<<std::endl;
//         std::cout << "压缩吞吐量: " << (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0)
//                   << " GB/s" << std::endl;

//         // 显示每个chunk的详细信息（可选）
//         if (visualize && completedChunks <= 10) { // 只显示前10个chunk避免输出过多
//             std::cout << "\n===== Chunk详细信息 =====" << std::endl;
//             for (size_t i = 0; i < std::min(completedChunks, (size_t)10); ++i) {
//                 std::cout << "Chunk " << i << ": " << chunkElementCountsVec[i] << " 元素, "
//                          << chunkSizes[i] << " 字节压缩" << std::endl;
//             }
//             if (completedChunks > 10) {
//                 std::cout << "... 还有 " << (completedChunks - 10) << " 个chunk" << std::endl;
//             }
//         }
//     }

//     // ---------- 清理资源 ----------
//     // 清理设备内存
//     for (int i = 0; i < NUM_STREAMS; i++) {
//         cudaCheckError(cudaFree(d_out[i]));
//         cudaCheckError(cudaFree(d_in[i]));
//     }

//     // 清理流和事件
//     for (int i = 0; i < NUM_STREAMS; i++) {
//         cudaCheckError(cudaEventDestroy(evSize[i]));
//         cudaCheckError(cudaEventDestroy(evData[i]));
//         cudaCheckError(cudaStreamDestroy(streams[i]));
//     }

//     // 释放主机内存
//     cudaCheckError(cudaFreeHost(locCmpSize));
//     cudaCheckError(cudaFreeHost(h_cmp_offset));
//     cudaCheckError(cudaFreeHost(chunkElementCounts));

//     // 销毁全局事件
//     cudaCheckError(cudaEventDestroy(global_start_event));
//     cudaCheckError(cudaEventDestroy(global_end_event));

//     // 创建并返回完整结果
//     CompressionResult result;
//     result.analysis = analysis;
//     result.chunkSizes = std::move(chunkSizes);
//     result.chunkElementCounts = std::move(chunkElementCountsVec);
//     result.totalChunks = completedChunks;

//     return result;
// }

CompressionResult execute_pipeline(ProcessedData &data, size_t chunkSize, bool visualize = false) {
    cudaDeviceSynchronize();
    // 创建时间线记录事件
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    //初始化
    cudaEvent_t init_start_event,init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);

    //初始化
    cudaDeviceSynchronize();//测量前确保稳定
    cudaEventRecord(init_start_event);

    // 计算总chunk数量
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks==1)
    // if(1)
    {
        chunkSize=data.nbEle;
    }

    printf("totalChunks: %d\n",totalChunks);
    // 主机侧内存分配
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, sizeof(unsigned int) * totalChunks, cudaHostAllocDefault));
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * totalChunks + 1, cudaHostAllocDefault));

    // 初始化偏移量数组
    h_cmp_offset[0] = 0;

    bool of_rd[totalChunks+ NUM_STREAMS]={0};//内存偏移准备完成信号
    of_rd[0]=true;

    // 新增：记录每个chunk信息的数组
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, sizeof(size_t) * totalChunks, cudaHostAllocDefault));

    // 用于存储每个chunk的压缩信息
    std::vector<size_t> chunkSizes;
    std::vector<size_t> chunkElementCountsVec;
    chunkSizes.reserve(totalChunks);
    chunkElementCountsVec.reserve(totalChunks);

    // ---------- 创建流池和事件 ----------
    const int MAX_EVENTS_PER_TYPE = totalChunks + NUM_STREAMS; // 预留一些额外的事件
    
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t kernal_start[MAX_EVENTS_PER_TYPE]; // 核函数启动
    cudaEvent_t evSize[MAX_EVENTS_PER_TYPE]; // size 拷贝完成
    cudaEvent_t evData[MAX_EVENTS_PER_TYPE]; // 数据拷贝完成
    Stage stage[NUM_STREAMS]; // 每流状态

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }

    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&kernal_start[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }


    // ---------- 为每个流分配固定设备缓冲 ----------
    double *d_in[NUM_STREAMS];
    unsigned char *d_out[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheckError(cudaMalloc(&d_in[i], chunkSize * sizeof(double)));
        cudaCheckError(cudaMalloc(&d_out[i], chunkSize * sizeof(double)));
    }


    cudaEventRecord(init_end_event);
    // 等待所有操作完成
    cudaEventSynchronize(init_end_event);

    // ---------- 主循环：轮叫 stream 直到数据处理完 ----------
    size_t processedEle = 0; // 已完成元素数
    int active = 0;
    size_t totalCmpSize = 0; // 总压缩大小
    size_t completedChunks = 0; // 已完成的chunk数量
    // 记录全局开始时间
    cudaDeviceSynchronize();//测量前确保稳定
    cudaEventRecord(global_start_event);

    std::vector<int> chunkIDX(NUM_STREAMS);//和compINFO进行绑定
    while (processedEle < data.nbEle-64 || active > 0) {
        int progress=0;
        for (int s = 0; s < NUM_STREAMS; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedEle < data.nbEle) {
                        // 计算本批次元素数
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo==0)
                        {
                            continue;
                        }
                        progress=1;
                        // printf("idx: %d stream: %d IDLE\n",completedChunks,s);
                        chunkIDX[s]=completedChunks;
                        completedChunks++;
                        // 记录当前流处理的元素数量
                        chunkElementCounts[s] = todo;

                        // 异步 H→D 拷贝
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double), // 修正：使用实际元素数
                            cudaMemcpyHostToDevice,
                            streams[s]));

                        // cudaCheckError(cudaEventRecord(kernal_start[chunkIDX[s]], streams[s]));
                        // 调用你的压缩接口（内部处理所有临时内存）
                        GDFCompressor::GDFC_compress_stream(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s]],  // 直接传递压缩大小指针
                            todo,
                            streams[s]);
                        // cudaStreamSynchronize(streams[s]);
                        // 记录尺寸事件
                        active += 1;
                        processedEle += todo;
                        // stage[s] = WAIT;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));

                        stage[s] = SIZE_PENDING;
                    }
                    break;
                // case WAIT:
                //         if(of_rd[chunkIDX[s]])
                //         {
                //         }
                //         else
                //         {
                //             printf("idx: %d,stream: %d NOOFFSET\n",chunkIDX[s],s);
                //         }
                //         break;
                case SIZE_PENDING:
                    // if(!of_rd[chunkIDX[s]])
                    // {
                    //     printf("idx: %d,stream: %d NOOFFSET\n",chunkIDX[s],s);
                    // }
                    // else if(cudaEventQuery(evSize[chunkIDX[s]]) != cudaSuccess)
                    // {
                    //     printf("idx: %d,stream: %d NOSIZE_EVENT\n",chunkIDX[s],s);
                    // }
                    // 查询尺寸已拷回？&& 偏移已经准备好
                    if(cudaEventQuery(evSize[chunkIDX[s]]) == cudaSuccess && of_rd[chunkIDX[s]]) {
                        progress=1;
                        // 计算压缩后的字节大小
                        int idx=chunkIDX[s];
                        // printf("idx: %d,stream: %d SIZE\n",idx,s);
                        unsigned int compressedBits = locCmpSize[idx];
                        unsigned int compressedBytes = (compressedBits + 7) / 8;

                        // 异步 D→H 拷贝结果
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,  // 使用实际压缩大小
                            cudaMemcpyDeviceToHost,
                            streams[s]));

                        // 更新偏移量

                        h_cmp_offset[idx+1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx+1]=true;//给下一个准备好了偏移
                        // printf("idx: %d,stream: %d offready\n",idx+1,s);
                        chunkSizes.push_back(compressedBytes);
                        chunkElementCountsVec.push_back(chunkElementCounts[s]);
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;

                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        // 累计压缩大小
                        // printf("idx: %d stream: %d DATA\n",chunkIDX[s],s);
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s]] + 7) / 8;
                        totalCmpSize += compressedBytes;
                        progress=1;
                        stage[s] = IDLE;
                        active -= 1;
                        // cudaEventRecord(evData[s], streams[s]);
                    }
                    break;
            }
        }
        // 避免忙等待 - 如果没有进展，短暂休眠
        if (!progress) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }

    // 等待所有流完成
    for (int i = 0; i < (NUM_STREAMS>totalChunks?totalChunks:NUM_STREAMS); i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    // 记录全局结束时间
    cudaEventRecord(global_end_event);

    // 等待所有操作完成
    cudaEventSynchronize(global_end_event);
    float kernalTime=0;
    // for(int i=0;i<completedChunks;i++)
    // {
    //     float tmpTime;
    //     cudaEventElapsedTime(&tmpTime, kernal_start[i], evSize[i]);
    //     kernalTime+=tmpTime;
    // }
    // 计算总时间
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    float initTime;
    cudaEventElapsedTime(&initTime, init_start_event, init_end_event);
    // 计算压缩比
    double compressionRatio =totalCmpSize/static_cast<double>(data.nbEle * sizeof(double)) ;

    // 创建分析结果
    PipelineAnalysis analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(double)/1024/1024;
    analysis.comp_time = totalTime;
    analysis.comp_throughout=(data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    analysis.kernal_comp = kernalTime;
    *data.cmpSize=totalCmpSize;
//     CompressionInfo tmp{
// compressionRatio,0,totalTime,(data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0),0,0,0


//     };
    if (visualize) {
        std::cout << "===== 压缩统计 =====" << std::endl;
        std::cout << "原始大小: " << data.nbEle * sizeof(double) << " 字节 ("
                  << data.nbEle * sizeof(double) / (1024 * 1024) << " MB)" << std::endl;
        std::cout << "压缩后大小: " << totalCmpSize << " 字节 ("
                  << totalCmpSize / (1024 * 1024) << " MB)" << std::endl;
        std::cout << "压缩比: " << compressionRatio << std::endl;
        std::cout << "总chunk数: " << completedChunks << std::endl;

        std::cout << "总执行时间: " << totalTime << " ms( " << initTime <<") "<<std::endl;
        std::cout << "压缩吞吐量: " << (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0)
                  << " GB/s" << std::endl;

        // 显示每个chunk的详细信息（可选）
        if (visualize && completedChunks <= 10) { // 只显示前10个chunk避免输出过多
            std::cout << "\n===== Chunk详细信息 =====" << std::endl;
            for (size_t i = 0; i < std::min(completedChunks, (size_t)10); ++i) {
                std::cout << "Chunk " << i << ": " << chunkElementCountsVec[i] << " 元素, "
                         << chunkSizes[i] << " 字节压缩" << std::endl;
            }
            if (completedChunks > 10) {
                std::cout << "... 还有 " << (completedChunks - 10) << " 个chunk" << std::endl;
            }
        }
    }

    // ---------- 清理资源 ----------
    // 清理设备内存
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }

    // 清理流和事件
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }

    // 释放主机内存
    cudaCheckError(cudaFreeHost(locCmpSize));
    cudaCheckError(cudaFreeHost(h_cmp_offset));
    cudaCheckError(cudaFreeHost(chunkElementCounts));

    // 销毁全局事件
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));

    // 创建并返回完整结果
    CompressionResult result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    // result.cmpInfo=tmp;
    return result;
}
/*
void comp(ProcessedData &data,size_t chunk = 0)
{
    cudaDeviceSynchronize();//测量前确保稳定
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    size_t nbEle = data.nbEle;

    unsigned int cmpSize = 0; // 将由压缩函数设置
    size_t cmpSize0 = 0; // 将由压缩函数设置

    double* d_oriData;
    double* d_decData;
    double* d_decData2;
    unsigned char* d_cmpBytes;
    unsigned char* d_cmpBytes2;
    unsigned char* h_cmpBytes;
    double* h_decData;
    double* h_decData2;

    // cudaHostAlloc((void**)&h_cmpBytes, nbEle * sizeof(double) * 2, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_decData, nbEle * sizeof(double), cudaHostAllocDefault);
    // cudaHostAlloc((void**)&h_decData2, nbEle * sizeof(double), cudaHostAllocDefault);


    cudaMalloc(&d_decData, nbEle * sizeof(double));
    // cudaMalloc(&d_decData2, nbEle * sizeof(double));
    // cudaMalloc(&d_oriData, nbEle * sizeof(double));
    // cudaMalloc(&d_cmpBytes, nbEle * 2 * sizeof(double));
    cudaMalloc(&d_cmpBytes2, nbEle * 2 * sizeof(double));


    // cudaMemcpyAsync(d_oriData, data.oriData, nbEle * sizeof(double), cudaMemcpyHostToDevice,stream);

    // GDFCompressor::GDFC_compress_stream(d_oriData, d_cmpBytes, &cmpSize,nbEle, stream);
    // cudaStreamSynchronize(stream);

    // cudaMemcpyAsync(h_cmpBytes, d_cmpBytes, (cmpSize + 7) / 8, cudaMemcpyDeviceToHost,stream);
    // cudaStreamSynchronize(stream);  // 确保数据复制完成

    // GDFC.GDFC_compress(d_oriData, d_cmpBytes, &cmpSize0,nbEle, stream);
    // cudaStreamSynchronize(stream);

    // cudaMemcpyAsync(h_cmpBytes, d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost,stream);
    // cudaStreamSynchronize(stream);  // 确保数据复制完成

    // GDFCompressor GDFC;
    // GDFC.GDFC_compress(d_oriData, d_cmpBytes,nbEle, &cmpSize0, stream);
    // cudaStreamSynchronize(stream);

    // cudaMemcpyAsync(h_cmpBytes, d_cmpBytes, cmpSize0, cudaMemcpyDeviceToHost,stream);
    // cudaStreamSynchronize(stream);  // 确保数据复制完成
    //     // 打印压缩结果
    // printf("\n压缩成功！压缩后大小: %zu 字节 (%.2f%% 原始大小)\n",
    //     cmpSize0, (float)cmpSize0/ (nbEle * sizeof(double)) * 100.0f);




    printf("st前16字节(十六进制): ");
    for (size_t i = 0; i < 16*8; ++i) {
        if((i)%16==0)
        {
            printf("\n");
        }
        printf("%02X ", data.cmpBytes[i]);  // 十六进制格式打印
    }
    printf("\n");

    // printf("comp前16字节(十六进制): ");
    // for (size_t i = 0; i < 16*8; ++i) {
    //     if((i)%16==0)
    //     {
    //         printf("\n");
    //     }
    //     if(h_cmpBytes[i]!=data.cmpBytes[i])
    //     {
    //         printf("%02X ", h_cmpBytes[i]);  // 十六进制格式打印
    //     }
    //     else
    //     {
    //         printf("LL ");
    //     }
    // }
    // printf("\n");

    //解压流式压缩数据
    cudaDeviceSynchronize();//测量前确保稳定
    size_t compSize_dec=*data.cmpSize;//(cmpSize + 7) / 8;10088494;//
    printf("cmpSize:%zu\n",compSize_dec);
    cudaMemcpyAsync(d_cmpBytes2, data.cmpBytes,compSize_dec, cudaMemcpyHostToDevice,stream);

    GDFDecompressor GDFD;
    GDFD.GDFC_decompress(d_decData, d_cmpBytes2, nbEle, compSize_dec, stream);
    cudaStreamSynchronize(stream);

    cudaMemcpyAsync(h_decData, d_decData, nbEle * sizeof(double), cudaMemcpyDeviceToHost,stream);
    cudaStreamSynchronize(stream);


    // //解压单流压缩数据
    // GDFD.GDFC_decompress(d_decData2, d_cmpBytes, nbEle, compSize_dec, stream);
    // cudaStreamSynchronize(stream);

    // cudaMemcpyAsync(h_decData2, d_decData2, nbEle * sizeof(double), cudaMemcpyDeviceToHost,stream);
    // cudaStreamSynchronize(stream);

    for(int i=0;i<nbEle;i++)
    {
        if(data.oriData[i]!=h_decData[i])
        {
            printf("idx: %d ori: %f,dec_ps: %f\n",i,data.oriData[i],h_decData[i]);
            break;
        }
        // if(i%(1024*16)==10)
        // {
        //     i+=1024*16-10;
        // }
        // printf("ori: %f,dec_pp: %f,dec_ps: %f,dec_ss: %f\n",data.oriData[i],data.decData[i],h_decData[i],h_decData2[i]);
    }

    cudaFree(d_oriData);
    cudaFree(d_cmpBytes);
    cudaFree(d_cmpBytes2);
    cudaFree(d_decData);
    cudaFreeHost(h_cmpBytes);
    cudaFreeHost(h_decData);
    cudaStreamDestroy(stream);

}

*/
// 更新的CompressedData结构体
struct CompressedData {
    unsigned char* cmpBytes;                // 压缩数据缓冲
    size_t totalCompressedSize;             // 总压缩大小
    size_t totalElements;                   // 原始数据元素总数
    std::vector<size_t> chunkSizes;         // 每个chunk的压缩大小（字节）
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                     // 总chunk数量
};

// 辅助函数：从压缩结果创建CompressedData结构
CompressedData createCompressedData(const CompressionResult& compResult,
                                  const ProcessedData& originalData) {
    CompressedData compData;
    compData.cmpBytes = originalData.cmpBytes;  // 指向压缩数据缓冲
    compData.totalCompressedSize = compResult.analysis.total_compressed_size;
    compData.totalElements = originalData.nbEle;
    compData.chunkSizes = compResult.chunkSizes;
    compData.chunkElementCounts = compResult.chunkElementCounts;
    compData.totalChunks = compResult.totalChunks;

    return compData;
}

// 执行解压流程函数
PipelineAnalysis execute_decompression_pipeline(const CompressedData &compData, ProcessedData &decompData, bool visualize = false) {
    cudaDeviceSynchronize();
    // 创建时间线记录事件
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);

    // 主机侧内存分配 - 用于记录每个流的处理信息
    size_t *streamChunkIds;
    cudaCheckError(cudaHostAlloc((void**)&streamChunkIds, sizeof(size_t) * NUM_STREAMS, cudaHostAllocDefault));
    size_t *streamOutputOffsets;
    cudaCheckError(cudaHostAlloc((void**)&streamOutputOffsets, sizeof(size_t) * NUM_STREAMS, cudaHostAllocDefault));

    // ---------- 创建流池和事件 ----------
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t evSize[NUM_STREAMS]; // 压缩数据拷贝完成
    cudaEvent_t evData[NUM_STREAMS]; // 解压数据拷贝完成
    Stage stage[NUM_STREAMS]; // 每流状态

    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
        stage[i] = IDLE;
    }

    // ---------- 为每个流分配固定设备缓冲 ----------
    unsigned char *d_in[NUM_STREAMS];  // 压缩数据输入缓冲
    double *d_out[NUM_STREAMS];        // 解压数据输出缓冲

    // 计算最大chunk大小用于内存分配
    size_t maxChunkSize = 0;
    size_t maxCompressedSize = 0;
    for (size_t i = 0; i < compData.totalChunks; ++i) {
        maxChunkSize = std::max(maxChunkSize, compData.chunkElementCounts[i]);
        maxCompressedSize = std::max(maxCompressedSize, compData.chunkSizes[i]);
    }

    for (int i = 0; i < NUM_STREAMS; ++i) {
        // 为压缩数据分配缓冲
        cudaCheckError(cudaMalloc(&d_in[i], maxCompressedSize));
        // 为解压数据分配缓冲
        cudaCheckError(cudaMalloc(&d_out[i], maxChunkSize * sizeof(double)));
    }

    // ---------- 主循环：轮叫 stream 直到数据处理完 ----------
    size_t processedChunks = 0; // 已完成chunk数
    size_t processedElements = 0; // 已完成元素数
    int active = 0;
    size_t totalDecompSize = 0; // 总解压大小
    size_t compressedDataOffsets[compData.totalChunks]={0};
    for (size_t i = 0; i < compData.totalChunks -1; ++i) {
        compressedDataOffsets[i+1] += compData.chunkSizes[i];
    }
    // 记录全局开始时间
    cudaDeviceSynchronize(); // 测量前确保稳定
    cudaEventRecord(global_start_event);

    while (processedChunks < compData.totalChunks || active > 0) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedChunks < compData.totalChunks) {
                        active += 1;
                        size_t chunkId = processedChunks;
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        size_t currentChunkCompressedSize = compData.chunkSizes[chunkId];

                        // 记录当前流处理的chunk信息
                        streamChunkIds[s] = chunkId;
                        streamOutputOffsets[s] = processedElements;

                        // 计算压缩数据在缓冲中的偏移
                        size_t compressedDataOffset = compressedDataOffsets[chunkId];
                        // for (size_t i = 0; i < chunkId; ++i) {
                        //     compressedDataOffset += compData.chunkSizes[i];
                        // }

                        // 异步 H→D 拷贝压缩数据
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            compData.cmpBytes + compressedDataOffset,
                            currentChunkCompressedSize,
                            cudaMemcpyHostToDevice,
                            streams[s]));

                        // 调用解压接口
                        GDFDecompressor GDFD;
                        GDFD.GDFC_decompress_stream_optimized(
                        d_out[s],                    // 解压输出
                        d_in[s],                     // 压缩输入
                        currentChunkElements,        // 原始元素数量
                        currentChunkCompressedSize,  // 压缩数据大小
                        streams[s]);

                        // 记录压缩数据拷贝和解压完成事件
                        cudaCheckError(cudaEventRecord(evSize[s], streams[s]));

                        stage[s] = SIZE_PENDING;
                        processedChunks++;
                        processedElements += currentChunkElements;

                        size_t outputOffset = streamOutputOffsets[s];

                        // 异步 D→H 拷贝解压结果
                        cudaCheckError(cudaMemcpyAsync(
                            decompData.decData + outputOffset,
                            d_out[s],
                            currentChunkElements * sizeof(double),
                            cudaMemcpyDeviceToHost,
                            streams[s]));

                        cudaCheckError(cudaEventRecord(evData[s], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;

                case DATA_PENDING:
                    if (cudaEventQuery(evData[s]) == cudaSuccess) {
                        // 累计解压大小
                        size_t chunkId = streamChunkIds[s];
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        totalDecompSize += currentChunkElements * sizeof(double);

                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
            }
        }
    }
    // 等待所有流完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaStreamSynchronize(streams[i]));
        cudaCheckError(cudaEventSynchronize(evData[i]));
    }
    // 记录全局结束时间
    cudaEventRecord(global_end_event);

    // 等待所有操作完成
    cudaEventSynchronize(global_end_event);

    // 计算总时间
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);

    // 创建并返回分析结果
    PipelineAnalysis result;
    result.compression_ratio =totalDecompSize/ static_cast<double>(compData.totalCompressedSize) ;
    result.total_compressed_size = compData.totalCompressedSize;
    result.total_size = totalDecompSize / 1024 / 1024;
    result.decomp_time = totalTime;
    result.decomp_throughout=(totalDecompSize  / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    // for(size_t i=0;i<10;i++)
    // {
    //     printf("ori: %f , dec: %f \n",decompData.oriData[i],decompData.decData[i]);
    // }
    // for(int i=0,z=0;i<compData.totalChunks ;i++)
    // {
    //     for(int j=z;j<z+2;j++)
    //     printf("idx: %d ,ori: %f , dec: %f \n",j,decompData.oriData[j],decompData.decData[j]);
    //     z+=compData.chunkElementCounts[i];
    // }
    // printf("\n");
    // for(size_t i=0,j=0;i<10;i++)
    // {
    //     while(decompData.oriData[j]==decompData.decData[j]&&j<processedElements)
    //     {
    //         j++;
    //     }
    //     printf("idx: %d ,ori: %f , dec: %f \n",j,decompData.oriData[j],decompData.decData[j]);
    //     j++;
    // }
    if (visualize) {
        std::cout << "===== 解压统计 =====" << std::endl;
        std::cout << "压缩大小: " << compData.totalCompressedSize << " 字节 ("
                  << compData.totalCompressedSize / (1024 * 1024) << " MB)" << std::endl;
        std::cout << "解压后大小: " << totalDecompSize << " 字节 ("
                  << totalDecompSize / (1024 * 1024) << " MB)" << std::endl;
        std::cout << "原始压缩比: " << result.compression_ratio << std::endl;
        std::cout << "处理chunk数: " << compData.totalChunks << std::endl;

        std::cout << "总执行时间: " << totalTime << " ms" << std::endl;
        std::cout << "解压吞吐量: " << (totalDecompSize / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0)
                  << " GB/s" << std::endl;
    }

    // ---------- 清理资源 ----------
    // 清理设备内存
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }

    // 清理流和事件
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }

    // 释放主机内存
    cudaCheckError(cudaFreeHost(streamChunkIds));
    cudaCheckError(cudaFreeHost(streamOutputOffsets));

    // 销毁全局事件
    cudaCheckError(cudaEventDestroy(global_start_event));
    cudaCheckError(cudaEventDestroy(global_end_event));

    return result;
}

// 输出块大小与运行时间关系的CSV文件
void output_blocksize_timing_csv(const std::vector<PipelineAnalysis> &results, const std::string &filename) {
    std::ofstream csv_file(filename, std::ios::app);
    if (!csv_file.is_open()) {
        std::cerr << "无法创建CSV文件: " << filename << std::endl;
        return;
    }
    bool write_header = csv_file.tellp() == 0;
    if (write_header) {
        // 写入CSV头
        csv_file << title << "\n数据量(MB),块大小(KB),平均H2D时间(ms),平均压缩时间(ms),平均D2H时间(ms),加速比,总时间,吞吐量比例,压缩率\n";
    } else {
        csv_file << "\n" << title << "\n数据量(MB),块大小(KB),平均H2D时间(ms),平均压缩时间(ms),平均D2H时间(ms),加速比,总时间,吞吐量比例,压缩率\n";
    }

    // 写入每个块大小的数据
    for (const auto &result: results) {
        csv_file << result.total_size << ","
                << (result.chunk_size * sizeof(double) / 1024) << ","
                << result.avg_h2d << ","
                << result.avg_comp << ","
                << result.avg_d2h << ","
                << result.speedup << ","
                << result.end_time << ","
                << (result.avg_comp > 0 ? (result.avg_h2d + result.avg_d2h) / result.avg_comp : 0) << ","
                << result.compression_ratio << "\n";
    }

    csv_file.close();
    std::cout << "已将块大小与运行时间关系保存到 " << filename << std::endl;
}

// 可视化块大小与阶段时间的关系
void visualize_stage_timing_relationship(const std::vector<PipelineAnalysis> &results) {
    printf("\n===== 块大小与运行时间关系分析 =====\n");
    printf("总大小(MB) \t块大小(KB) \tH2D(ms) \tComp(ms) \tD2H(ms) \t比例(IO/Comp) \t总时间(ms) \t加速比 \t压缩率 \n");
    printf("---------------------------------------------------------------------------\n");

    for (const auto &result: results) {
        float io_comp_ratio = result.avg_comp > 0 ? (result.avg_h2d + result.avg_d2h) / result.avg_comp : 0;

        printf("%8.2f \t %8.2f \t%7.2f \t%7.2f \t%7.2f \t%7.2f \t%7.2f \t%7.2f  \t%7.2f\n",
               result.total_size,
               (result.chunk_size * sizeof(double) / 1024.0),
               result.avg_h2d,
               result.avg_comp,
               result.avg_d2h,
               io_comp_ratio,
               result.end_time,
               result.speedup,
               result.compression_ratio);
    }
}

// 从文件测试多个块大小
int test_multiple_blocksizes(const std::string &file_path, const std::vector<size_t> &block_sizes_kb) {
    printf("=================================================\n");
    printf("=======Testing GDFC with Different Block Sizes===\n");
    printf("=================================================\n");

    // 准备数据
    ProcessedData data = prepare_data(file_path);
    if (data.nbEle == 0) {
        printf("错误: 无法读取文件数据\n");
        return 1;
    }
    printf("GPU预热中...\n");
    size_t warmup_chunk = data.nbEle; // 单流
    execute_pipeline(data, warmup_chunk, false);
    cudaDeviceSynchronize(); // 确保预热完成

    // 检查GPU可用内存
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);

    std::vector<PipelineAnalysis> results;

    // 对每个块大小进行测试
    for (size_t block_size_kb: block_sizes_kb) {
        // 将KB转换为元素数量 (double = 8 bytes)
        size_t chunkSize = (block_size_kb * 1024) / sizeof(double);

        printf("\n[测试块大小: %zu KB (%zu 元素)]\n", block_size_kb, chunkSize);

        PipelineAnalysis result_avg;
        // 压缩阶段
        // CompressionResult compResult = execute_pipeline(data, chunkSize, true);
        // cudaDeviceSynchronize(); // 确保预热完成
        // CompressedData compData = createCompressedData(compResult, data);
        // PipelineAnalysis decompAnalysis = execute_decompression_pipeline(compData, data, true);

        // comp(data,chunkSize);
        for(int i=0;i<3;i++)
        {
            CompressionResult compResult = execute_pipeline(data, chunkSize, true);
            cudaDeviceSynchronize(); 
            PipelineAnalysis result = compResult.analysis;
            result_avg.comp_time+=result.comp_time/3;
            
        }
        result_avg.chunk_size+=chunkSize;
        results.push_back(result_avg);



        // 稍微延迟一下，让GPU冷却
        std::this_thread::sleep_for(std::chrono::milliseconds(500*6));
    }

    // 可视化块大小与阶段时间的关系
    visualize_stage_timing_relationship(results);

    // 输出CSV数据以便外部绘图分析
    output_blocksize_timing_csv(results, "block_size_timing_analysis.csv");

    // 清理资源
    cleanup_data(data);

    return 0;
}

// 生成数据测试多个块大小
int test_multiple_blocksizes_generated(size_t data_size_mb, const std::vector<size_t> &block_sizes_kb,
                                       int pattern_type = 0) {
    printf("=================================================\n");
    printf("=======Testing GDFC with Different Block Sizes===\n");
    printf("=================================================\n");


    // 准备数据
    // 将MB转换为元素数量 (每个元素是double类型，8字节)
    size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
    ProcessedData data = prepare_data("", nbEle, pattern_type);
    if (data.nbEle == 0) {
        printf("错误: 无法读取文件数据\n");
        return 1;
    }
    printf("GPU预热中...\n");
    size_t warmup_chunk = data.nbEle; // 单流
    execute_pipeline(data, warmup_chunk, false);
    cudaDeviceSynchronize(); // 确保预热完成


    // 检查GPU可用内存
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    // size_t poolSize = freeMem * 0.4;
    // poolSize = (poolSize + 1024 * 2 * sizeof(double) - 1) & ~(1024 * 2 * sizeof(double) - 1); // 向上对齐

    std::vector<PipelineAnalysis> results;

    // 对每个块大小进行测试
    for (size_t block_size_kb: block_sizes_kb) {
        // 将KB转换为元素数量 (double = 8 bytes)
        size_t chunkSize = (block_size_kb * 1024) / sizeof(double);

        printf("\n[测试块大小: %zu KB (%zu 元素)]\n", block_size_kb, chunkSize);

        // 执行压缩并获取结果
        for(int i=0;i<3;i++)
        {
            CompressionResult compResult = execute_pipeline(data, chunkSize, true);
            PipelineAnalysis result = compResult.analysis;
            results.push_back(result);
        }

        // 稍微延迟一下，让GPU冷却
        std::this_thread::sleep_for(std::chrono::milliseconds(500*3));
    }

    // 可视化块大小与阶段时间的关系
    visualize_stage_timing_relationship(results);

    // 输出CSV数据以便外部绘图分析
    output_blocksize_timing_csv(results, "block_size_timing_analysis_generated.csv");

    // 清理资源
    cleanup_data(data);

    return 0;
}

// 生成二次方增长的块大小序列
std::vector<size_t> generate_power2_blocksizes(size_t min_kb, size_t max_kb) {
    std::vector<size_t> sizes;
    for (size_t size = min_kb; size <= max_kb; size *= 2) {
        sizes.push_back(size);
    }
    return sizes;
}

// 生成线性增长的块大小序列
std::vector<size_t> generate_linear_blocksizes(size_t min_kb, size_t max_kb, size_t step_kb) {
    std::vector<size_t> sizes;
    for (size_t size = min_kb; size <= max_kb; size += step_kb) {
        sizes.push_back(size);
    }
    return sizes;
}


void warmup()
{
        size_t nbEle = (1*1024 * 1024) / sizeof(double);
        ProcessedData data = prepare_data("", nbEle, 0);
        if (data.nbEle == 0) {
            printf("错误: 无法读取文件数据\n");
            return ;
        }
        printf("GPU预热中...\n");
        size_t warmup_chunk = data.nbEle; // 单流
        execute_pipeline(data, warmup_chunk, false);
        cudaDeviceSynchronize(); // 确保预热完成
}

int setChunk(int nbEle)
{
    size_t chunkSize=1;
    size_t temp=nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
    while(chunkSize<=MAX_NUMS_PER_CHUNK && chunkSize<=temp)
    {
        chunkSize*=2;
    }
    return chunkSize;
}

CompressionInfo test_compression(ProcessedData data, size_t chunkSize)
{

    CompressionResult compResult = execute_pipeline(data, chunkSize, false);
    cudaDeviceSynchronize(); // 确保预热完成
    CompressedData compData = createCompressedData(compResult, data);
    PipelineAnalysis decompAnalysis = execute_decompression_pipeline(compData, data, false);
    return CompressionInfo{compResult.analysis.compression_ratio,0,compResult.analysis.comp_time,compResult.analysis.comp_throughout,0,decompAnalysis.decomp_time,decompAnalysis.decomp_throughout};
}
// 主要测试函数 - 支持文件路径或生成数据
int test(const std::string &file_path = "", size_t data_size_mb = 0, int pattern_type = 0) {
    warmup();
    if (!file_path.empty()) {
        size_t chunkSize=1;
        ProcessedData data = prepare_data(file_path);
        // size_t poolSize = setup_gpu_memory_pool(data.nbEle, chunkSize);
        // size_t temp=data.nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
        // while(chunkSize<=MAX_NUMS_PER_CHUNK && chunkSize<=temp)
        // {
        //     chunkSize*=2;
        // }
        chunkSize=setChunk(data.nbEle);
        CompressionInfo a=test_compression(data,chunkSize);
        for(int i=0;i<2;i++)
        {
            a+=test_compression(data,chunkSize);
        }
        a=a/3;
        a.print();
        cleanup_data(data);
        return 0;
    } else if (data_size_mb > 0) {
        size_t chunkSize;
        size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
        ProcessedData data = prepare_data("", nbEle, pattern_type);
        // size_t temp=data.nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
        // while(chunkSize<=MAX_NUMS_PER_CHUNK && chunkSize<=temp)
        // {
        //     chunkSize*=2;
        // }
        chunkSize=setChunk(data.nbEle);
        CompressionInfo a=test_compression(data,chunkSize);
        for(int i=0;i<2;i++)
        {
            a+=test_compression(data,chunkSize);
        }
        a=a/3;
        a.print();
        // CompressionResult compResult = execute_pipeline(data, chunkSize, true);
        // cudaDeviceSynchronize(); // 确保预热完成
        // CompressedData compData = createCompressedData(compResult, data);
        // PipelineAnalysis decompAnalysis = execute_decompression_pipeline(compData, data, true);
        cleanup_data(data);
        return 0;
    } else {
        printf("错误: 必须提供文件路径或数据生成参数\n");
        return 1;
    }
}

int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    if (argc < 2) {
        printf("使用方法:\n");
        printf("  %s --file <file_path> : 从文件测试\n", argv[0]);
        printf("  %s --dir <directory_path> : 测试目录中所有文件\n", argv[0]);
        printf("  %s --generate <size_in_mb> [pattern_type] : 生成数据测试\n", argv[0]);
        printf("    pattern_type: 0=随机数据, 1=线性增长, 2=正弦波, 3=阶梯\n");
        printf("  %s --analyze-blocks <file_path> : 分析不同块大小的性能\n", argv[0]);
        printf("  %s --analyze-blocks-gen <size_in_mb> [pattern_type] : 使用生成数据分析不同块大小的性能\n", argv[0]);
        return 1;
    }

    std::string arg = argv[1];

    if (arg == "--file" && argc >= 3) {
        std::string file_path = argv[2];
        test(file_path);
    } else if (arg == "--dir" && argc >= 3) {
        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }

        for (const auto &entry: fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                test(file_path);
                std::cout << "---------------------------------------------" << std::endl;
            }
        }
    } else if (arg == "--generate" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

        test("", data_size_mb, pattern_type);
    } else if (arg == "--analyze-blocks" && argc >= 3) {
        std::string file_path = argv[2];
        title = "analyze-blocks " + file_path;
        // 创建不同大小的块序列
        // 从16mbB到512MB，以二次方增长
        std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024/4, 2*64*1024);
        // std::vector<size_t> block_sizes = generate_linear_blocksizes(32 * 1024, 3 * 64 * 1024, 1024 * 32);
        test_multiple_blocksizes(file_path, block_sizes);
    } else if (arg == "--analyze-blocks-gen" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

        std::string pattern_str = (argc >= 4) ? argv[3] : "0";
        title = std::string("analyze-blocks-gen ") + argv[2] + " " + pattern_str;

        // 创建不同大小的块序列
        // 从16mbB到512MB，以二次方增长
        std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024, 8*64*1024);
        // std::vector<size_t> block_sizes = generate_linear_blocksizes(32 * 1024, 3 * 64 * 1024, 1024 * 32);

        test_multiple_blocksizes_generated(data_size_mb, block_sizes, pattern_type);
    } else if (arg == "--analyze-blocks-custom" && argc >= 3) {
        std::string file_path = argv[2];

        // 使用自定义块大小序列
        std::vector<size_t> block_sizes;

        // 从命令行参数解析块大小
        for (int i = 3; i < argc; i++) {
            block_sizes.push_back(std::stoul(argv[i]));
        }

        if (block_sizes.empty()) {
            std::cerr << "错误: 需要至少一个块大小参数" << std::endl;
            return 1;
        }

        test_multiple_blocksizes(file_path, block_sizes);
    } else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }

    return 0;
}
