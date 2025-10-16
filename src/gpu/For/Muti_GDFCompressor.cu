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

#define NUM_STREAMS 16
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

enum Stage { IDLE, SIZE_PENDING, DATA_PENDING };

// 核心数据结构
struct DataBuffer {
    double *oriData;           // 原始数据
    unsigned char *cmpBytes;   // 压缩数据缓冲
    double *decData;           // 解压数据
    size_t totalElements;      // 总元素数
    size_t totalCompressedSize;// 总压缩大小（字节）
    
    // 构造函数
    DataBuffer(size_t elements) : totalElements(elements), totalCompressedSize(0) {
        oriData = nullptr;
        cmpBytes = nullptr; 
        decData = nullptr;
    }
};

// 统一的性能分析结构 - 合并压缩和解压缩指标
struct PerformanceMetrics {
    // 基础信息
    float total_size_mb = 0;        // 原始数据大小(MB)
    float compressed_size_mb = 0;   // 压缩后大小(MB)  
    float compression_ratio = 0;    // 压缩率
    size_t chunk_size = 0;          // 块大小
    size_t total_chunks = 0;        // 总块数
    
    // 时间指标
    float comp_time = 0;            // 压缩时间(ms)
    float decomp_time = 0;          // 解压时间(ms)
    float kernel_comp_time = 0;     // 压缩核函数时间(ms)
    float kernel_decomp_time = 0;   // 解压核函数时间(ms)
    
    // 吞吐量指标
    float comp_throughput = 0;      // 压缩吞吐量(GB/s)
    float decomp_throughput = 0;    // 解压吞吐量(GB/s)
    
    // 块信息
    std::vector<size_t> chunk_compressed_sizes;  // 每块压缩大小(字节)
    std::vector<size_t> chunk_element_counts;    // 每块元素数量
    
    // 辅助方法
    void calculateMetrics(size_t originalBytes, size_t compressedBytes, 
                         float compTime, float decompTime = 0) {
        total_size_mb = originalBytes / (1024.0f * 1024.0f);
        compressed_size_mb = compressedBytes / (1024.0f * 1024.0f);
        compression_ratio = static_cast<float>(compressedBytes) / originalBytes;
        
        comp_time = compTime;
        decomp_time = decompTime;
        
        if (compTime > 0) {
            comp_throughput = (originalBytes / (1024.0f * 1024.0f * 1024.0f)) / (compTime / 1000.0f);
        }
        if (decompTime > 0) {
            decomp_throughput = (originalBytes / (1024.0f * 1024.0f * 1024.0f)) / (decompTime / 1000.0f);
        }
    }
    
    void printCompressionStats() const {
        std::cout << "===== 压缩统计 =====" << std::endl;
        std::cout << "原始大小: " << total_size_mb << " MB" << std::endl;
        std::cout << "压缩后大小: " << compressed_size_mb << " MB" << std::endl;
        std::cout << "压缩比: " << compression_ratio << std::endl;
        std::cout << "总chunk数: " << total_chunks << std::endl;
        std::cout << "压缩时间: " << comp_time << " ms" << std::endl;
        std::cout << "压缩吞吐量: " << comp_throughput << " GB/s" << std::endl;
    }
    
    void printDecompressionStats() const {
        std::cout << "===== 解压统计 =====" << std::endl;
        std::cout << "压缩大小: " << compressed_size_mb << " MB" << std::endl;
        std::cout << "解压后大小: " << total_size_mb << " MB" << std::endl;
        std::cout << "解压时间: " << decomp_time << " ms" << std::endl;
        std::cout << "解压吞吐量: " << decomp_throughput << " GB/s" << std::endl;
    }
};

// 简化的压缩器主类
class MultiStreamGDFCompressor {
private:
    static constexpr int MAX_EVENTS_PER_TYPE = 1024; // 预分配事件数量
    
public:
    MultiStreamGDFCompressor() = default;
    ~MultiStreamGDFCompressor() = default;
    
    // 压缩接口 - 返回性能指标，压缩数据存储在buffer中
    PerformanceMetrics compress(DataBuffer& buffer, size_t chunkSize, bool visualize = false);
    
    // 解压接口 - 使用压缩时的性能指标信息
    PerformanceMetrics decompress(DataBuffer& buffer, const PerformanceMetrics& compMetrics, bool visualize = false);
    
private:
    // 辅助方法
    size_t calculateOptimalChunkSize(size_t totalElements);
    void cleanupResources(cudaStream_t streams[], cudaEvent_t events[][MAX_EVENTS_PER_TYPE], 
                         double* d_in[], unsigned char* d_out[], int streamCount);
};

// 压缩实现
PerformanceMetrics MultiStreamGDFCompressor::compress(DataBuffer& buffer, size_t chunkSize, bool visualize) {
    cudaDeviceSynchronize();
    
    // 事件创建
    cudaEvent_t global_start, global_end, init_start, init_end;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_end);
    cudaEventCreate(&init_start);
    cudaEventCreate(&init_end);
    
    cudaEventRecord(init_start);
    
    // 计算chunk数量
    size_t totalChunks = (buffer.totalElements + chunkSize - 1) / chunkSize;
    if (totalChunks == 1) {
        chunkSize = buffer.totalElements;
    }
    
    // 内存分配
    unsigned int *locCmpSize;
    unsigned int *h_cmp_offset;
    cudaHostAlloc((void**)&locCmpSize, sizeof(unsigned int) * totalChunks, cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault);
    
    h_cmp_offset[0] = 0;
    bool offset_ready[totalChunks + NUM_STREAMS] = {false};
    offset_ready[0] = true;
    
    // 流和事件创建
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t evSize[MAX_EVENTS_PER_TYPE];
    cudaEvent_t evData[MAX_EVENTS_PER_TYPE];
    Stage stage[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        stage[i] = IDLE;
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming);
    }
    
    // 设备内存分配
    double *d_in[NUM_STREAMS];
    unsigned char *d_out[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMalloc(&d_in[i], chunkSize * sizeof(double));
        cudaMalloc(&d_out[i], chunkSize * sizeof(double));
    }
    
    cudaEventRecord(init_end);
    cudaEventSynchronize(init_end);
    
    // 主处理循环
    size_t processedElements = 0;
    size_t completedChunks = 0;
    size_t totalCompressedSize = 0;
    int activeStreams = 0;
    
    std::vector<int> chunkIndices(NUM_STREAMS);
    PerformanceMetrics metrics;
    
    cudaEventRecord(global_start);
    
    while (processedElements < buffer.totalElements || activeStreams > 0) {
        bool progress = false;
        
        for (int s = 0; s < NUM_STREAMS; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedElements < buffer.totalElements) {
                        size_t elementsToProcess = std::min(chunkSize, buffer.totalElements - processedElements);
                        if (elementsToProcess == 0) continue;
                        
                        progress = true;
                        chunkIndices[s] = completedChunks++;
                        
                        // H2D拷贝和压缩
                        cudaMemcpyAsync(d_in[s], buffer.oriData + processedElements,
                                      elementsToProcess * sizeof(double), cudaMemcpyHostToDevice, streams[s]);
                        
                        GDFCompressor::GDFC_compress_stream(d_in[s], d_out[s], &locCmpSize[chunkIndices[s]], 
                                                          elementsToProcess, streams[s]);
                        
                        cudaEventRecord(evSize[chunkIndices[s]], streams[s]);
                        
                        activeStreams++;
                        processedElements += elementsToProcess;
                        stage[s] = SIZE_PENDING;
                    }
                    break;
                    
                case SIZE_PENDING:
                    if (cudaEventQuery(evSize[chunkIndices[s]]) == cudaSuccess && 
                        offset_ready[chunkIndices[s]]) {
                        progress = true;
                        int idx = chunkIndices[s];
                        unsigned int compressedBytes = (locCmpSize[idx] + 7) / 8;
                        
                        // D2H拷贝结果
                        cudaMemcpyAsync(buffer.cmpBytes + h_cmp_offset[idx], d_out[s],
                                      compressedBytes, cudaMemcpyDeviceToHost, streams[s]);
                        
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        offset_ready[idx + 1] = true;
                        
                        metrics.chunk_compressed_sizes.push_back(compressedBytes);
                        // 这里需要记录元素数量，简化处理
                        size_t elements = std::min(chunkSize, buffer.totalElements - idx * chunkSize);
                        metrics.chunk_element_counts.push_back(elements);
                        
                        cudaEventRecord(evData[chunkIndices[s]], streams[s]);
                        stage[s] = DATA_PENDING;
                    }
                    break;
                    
                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIndices[s]]) == cudaSuccess) {
                        progress = true;
                        unsigned int compressedBytes = (locCmpSize[chunkIndices[s]] + 7) / 8;
                        totalCompressedSize += compressedBytes;
                        
                        stage[s] = IDLE;
                        activeStreams--;
                    }
                    break;
            }
        }
        
        if (!progress) {
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaStreamSynchronize(streams[i]);
            }
        }
    }
    
    // 等待完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(global_end);
    cudaEventSynchronize(global_end);
    
    // 计算性能指标
    float totalTime, initTime;
    cudaEventElapsedTime(&totalTime, global_start, global_end);
    cudaEventElapsedTime(&initTime, init_start, init_end);
    
    buffer.totalCompressedSize = totalCompressedSize;
    metrics.calculateMetrics(buffer.totalElements * sizeof(double), totalCompressedSize, totalTime);
    metrics.chunk_size = chunkSize;
    metrics.total_chunks = completedChunks;
    
    if (visualize) {
        metrics.printCompressionStats();
    }
    
    // 清理资源
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaEventDestroy(evSize[i]);
        cudaEventDestroy(evData[i]);
    }
    
    cudaFreeHost(locCmpSize);
    cudaFreeHost(h_cmp_offset);
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_end);
    cudaEventDestroy(init_start);
    cudaEventDestroy(init_end);
    
    return metrics;
}

// 解压实现 
PerformanceMetrics MultiStreamGDFCompressor::decompress(DataBuffer& buffer, 
                                                       const PerformanceMetrics& compMetrics, 
                                                       bool visualize) {
    cudaDeviceSynchronize();
    
    cudaEvent_t global_start, global_end;
    cudaEventCreate(&global_start);
    cudaEventCreate(&global_end);
    
    // 流和事件创建
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t evSize[NUM_STREAMS];
    cudaEvent_t evData[NUM_STREAMS];
    Stage stage[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming);
        stage[i] = IDLE;
    }
    
    // 计算最大chunk大小
    size_t maxChunkElements = *std::max_element(compMetrics.chunk_element_counts.begin(), 
                                               compMetrics.chunk_element_counts.end());
    size_t maxCompressedSize = *std::max_element(compMetrics.chunk_compressed_sizes.begin(), 
                                                compMetrics.chunk_compressed_sizes.end());
    
    // 设备内存分配
    unsigned char *d_in[NUM_STREAMS];
    double *d_out[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaMalloc(&d_in[i], maxCompressedSize);
        cudaMalloc(&d_out[i], maxChunkElements * sizeof(double));
    }
    
    // 计算压缩数据偏移
    std::vector<size_t> compressedOffsets(compMetrics.total_chunks + 1, 0);
    for (size_t i = 0; i < compMetrics.total_chunks; ++i) {
        compressedOffsets[i + 1] = compressedOffsets[i] + compMetrics.chunk_compressed_sizes[i];
    }
    
    // 主处理循环
    size_t processedChunks = 0;
    size_t processedElements = 0;
    int activeStreams = 0;
    std::vector<size_t> streamChunkIds(NUM_STREAMS);
    std::vector<size_t> streamOutputOffsets(NUM_STREAMS);
    
    cudaEventRecord(global_start);
    
    while (processedChunks < compMetrics.total_chunks || activeStreams > 0) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            switch (stage[s]) {
                case IDLE:
                    if (processedChunks < compMetrics.total_chunks) {
                        activeStreams++;
                        size_t chunkId = processedChunks++;
                        size_t currentElements = compMetrics.chunk_element_counts[chunkId];
                        size_t currentCompressedSize = compMetrics.chunk_compressed_sizes[chunkId];
                        
                        streamChunkIds[s] = chunkId;
                        streamOutputOffsets[s] = processedElements;
                        processedElements += currentElements;
                        
                        // H2D拷贝压缩数据
                        cudaMemcpyAsync(d_in[s], buffer.cmpBytes + compressedOffsets[chunkId],
                                      currentCompressedSize, cudaMemcpyHostToDevice, streams[s]);
                        
                        // 解压
                        GDFDecompressor GDFD;
                        GDFD.GDFC_decompress_stream_optimized(d_out[s], d_in[s], currentElements,
                                                            currentCompressedSize, streams[s]);
                        
                        // D2H拷贝解压结果
                        cudaMemcpyAsync(buffer.decData + streamOutputOffsets[s], d_out[s],
                                      currentElements * sizeof(double), cudaMemcpyDeviceToHost, streams[s]);
                        
                        cudaEventRecord(evData[s], streams[s]);
                        stage[s] = DATA_PENDING;
                    }
                    break;
                    
                case DATA_PENDING:
                    if (cudaEventQuery(evData[s]) == cudaSuccess) {
                        stage[s] = IDLE;
                        activeStreams--;
                    }
                    break;
            }
        }
    }
    
    // 等待完成
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(global_end);
    cudaEventSynchronize(global_end);
    
    // 计算性能指标
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start, global_end);
    
    PerformanceMetrics decompMetrics = compMetrics; // 复制压缩时的信息
    decompMetrics.decomp_time = totalTime;
    decompMetrics.decomp_throughput = (buffer.totalElements * sizeof(double) / (1024.0f * 1024.0f * 1024.0f)) / (totalTime / 1000.0f);
    
    if (visualize) {
        decompMetrics.printDecompressionStats();
    }
    
    // 清理资源
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_in[i]);
        cudaFree(d_out[i]);
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(evSize[i]);
        cudaEventDestroy(evData[i]);
    }
    
    cudaEventDestroy(global_start);
    cudaEventDestroy(global_end);
    
    return decompMetrics;
}

// 辅助函数
size_t MultiStreamGDFCompressor::calculateOptimalChunkSize(size_t totalElements) {
    size_t chunkSize = 1;
    size_t temp = totalElements / NUM_STREAMS;
    while (chunkSize <= MAX_NUMS_PER_CHUNK && chunkSize <= temp) {
        chunkSize *= 2;
    }
    return chunkSize;
}