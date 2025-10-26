// Falcon_float_pipeline.cuh
// 32位浮点数流水线压缩和解压缩类
// 路径: Falcon\include\Falcon_float_pipeline.cuh

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "Falcon_float_compressor.cuh"
#include "Falcon_float_decompressor.cuh"

// 流水线性能分析结构体
struct PipelineAnalysis_32 {
    float total_size = 0;           // 数据总大小(MB)
    float compression_ratio = 0;    // 压缩率
    float comp_time = 0;            // 压缩时间(ms)
    float decomp_time = 0;          // 解压缩时间(ms)
    float comp_throughout = 0;      // 压缩吞吐量(GB/s)
    float decomp_throughout = 0;    // 解压缩吞吐量(GB/s)
    float total_compressed_size = 0; // 压缩后总大小(MB)
    size_t chunk_size = 0;          // 块大小
};

// 压缩结果结构体
struct CompressionResult_32 {
    PipelineAnalysis_32 analysis;
    std::vector<size_t> chunkSizes;        // 每个chunk的压缩大小(字节)
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                    // 总chunk数量
};

// 压缩数据结构体
struct CompressedData_32 {
    unsigned char* cmpBytes;                // 压缩数据缓冲
    size_t totalCompressedSize;             // 总压缩大小
    size_t totalElements;                   // 原始数据元素总数
    std::vector<size_t> chunkSizes;         // 每个chunk的压缩大小(字节)
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                     // 总chunk数量
};

// 处理数据结构体
struct ProcessedData_32 {
    float *oriData;
    unsigned char *cmpBytes;
    unsigned int *cmpSize;
    float *decData;
    size_t nbEle;
};

// 流水线阶段枚举
enum Stage_32 { 
    IDLE_32,           // 空闲
    SIZE_PENDING_32,   // 等待大小信息
    DATA_PENDING_32    // 等待数据传输
};

// 流水线压缩解压缩类
class FalconPipeline_32 {
public:
    FalconPipeline_32() {
        NUM_STREAMS = 16;
    }
    
    FalconPipeline_32(int numStreams) : NUM_STREAMS(numStreams) {}
    ~FalconPipeline_32();

    // 执行压缩流水线
    CompressionResult_32 executeCompressionPipeline(
        ProcessedData_32 &data, 
        size_t chunkSize);

    // 执行压缩流水线(指定流数量)
    CompressionResult_32 executeCompressionPipeline(
        ProcessedData_32 &data, 
        size_t chunkSize,
        int numStreams);

    // 执行解压缩流水线
    PipelineAnalysis_32 executeDecompressionPipeline(
        const CompressionResult_32& compResult,
        ProcessedData_32 &decompData,
        bool visualize = true);

    // 执行解压缩流水线(指定流数量)
    PipelineAnalysis_32 executeDecompressionPipeline(
        const CompressionResult_32& compResult,
        ProcessedData_32 &decompData,
        int numStreams,
        bool visualize = true);

    // 辅助函数:从压缩结果创建CompressedData结构
    static CompressedData_32 createCompressedData(
        const CompressionResult_32& compResult,
        const ProcessedData_32& originalData);

    // 设置流数量
    void setNumStreams(int numStreams) { 
        NUM_STREAMS = numStreams; 
    }

    // 获取当前流数量
    int getNumStreams() const { 
        return NUM_STREAMS; 
    }

private:
    int NUM_STREAMS;  // CUDA流数量

    // 内部实现函数
    CompressionResult_32 executeCompressionPipelineImpl(
        ProcessedData_32 &data,
        size_t chunkSize,
        int numStreams);

    PipelineAnalysis_32 executeDecompressionPipelineImpl(
        const CompressionResult_32& compResult,
        ProcessedData_32 &decompData,
        int numStreams,
        bool visualize);
};

// CUDA错误检查宏
#ifndef cudaCheckError_32
#define cudaCheckError_32(ans) { gpuAssert_32((ans), __FILE__, __LINE__); }
inline void gpuAssert_32(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

// 清理资源函数
void cleanup_data_32(ProcessedData_32 &data);