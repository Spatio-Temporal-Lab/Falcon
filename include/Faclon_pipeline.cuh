// #include "data/dataset_utils.hpp"
// #include "Faclon_decompressor.cuh"
// #include "Faclon_compressor.cuh"

// Faclon_pipeline.cuh
// 流水线压缩和解压缩类
// 路径: Falcon\include\Faclon_pipeline.cuh

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cstddef>
#include "Faclon_compressor.cuh"
#include "Faclon_decompressor.cuh"

// 流水线性能分析结构体
struct PipelineAnalysis {
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
struct CompressionResult {
    PipelineAnalysis analysis;
    std::vector<size_t> chunkSizes;        // 每个chunk的压缩大小(字节)
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                    // 总chunk数量
};

// 压缩数据结构体
struct CompressedData {
    unsigned char* cmpBytes;                // 压缩数据缓冲
    size_t totalCompressedSize;             // 总压缩大小
    size_t totalElements;                   // 原始数据元素总数
    std::vector<size_t> chunkSizes;         // 每个chunk的压缩大小(字节)
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                     // 总chunk数量
};

// 处理数据结构体
struct ProcessedData {
    double *oriData;
    unsigned char *cmpBytes;
    unsigned int *cmpSize;
    double *decData;
    size_t nbEle;
};

// 流水线阶段枚举
enum Stage { 
    IDLE,           // 空闲
    SIZE_PENDING,   // 等待大小信息
    DATA_PENDING    // 等待数据传输
};


// 流水线压缩解压缩类
class FaclonPipeline {
public:
    FaclonPipeline()
    {
        NUM_STREAMS = 16;
    }
    FaclonPipeline(int numStreams);
    ~FaclonPipeline();

    // 执行压缩流水线
    CompressionResult executeCompressionPipeline(
        ProcessedData &data, 
        size_t chunkSize);

    // 执行压缩流水线(指定流数量)
    CompressionResult executeCompressionPipeline(
        ProcessedData &data, 
        size_t chunkSize,
        int numStreams);

    // 执行解压缩流水线
    PipelineAnalysis executeDecompressionPipeline(
        const CompressionResult& compResult,
        ProcessedData &decompData,
        bool visualize = true);

    // 执行解压缩流水线(指定流数量)
    PipelineAnalysis executeDecompressionPipeline(
        const CompressionResult& compResult,
        ProcessedData &decompData,
        int numStreams,
        bool visualize = true);

    // 辅助函数:从压缩结果创建CompressedData结构
    static CompressedData createCompressedData(
        const CompressionResult& compResult,
        const ProcessedData& originalData);

    // 设置流数量
    void setNumStreams(int numStreams) { 
        NUM_STREAMS = numStreams; 
    }

    // 获取当前流数量
    int getNumStreams() const { 
        return NUM_STREAMS; 
    }

//消融实验一：核函数

    //消融实验1.1：全稀疏与全稠密 （解压与标准版一致）
        //  NoPack

    CompressionResult executeCompressionPipelineNoPack(
    ProcessedData &data, 
    size_t chunkSize);

        //  Spare

    CompressionResult executeCompressionPipelineSpare(
    ProcessedData &data, 
    size_t chunkSize);

    //消融实验1.2：暴力计算
    
        //  Br
    
    CompressionResult executeCompressionPipelineBr(
    ProcessedData &data, 
    size_t chunkSize);   

//消融实验三：流水线

    //阻塞

    CompressionResult executeCompressionPipelineBlock(
        ProcessedData &data,
        size_t chunkSize);

    //非阻塞

    CompressionResult executeCompressionPipelineNoBlock(
        ProcessedData &data,
        size_t chunkSize);

private:
    int NUM_STREAMS;  // CUDA流数量

    // 内部实现函数
    CompressionResult executeCompressionPipelineImpl(
        ProcessedData &data,
        size_t chunkSize,
        int numStreams);

    PipelineAnalysis executeDecompressionPipelineImpl(
        const CompressionResult& compResult,
        ProcessedData &decompData,
        int numStreams,
        bool visualize);
    
// 消融实验（核函数）

    // NOPACK
    CompressionResult executeCompressionPipelineImpl_NoPack(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams);


    // SPARE
    CompressionResult executeCompressionPipelineImpl_Spare(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams);

    // BR
    CompressionResult executeCompressionPipelineImpl_Br(
    ProcessedData &data,
    size_t chunkSize,
    int numStreams);

};

// CUDA错误检查宏
#ifndef cudaCheckError
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

// 清理资源函数
void cleanup_data(ProcessedData &data);

