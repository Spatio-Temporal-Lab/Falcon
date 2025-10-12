#include "data/dataset_utils.hpp"
#include "Faclon_decompressor.cuh"
#include "Faclon_compressor.cuh"

// 流水线总体性能分析
struct PipelineAnalysis {
    float total_size = 0;           // 数据总大小
    float compression_ratio = 0;    // 压缩率
    float comp_time=0;              // 压缩时间
    float decomp_time=0;            // 解压缩时间
    float comp_throughout=0;        // 压缩吞吐量
    float decomp_throughout=0;      // 解压缩吞吐量

    float total_compressed_size = 0; //压缩后总大小
    size_t chunk_size = 0; // 记录块大小
};

// 数据预处理函数
struct ProcessedData {
    double *oriData;
    unsigned char *cmpBytes;
    unsigned int *cmpSize;
    double *decData;
    size_t nbEle;
};

enum Stage { IDLE, SIZE_PENDING, DATA_PENDING };

// 压缩结果结构体
struct CompressionResult {
    PipelineAnalysis analysis;
    std::vector<size_t> chunkSizes;        // 每个chunk的压缩大小（字节）
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                    // 总chunk数量
    // CompressionInfo cmpInfo;
};

// 更新的CompressedData结构体
struct CompressedData {
    unsigned char* cmpBytes;                // 压缩数据缓冲
    size_t totalCompressedSize;             // 总压缩大小
    size_t totalElements;                   // 原始数据元素总数
    std::vector<size_t> chunkSizes;         // 每个chunk的压缩大小（字节）
    std::vector<size_t> chunkElementCounts; // 每个chunk的元素数量
    size_t totalChunks;                     // 总chunk数量
};
