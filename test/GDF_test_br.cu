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
#include "Faclon_decompressor.cuh"
#include "Faclon_compressor.cuh"
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


// 流水线总体性能分析
struct PipelineAnalysis {
    float total_size = 0;           // 数据总大小
    float compression_ratio = 0;    // 压缩率
    float comp_time=0;              // 压缩时间
    float decomp_time=0;            // 解压缩时间
    float comp_throughout=0;        // 压缩吞吐量
    float decomp_throughout=0;      // 解压缩吞吐量
    
    float kernal_comp = 0;          // 压缩核函数时间
    float kernal_decomp = 0;        // 解压核函数时间

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
        // printf("生成 %zu 个元素的测试数据 (模式: %d)\n", generate_size, pattern_type);
        data = generate_test_data(generate_size, pattern_type);
        result.nbEle = generate_size;
    } else if (!source_path.empty()) {
        // 从文件读取数据
        // printf("从文件加载数据: %s\n", source_path.c_str());
        data = read_data(source_path);
        result.nbEle = data.size();
    } else {
        printf("错误: 未指定数据源\n");
        result.nbEle = 0;
        result.oriData = nullptr;
        result.cmpBytes = nullptr;
        return result;
    }
    if(result.nbEle <= 0)
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

    printf("totalChunks: %zu\n",totalChunks);
    // 主机侧内存分配
    // unsigned int *locCmpSize;
    // cudaCheckError(cudaHostAlloc((void**)&locCmpSize, sizeof(unsigned int) * totalChunks, cudaHostAllocDefault));
    // 分配时添加保护
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
    sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));  // +2 for guards

    // 设置保护值
    locCmpSize[0] = 0xDEADBEEF;  // 前保护
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;  // 后保护


    
    unsigned int *h_cmp_offset;
    // cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * totalChunks + 1, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    // 初始化偏移量数组
    h_cmp_offset[0] = 0;

    bool of_rd[totalChunks+ NUM_STREAMS]={0};//内存偏移准备完成信号
    of_rd[0]=true;

    // 新增：记录每个chunk信息的数组
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, sizeof(size_t) * totalChunks, cudaHostAllocDefault));

    // 用于存储每个chunk的压缩信息
    // std::vector<size_t> chunkSizes;
    // std::vector<size_t> chunkElementCountsVec;
    // chunkSizes.reserve(totalChunks);
    // chunkElementCountsVec.reserve(totalChunks);

    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);

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
    while (processedEle < data.nbEle || active > 0) {
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
                        chunkIDX[s]=completedChunks;//记录当前处理的chunks
                        completedChunks++;
                        // 记录当前流处理的元素数量
                        chunkElementCounts[chunkIDX[s]] = todo;

                        // 异步 H→D 拷贝
                        cudaCheckError(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(double), // 修正：使用实际元素数
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        // cudaStreamSynchronize(streams[s]);

                        // cudaCheckError(cudaEventRecord(kernal_start[chunkIDX[s]], streams[s]));
                        // 调用你的压缩接口（内部处理所有临时内存）
                        // GDFCompressor::GDFC_compress_stream(
                        //     d_in[s],
                        //     d_out[s],
                        //     &locCmpSize[chunkIDX[s]],  // 直接传递压缩大小指针
                        //     todo,
                        //     streams[s]);
                        // 使用时偏移
                        GDFCompressor_opt::GDFC_compress_br(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],  // +1 因为有前保护
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
                        if (locCmpSize[0] != 0xDEADBEEF || locCmpSize[totalChunks + 1] != 0xCAFEBABE) {
                            printf("错误：内存保护值被覆写！\n");
                        }
                        progress=1;
                        // 计算压缩后的字节大小
                        int idx=chunkIDX[s];
                        // printf("idx: %d,stream: %d SIZE\n",idx,s);
                        unsigned int compressedBits = locCmpSize[idx+1];//因为有前缀保护
                        unsigned int compressedBytes = (compressedBits + 7) / 8;

                        // 异步 D→H 拷贝结果
                        cudaCheckError(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,  // 使用实际压缩大小
                            cudaMemcpyDeviceToHost,
                            streams[s]));

                        // 更新偏移量
                        // if (idx == 11) {
                        //     printf("Stream %d, Chunk 11: 原始locCmpSize=%u (bits或bytes?)\n", s, compressedBits);
                        // }
                        h_cmp_offset[idx+1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx+1]=true;//给下一个准备好了偏移
                        // printf("idx: %d,stream: %d offready\n",idx+1,s);
                        // chunkSizes.push_back(compressedBytes);
                        // chunkElementCountsVec.push_back(chunkElementCounts[s]);
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                        // if (1) {  // 只打印前几个chunk
                        //     printf("压缩 Chunk %d: cmp_offset=%u, cmp_size=%u bytes, elements=%zu，elements_off=%zu\n", 
                        //         idx, h_cmp_offset[idx], compressedBytes, chunkElementCounts[idx],chunkSize*idx);
                        // }
                    }
                    break;

                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        // 累计压缩大小
                        // printf("idx: %d stream: %d DATA\n",chunkIDX[s],s);
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s]+1] + 7) / 8;//+1因为有前缀保护
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
        cudaCheckError(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError(cudaEventDestroy(evSize[i]));
        cudaCheckError(cudaEventDestroy(evData[i]));
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
PipelineAnalysis execute_decompression_pipeline(const CompressionResult& compResult, ProcessedData &decompData, bool visualize = false) {
    CompressedData compData = createCompressedData(compResult, decompData);
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
    for (size_t i = 0; i < compData.totalChunks - 1; ++i) {
        compressedDataOffsets[i+1] = compressedDataOffsets[i] + compData.chunkSizes[i];  // 累加偏移
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
                        GDFDecompressor_opt GDFD;
                        GDFD.GDFC_decompress_stream_optimized(
                        d_out[s],                    // 解压输出
                        d_in[s],                     // 压缩输入
                        currentChunkElements,        // 原始元素数量
                        currentChunkCompressedSize,  // 压缩数据大小
                        streams[s]);

                        // 记录压缩数据拷贝和解压完成事件
                        cudaCheckError(cudaEventRecord(evSize[s], streams[s]));

                        // stage[s] = SIZE_PENDING;
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
                        // if (1) {
                        //     printf("解压 Chunk %zu: input_offset=%zu, output_offset=%zu, size=%zu bytes, elements=%zu\n",
                        //         chunkId, compressedDataOffset, outputOffset, 
                        //         currentChunkCompressedSize, currentChunkElements);
                        // }
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
    result.total_compressed_size = compData.totalCompressedSize/1024.0/1024.0;
    result.total_size = totalDecompSize / 1024.0 / 1024.0;
    result.decomp_time = totalTime;
    result.decomp_throughout=(totalDecompSize  / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    // size_t expectedSize = compData.totalElements * sizeof(double);
    // if (totalDecompSize != expectedSize) {
    //     printf("警告：解压缩大小不匹配！期望: %zu, 实际: %zu\n", 
    //         expectedSize, totalDecompSize);
    // }
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
    for(int z=0,tmp=0,k=0;z<3;z++)//循环三次
    {   
        for(size_t i=0,j=tmp;i<3;i++)//找到不等的然后输出，然后跳到下一个block
        {
            while(decompData.oriData[j]==decompData.decData[j]&&j<processedElements)//找到不等的值
            {
                j++;
                if(j>=processedElements)
                {
                    // printf("success!!\n");   
                    break;
                }
            }
            if(j>=processedElements)
            {
                printf("success!!\n");   
                break;
            }
            while(tmp<=j)
            {
                tmp+=compData.chunkElementCounts[k];
                k++;
            }
            // printf("chunk:%d,idx: %d ,ori: %.16f , dec: %.16f \n",k-1,j,decompData.oriData[j],decompData.decData[j]);
            j++;
        }
    }
    
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
// 单独测试chunk 11
void test_chunk(int idx,ProcessedData &data) {
    size_t chunk_start = idx * 262144;
    size_t chunk_size = 262144;
    if(data.nbEle<chunk_start)
    {
        return;
    }
    // 分配设备内存
    double *d_in, *d_out_decomp;
    unsigned char *d_out_comp;
    unsigned int h_comp_size;
    
    cudaMalloc(&d_in, chunk_size * sizeof(double));
    cudaMalloc(&d_out_comp, chunk_size * sizeof(double));
    cudaMalloc(&d_out_decomp, chunk_size * sizeof(double));
    
    // 拷贝数据
    cudaMemcpy(d_in, data.oriData + chunk_start, 
               chunk_size * sizeof(double), cudaMemcpyHostToDevice);
    
    // 压缩
    GDFCompressor_opt::GDFC_compress_stream(
        d_in, d_out_comp, &h_comp_size, chunk_size, 0);
    cudaDeviceSynchronize();
    
    printf("Chunk %d 单独测试: 压缩大小=%u bytes\n",idx, (h_comp_size + 7) / 8);
    
    // 解压
    GDFDecompressor_opt GDFD;
    GDFD.GDFC_decompress_stream_optimized(
        d_out_decomp, d_out_comp, chunk_size, 
        (h_comp_size + 7) / 8, 0);
    cudaDeviceSynchronize();
    
    // 验证
    double *h_decomp = new double[chunk_size];
    cudaMemcpy(h_decomp, d_out_decomp, 
               chunk_size * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (size_t i = 0; i < 10; i++) {
        printf("单独测试[%zu]: ori=%.10f, dec=%.10f\n", 
               chunk_start+i, data.oriData[chunk_start + i], h_decomp[i]);
    }
    
    delete[] h_decomp;
    cudaFree(d_in);
    cudaFree(d_out_comp);
    cudaFree(d_out_decomp);
}
// 输出块大小与运行时间关系的CSV文件
// void output_blocksize_timing_csv(const std::vector<PipelineAnalysis> &results, const std::string &filename) {
//     std::ofstream csv_file(filename, std::ios::app);
//     if (!csv_file.is_open()) {
//         std::cerr << "无法创建CSV文件: " << filename << std::endl;
//         return;
//     }
//     bool write_header = csv_file.tellp() == 0;
//     if (write_header) {
//         // 写入CSV头
//         csv_file << title << "\n数据量(MB),块大小(KB),平均H2D时间(ms),平均压缩时间(ms),平均D2H时间(ms),加速比,总时间,吞吐量比例,压缩率\n";
//     } else {
//         csv_file << "\n" << title << "\n数据量(MB),块大小(KB),平均H2D时间(ms),平均压缩时间(ms),平均D2H时间(ms),加速比,总时间,吞吐量比例,压缩率\n";
//     }

//     // 写入每个块大小的数据
//     for (const auto &result: results) {
//         csv_file << result.total_size << ","
//                 << (result.chunk_size * sizeof(double) / 1024) << ","
//                 << result.avg_h2d << ","
//                 << result.avg_comp << ","
//                 << result.avg_d2h << ","
//                 << result.speedup << ","
//                 << result.end_time << ","
//                 << (result.avg_comp > 0 ? (result.avg_h2d + result.avg_d2h) / result.avg_comp : 0) << ","
//                 << result.compression_ratio << "\n";
//     }

//     csv_file.close();
//     std::cout << "已将块大小与运行时间关系保存到 " << filename << std::endl;
// }

// 可视化块大小与阶段时间的关系
// void visualize_stage_timing_relationship(const std::vector<PipelineAnalysis> &results) {
//     printf("\n===== 块大小与运行时间关系分析 =====\n");
//     printf("总大小(MB) \t块大小(KB) \tH2D(ms) \tComp(ms) \tD2H(ms) \t比例(IO/Comp) \t总时间(ms) \t加速比 \t压缩率 \n");
//     printf("---------------------------------------------------------------------------\n");

//     for (const auto &result: results) {
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
//                result.compression_ratio);
//     }
// }
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
    size_t chunkSize=1025;
    size_t temp=nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
    //可用的显存
    // size_t availableMemory = getAvailableGPUMemory();
    size_t availableMemory, totalMem;
    cudaMemGetInfo(&availableMemory, &totalMem);
    size_t limit=availableMemory/(4 * NUM_STREAMS * sizeof(double) * 2);
    //最多同时有16流 chunkSize*NUM_STREAMS*8*sizeof(double) * 2<availableMemory/8*
    while(chunkSize<=limit//MAX_NUMS_PER_CHUNK 
            && chunkSize<=temp)
    {
        chunkSize*=2;
    }
    // chunkSize=chunkSize/2;
    chunkSize=chunkSize>1025?chunkSize:1025;
    printf("chunkSize:%zu\n",chunkSize);
    return chunkSize;
}

CompressionInfo test_compression(ProcessedData data, size_t chunkSize)
{
    // test_chunk(12,data);
    // test_chunk(17,data);
    // test_chunk(18,data);
    // cudaDeviceReset();

    CompressionResult compResult = execute_pipeline(data, chunkSize, false);
    cudaDeviceSynchronize(); 

    PipelineAnalysis decompAnalysis = execute_decompression_pipeline(compResult, data, false);
    cudaDeviceSynchronize(); 


    return CompressionInfo{
        decompAnalysis.total_size,
        decompAnalysis.total_compressed_size,
        compResult.analysis.compression_ratio,
        0,
        compResult.analysis.comp_time,
        compResult.analysis.comp_throughout,
        0,
        decompAnalysis.decomp_time,
        decompAnalysis.decomp_throughout};
}
// 主要测试函数 - 支持文件路径或生成数据
int test(const std::string &file_path = "", size_t data_size_mb = 0, int pattern_type = 0) {
    // warmup();
    cudaDeviceReset();

    if (!file_path.empty()) {
        // size_t chunkSize=1;
        // ProcessedData data = prepare_data(file_path);
        // chunkSize=setChunk(data.nbEle);
        // CompressionInfo a;
        // for(int i=0;i<3;i++)
        // {
        //     auto tmp=test_compression(data,chunkSize);
        //     a+=tmp;
        //     if(1)//(tmp.compression_ratio!=(a.compression_ratio/(i+1)))
        //     {
        //         printf("a:%.6f,get:%.6f\n",a.compression_ratio/(i+1),tmp.compression_ratio);
        //     }
        // }
        std::vector<double> file_data = read_data(file_path);
        size_t nbEle = file_data.size();
        
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            // 每次循环重置GPU并重新准备数据
            cudaDeviceReset();
            
            // 重新准备CUDA资源
            ProcessedData data = prepare_data(file_path);
            
            // 计算chunk大小
            size_t chunkSize = setChunk(data.nbEle);
            
            // 执行测试
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
            
            // 清理本次迭代的资源
            cleanup_data(data);
        }
        a=a/3;
        a.print();
        // cleanup_data(data);
        return 0;
    } else if (data_size_mb > 0) {
        size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
        
        // ProcessedData data = prepare_data("", nbEle, pattern_type);
        // chunkSize=setChunk(data.nbEle);
        // CompressionInfo a;
        // for(int i=0;i<3;i++)
        // {
        //     a+=test_compression(data,chunkSize);
        // }
        // a=a/3;
        // a.print();
        // cleanup_data(data);
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            // 每次循环重置GPU并重新准备数据
            cudaDeviceReset();
            
            // 重新准备CUDA资源
            ProcessedData data = prepare_data("", nbEle, pattern_type);
            
            // 计算chunk大小
            size_t chunkSize = setChunk(data.nbEle);
            
            // 执行测试
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
            
            // 清理本次迭代的资源
            cleanup_data(data);
        }
        a=a/3;
        a.print();
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
    } else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }

    return 0;
}

