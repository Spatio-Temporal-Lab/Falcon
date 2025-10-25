// Falcon_float_pipeline.cu
// 32位浮点数流水线压缩和解压缩实现
// 路径: Falcon\src\gpu\Falcon_float_pipeline.cu

#include "Falcon_float_pipeline.cuh"
#include <algorithm>
#include <iostream>
#include <cstring>

// 析构函数
FalconPipeline_32::~FalconPipeline_32() {
}

// 辅助函数:从压缩结果创建CompressedData结构
CompressedData_32 FalconPipeline_32::createCompressedData(
    const CompressionResult_32& compResult,
    const ProcessedData_32& originalData) {
    
    CompressedData_32 compData;
    compData.cmpBytes = originalData.cmpBytes;
    compData.totalCompressedSize = compResult.analysis.total_compressed_size;
    compData.totalElements = originalData.nbEle;
    compData.chunkSizes = compResult.chunkSizes;
    compData.chunkElementCounts = compResult.chunkElementCounts;
    compData.totalChunks = compResult.totalChunks;
    
    return compData;
}

// 清理资源函数
void cleanup_data_32(ProcessedData_32 &data) {
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

// 执行压缩流水线(使用类成员变量的流数量)
CompressionResult_32 FalconPipeline_32::executeCompressionPipeline(
    ProcessedData_32 &data,
    size_t chunkSize) {
    
    return executeCompressionPipelineImpl(data, chunkSize, NUM_STREAMS);
}

// 执行压缩流水线(指定流数量)
CompressionResult_32 FalconPipeline_32::executeCompressionPipeline(
    ProcessedData_32 &data,
    size_t chunkSize,
    int numStreams) {
    
    return executeCompressionPipelineImpl(data, chunkSize, numStreams);
}

// 压缩流水线内部实现
CompressionResult_32 FalconPipeline_32::executeCompressionPipelineImpl(
    ProcessedData_32 &data,
    size_t chunkSize,
    int numStreams) {
    
    cudaDeviceSynchronize();
    
    // 创建时间线记录事件
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    cudaEvent_t init_start_event, init_end_event;
    cudaEventCreate(&init_start_event);
    cudaEventCreate(&init_end_event);
    
    // 初始化
    cudaDeviceSynchronize();
    cudaEventRecord(init_start_event);
    
    // 计算总chunk数量
    size_t totalChunks = (data.nbEle + chunkSize - 1) / chunkSize;
    if(totalChunks == 1) {
        chunkSize = data.nbEle;
    }
    
    printf("totalChunks: %zu\n", totalChunks);
    
    // 主机侧内存分配
    unsigned int *locCmpSize;
    cudaCheckError_32(cudaHostAlloc((void**)&locCmpSize, 
        sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));
    
    // 设置保护值
    locCmpSize[0] = 0xDEADBEEF;
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;
    
    unsigned int *h_cmp_offset;
    cudaCheckError_32(cudaHostAlloc((void**)&h_cmp_offset, 
        sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    h_cmp_offset[0] = 0;
    
    bool *of_rd = new bool[totalChunks + numStreams]();
    of_rd[0] = true;
    
    size_t *chunkElementCounts;
    cudaCheckError_32(cudaHostAlloc((void**)&chunkElementCounts, 
        sizeof(size_t) * totalChunks, cudaHostAllocDefault));
    
    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);
    
    // 创建流池和事件
    const int MAX_EVENTS_PER_TYPE = totalChunks + numStreams;
    
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    cudaEvent_t *evData = new cudaEvent_t[MAX_EVENTS_PER_TYPE];
    Stage_32 *stage = new Stage_32[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE_32;
    }
    
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError_32(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError_32(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }
    
    // 为每个流分配固定设备缓冲
    float **d_in = new float*[numStreams];
    unsigned char **d_out = new unsigned char*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaMalloc(&d_in[i], chunkSize * sizeof(float)));
        cudaCheckError_32(cudaMalloc(&d_out[i], chunkSize * sizeof(float)));
    }
    
    cudaEventRecord(init_end_event);
    cudaEventSynchronize(init_end_event);
    
    // 主循环:轮询stream直到数据处理完
    size_t processedEle = 0;
    int active = 0;
    size_t totalCmpSize = 0;
    size_t completedChunks = 0;
    
    cudaDeviceSynchronize();
    cudaEventRecord(global_start_event);
    
    std::vector<int> chunkIDX(numStreams);
    
    while (processedEle < data.nbEle || active > 0) {
        int progress = 0;
        
        for (int s = 0; s < numStreams; ++s) {
            switch (stage[s]) {
                case IDLE_32:
                    if (processedEle < data.nbEle) {
                        size_t todo = std::min(chunkSize, data.nbEle - processedEle);
                        if(todo == 0) continue;
                        
                        progress = 1;
                        chunkIDX[s] = completedChunks;
                        completedChunks++;
                        
                        chunkElementCounts[chunkIDX[s]] = todo;
                        
                        // 异步H→D拷贝
                        cudaCheckError_32(cudaMemcpyAsync(
                            d_in[s],
                            data.oriData + processedEle,
                            todo * sizeof(float),
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        // 调用压缩接口
                        FalconCompressor::Falcon_compress_stream(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],
                            todo,
                            streams[s]);
                        
                        active += 1;
                        processedEle += todo;
                        stage[s] = SIZE_PENDING_32;
                        cudaCheckError_32(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                    
                case SIZE_PENDING_32:
                    if(cudaEventQuery(evSize[chunkIDX[s]]) == cudaSuccess && 
                       of_rd[chunkIDX[s]]) {
                        
                        if (locCmpSize[0] != 0xDEADBEEF || 
                            locCmpSize[totalChunks + 1] != 0xCAFEBABE) {
                            printf("错误:内存保护值被覆写!\n");
                        }
                        
                        progress = 1;
                        int idx = chunkIDX[s];
                        
                        unsigned int compressedBits = locCmpSize[idx + 1];
                        unsigned int compressedBytes = (compressedBits + 7) / 8;
                        
                        // 异步D→H拷贝结果
                        cudaCheckError_32(cudaMemcpyAsync(
                            data.cmpBytes + h_cmp_offset[idx],
                            d_out[s],
                            compressedBytes,
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        h_cmp_offset[idx + 1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx + 1] = true;
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        
                        cudaCheckError_32(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING_32;
                    }
                    break;
                    
                case DATA_PENDING_32:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s] + 1] + 7) / 8;
                        totalCmpSize += compressedBytes;
                        progress = 1;
                        stage[s] = IDLE_32;
                        active -= 1;
                    }
                    break;
            }
        }
        
        // 避免忙等待
        if (!progress) {
            for (int i = 0; i < numStreams; i++) {
                cudaCheckError_32(cudaStreamSynchronize(streams[i]));
            }
        }
    }
    
    // 等待所有流完成
    for (int i = 0; i < std::min(numStreams, (int)totalChunks); i++) {
        cudaCheckError_32(cudaStreamSynchronize(streams[i]));
        cudaCheckError_32(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    // 计算总时间
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    // 计算压缩比
    double compressionRatio = totalCmpSize / static_cast<double>(data.nbEle * sizeof(float));
    
    // 创建分析结果
    PipelineAnalysis_32 analysis;
    analysis.compression_ratio = compressionRatio;
    analysis.total_compressed_size = totalCmpSize;
    analysis.total_size = data.nbEle * sizeof(float) / 1024.0 / 1024.0;
    analysis.comp_time = totalTime;
    analysis.comp_throughout = (data.nbEle * sizeof(float) / 1024.0 / 1024.0 / 1024.0) / 
                              (totalTime / 1000.0);
    analysis.chunk_size = chunkSize;
    *data.cmpSize = totalCmpSize;
    
    // 清理设备内存
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaFree(d_out[i]));
        cudaCheckError_32(cudaFree(d_in[i]));
    }
    
    // 清理流和事件
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaStreamDestroy(streams[i]));
    }
    for (int i = 0; i < MAX_EVENTS_PER_TYPE; i++) {
        cudaCheckError_32(cudaEventDestroy(evSize[i]));
        cudaCheckError_32(cudaEventDestroy(evData[i]));
    }
    
    // 释放主机内存
    cudaCheckError_32(cudaFreeHost(locCmpSize));
    cudaCheckError_32(cudaFreeHost(h_cmp_offset));
    cudaCheckError_32(cudaFreeHost(chunkElementCounts));
    delete[] of_rd;
    
    // 销毁全局事件
    cudaCheckError_32(cudaEventDestroy(global_start_event));
    cudaCheckError_32(cudaEventDestroy(global_end_event));
    cudaCheckError_32(cudaEventDestroy(init_start_event));
    cudaCheckError_32(cudaEventDestroy(init_end_event));
    
    // 释放动态分配的数组
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    // 创建并返回完整结果
    CompressionResult_32 result;
    result.analysis = analysis;
    result.chunkSizes = std::move(chunkSizes);
    result.chunkElementCounts = std::move(chunkElementCountsVec);
    result.totalChunks = completedChunks;
    
    return result;
}

// 执行解压缩流水线(使用类成员变量的流数量)
PipelineAnalysis_32 FalconPipeline_32::executeDecompressionPipeline(
    const CompressionResult_32& compResult,
    ProcessedData_32 &decompData,
    bool visualize) {
    
    return executeDecompressionPipelineImpl(compResult, decompData, NUM_STREAMS, visualize);
}

// 执行解压缩流水线(指定流数量)
PipelineAnalysis_32 FalconPipeline_32::executeDecompressionPipeline(
    const CompressionResult_32& compResult,
    ProcessedData_32 &decompData,
    int numStreams,
    bool visualize) {
    
    return executeDecompressionPipelineImpl(compResult, decompData, numStreams, visualize);
}

// 解压缩流水线内部实现
PipelineAnalysis_32 FalconPipeline_32::executeDecompressionPipelineImpl(
    const CompressionResult_32& compResult,
    ProcessedData_32 &decompData,
    int numStreams,
    bool visualize) {
    
    CompressedData_32 compData = createCompressedData(compResult, decompData);
    
    cudaDeviceSynchronize();
    
    cudaEvent_t global_start_event, global_end_event;
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);
    
    // 主机侧内存分配
    size_t *streamChunkIds;
    cudaCheckError_32(cudaHostAlloc((void**)&streamChunkIds, 
        sizeof(size_t) * numStreams, cudaHostAllocDefault));
    
    size_t *streamOutputOffsets;
    cudaCheckError_32(cudaHostAlloc((void**)&streamOutputOffsets, 
        sizeof(size_t) * numStreams, cudaHostAllocDefault));
    
    // 创建流池和事件
    cudaStream_t *streams = new cudaStream_t[numStreams];
    cudaEvent_t *evSize = new cudaEvent_t[numStreams];
    cudaEvent_t *evData = new cudaEvent_t[numStreams];
    Stage_32 *stage = new Stage_32[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaStreamCreate(&streams[i]));
        cudaCheckError_32(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError_32(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
        stage[i] = IDLE_32;
    }
    
    // 计算最大chunk大小
    size_t maxChunkSize = 0;
    size_t maxCompressedSize = 0;
    for (size_t i = 0; i < compData.totalChunks; ++i) {
        maxChunkSize = std::max(maxChunkSize, compData.chunkElementCounts[i]);
        maxCompressedSize = std::max(maxCompressedSize, compData.chunkSizes[i]);
    }
    
    // 为每个流分配设备缓冲
    unsigned char **d_in = new unsigned char*[numStreams];
    float **d_out = new float*[numStreams];
    
    for (int i = 0; i < numStreams; ++i) {
        cudaCheckError_32(cudaMalloc(&d_in[i], maxCompressedSize));
        cudaCheckError_32(cudaMalloc(&d_out[i], maxChunkSize * sizeof(float)));
    }
    
    // 主循环
    size_t processedChunks = 0;
    size_t processedElements = 0;
    int active = 0;
    size_t totalDecompSize = 0;
    
    std::vector<size_t> compressedDataOffsets(compData.totalChunks);
    compressedDataOffsets[0] = 0;
    for (size_t i = 0; i < compData.totalChunks - 1; ++i) {
        compressedDataOffsets[i + 1] = compressedDataOffsets[i] + compData.chunkSizes[i];
    }
    
    cudaDeviceSynchronize();
    cudaEventRecord(global_start_event);
    
    while (processedChunks < compData.totalChunks || active > 0) {
        for (int s = 0; s < numStreams; ++s) {
            switch (stage[s]) {
                case IDLE_32:
                    if (processedChunks < compData.totalChunks) {
                        active += 1;
                        size_t chunkId = processedChunks;
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        size_t currentChunkCompressedSize = compData.chunkSizes[chunkId];
                        
                        streamChunkIds[s] = chunkId;
                        streamOutputOffsets[s] = processedElements;
                        
                        size_t compressedDataOffset = compressedDataOffsets[chunkId];
                        
                        // 异步H→D拷贝压缩数据
                        cudaCheckError_32(cudaMemcpyAsync(
                            d_in[s],
                            compData.cmpBytes + compressedDataOffset,
                            currentChunkCompressedSize,
                            cudaMemcpyHostToDevice,
                            streams[s]));
                        
                        // 调用解压接口
                        FalconDecompressor Falcon;
                        Falcon.Falcon_decompress_stream_optimized(
                            d_out[s],
                            d_in[s],
                            currentChunkElements,
                            currentChunkCompressedSize,
                            streams[s]);
                        
                        cudaCheckError_32(cudaEventRecord(evSize[s], streams[s]));
                        
                        processedChunks++;
                        processedElements += currentChunkElements;
                        
                        size_t outputOffset = streamOutputOffsets[s];
                        
                        // 异步D→H拷贝解压结果
                        cudaCheckError_32(cudaMemcpyAsync(
                            decompData.decData + outputOffset,
                            d_out[s],
                            currentChunkElements * sizeof(float),
                            cudaMemcpyDeviceToHost,
                            streams[s]));
                        
                        cudaCheckError_32(cudaEventRecord(evData[s], streams[s]));
                        stage[s] = DATA_PENDING_32;
                    }
                    break;
                    
                case DATA_PENDING_32:
                    if (cudaEventQuery(evData[s]) == cudaSuccess) {
                        size_t chunkId = streamChunkIds[s];
                        size_t currentChunkElements = compData.chunkElementCounts[chunkId];
                        totalDecompSize += currentChunkElements * sizeof(float);
                        
                        stage[s] = IDLE_32;
                        active -= 1;
                    }
                    break;
                    
                default:
                    break;
            }
        }
    }
    
    // 等待所有流完成
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaStreamSynchronize(streams[i]));
        cudaCheckError_32(cudaEventSynchronize(evData[i]));
    }
    
    cudaEventRecord(global_end_event);
    cudaEventSynchronize(global_end_event);
    
    // 计算总时间
    float totalTime;
    cudaEventElapsedTime(&totalTime, global_start_event, global_end_event);
    
    // 创建并返回分析结果
    PipelineAnalysis_32 result;
    result.compression_ratio = totalDecompSize / static_cast<double>(compData.totalCompressedSize);
    result.total_compressed_size = compData.totalCompressedSize / 1024.0 / 1024.0;
    result.total_size = totalDecompSize / 1024.0 / 1024.0;
    result.decomp_time = totalTime;
    result.decomp_throughout = (totalDecompSize / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0);
    
    // 验证解压数据(可选)
    if (visualize) {
        for(int z=0,tmp=0,k=0;z<3;z++) {   
            for(size_t i=0,j=tmp;i<3;i++) {
                while(decompData.oriData[j]==decompData.decData[j]&&j<processedElements) {
                    j++;
                    if(j>=processedElements) {
                        break;
                    }
                }
                if(j>=processedElements) {
                    printf("success!!\n");   
                    break;
                }
                while(tmp<=j) {
                    tmp+=compData.chunkElementCounts[k];
                    k++;
                }
                printf("chunk:%d,idx: %zu ,ori: %.8f , dec: %.8f \n",
                    k-1, j, decompData.oriData[j], decompData.decData[j]);
                j++;
            }
        }
    }
    
    // 清理设备内存
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaFree(d_out[i]));
        cudaCheckError_32(cudaFree(d_in[i]));
    }
    
    // 清理流和事件
    for (int i = 0; i < numStreams; i++) {
        cudaCheckError_32(cudaEventDestroy(evSize[i]));
        cudaCheckError_32(cudaEventDestroy(evData[i]));
        cudaCheckError_32(cudaStreamDestroy(streams[i]));
    }
    
    // 释放主机内存
    cudaCheckError_32(cudaFreeHost(streamChunkIds));
    cudaCheckError_32(cudaFreeHost(streamOutputOffsets));
    
    // 销毁全局事件
    cudaCheckError_32(cudaEventDestroy(global_start_event));
    cudaCheckError_32(cudaEventDestroy(global_end_event));
    
    // 释放动态分配的数组
    delete[] streams;
    delete[] evSize;
    delete[] evData;
    delete[] stage;
    delete[] d_in;
    delete[] d_out;
    
    return result;
}