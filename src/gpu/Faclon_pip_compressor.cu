#include<Faclon_pip_compressor.cuh>


CompressionResult execute_pipeline(ProcessedData &data, size_t chunkSize,int NUM_STREAM,bool visualize = false) {
    cudaDeviceSynchronize();
    // 创建时间线记录事件
    cudaEvent_t global_start_event, global_end_event; //全局开始和结束事件
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
    // 分配时添加保护
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
    sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));  // +2 for guards

    // 设置保护值
    locCmpSize[0] = 0xDEADBEEF;  // 前保护
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;  // 后保护


    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    // 初始化偏移量数组
    h_cmp_offset[0] = 0;

    bool of_rd[totalChunks+ NUM_STREAM]={0};//内存偏移准备完成信号
    of_rd[0]=true;

    // 新增：记录每个chunk信息的数组
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, sizeof(size_t) * totalChunks, cudaHostAllocDefault));

    // 用于存储每个chunk的压缩信息

    std::vector<size_t> chunkSizes(totalChunks);
    std::vector<size_t> chunkElementCountsVec(totalChunks);

    // ---------- 创建流池和事件 ----------
    const int MAX_EVENTS_PER_TYPE = totalChunks + NUM_STREAM; // 预留一些额外的事件
    
    cudaStream_t streams[NUM_STREAM];
    cudaEvent_t kernal_start[MAX_EVENTS_PER_TYPE]; // 核函数启动
    cudaEvent_t evSize[MAX_EVENTS_PER_TYPE]; // size 拷贝完成
    cudaEvent_t evData[MAX_EVENTS_PER_TYPE]; // 数据拷贝完成
    Stage stage[NUM_STREAM]; // 每流状态

    for (int i = 0; i < NUM_STREAM; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        stage[i] = IDLE;
    }

    for (int i = 0; i < MAX_EVENTS_PER_TYPE; ++i) {
        cudaCheckError(cudaEventCreateWithFlags(&kernal_start[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
    }


    // ---------- 为每个流分配固定设备缓冲 ----------
    double *d_in[NUM_STREAM];
    unsigned char *d_out[NUM_STREAM];
    for (int i = 0; i < NUM_STREAM; ++i) {
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

    std::vector<int> chunkIDX(NUM_STREAM);//和compINFO进行绑定
    while (processedEle < data.nbEle || active > 0) {
        int progress=0;
        for (int s = 0; s < NUM_STREAM; ++s) {
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
                        // 调用压缩接口（内部处理所有临时内存）
                        GDFCompressor_opt::GDFC_compress_stream(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],  // +1 因为有前保护
                            todo,
                            streams[s]);

                        // 记录尺寸事件
                        active += 1;
                        processedEle += todo;
                          
                        stage[s] = SIZE_PENDING;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                case SIZE_PENDING:
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
                        h_cmp_offset[idx+1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx+1]=true;//给下一个准备好了偏移
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;

                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        // 累计压缩大小
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s]+1] + 7) / 8;//+1因为有前缀保护
                        totalCmpSize += compressedBytes;
                        progress=1;
                        stage[s] = IDLE;
                        active -= 1;
                    }
                    break;
            }
        }
        // 避免忙等待 - 如果没有进展，短暂休眠
        if (!progress) {
            for (int i = 0; i < NUM_STREAM; i++) {
                cudaCheckError(cudaStreamSynchronize(streams[i]));
            }
        }
    }

    // 等待所有流完成
    for (int i = 0; i < (NUM_STREAM>totalChunks?totalChunks:NUM_STREAM); i++) {
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
    *data.cmpSize=totalCmpSize;

    // ---------- 清理资源 ----------
    // 清理设备内存
    for (int i = 0; i < NUM_STREAM; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }

    // 清理流和事件
    for (int i = 0; i < NUM_STREAM; i++) {
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


CompressionResult execute_pipeline(ProcessedData &data, size_t chunkSize) {
    cudaDeviceSynchronize();
    cudaEvent_t global_start_event, global_end_event; //全局开始和结束事件
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
    // 分配时添加保护
    unsigned int *locCmpSize;
    cudaCheckError(cudaHostAlloc((void**)&locCmpSize, 
    sizeof(unsigned int) * (totalChunks + 2), cudaHostAllocDefault));  // +2 for guards

    // 设置保护值
    locCmpSize[0] = 0xDEADBEEF;  // 前保护
    locCmpSize[totalChunks + 1] = 0xCAFEBABE;  // 后保护


    
    unsigned int *h_cmp_offset;
    cudaCheckError(cudaHostAlloc((void**)&h_cmp_offset, sizeof(unsigned int) * (totalChunks + 1), cudaHostAllocDefault));
    // 初始化偏移量数组
    h_cmp_offset[0] = 0;

    bool of_rd[totalChunks+ NUM_STREAMS]={0};//内存偏移准备完成信号
    of_rd[0]=true;

    // 新增：记录每个chunk信息的数组
    size_t *chunkElementCounts;
    cudaCheckError(cudaHostAlloc((void**)&chunkElementCounts, sizeof(size_t) * totalChunks, cudaHostAllocDefault));

    // 用于存储每个chunk的压缩信息

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
                        // 调用压缩接口（内部处理所有临时内存）
                        GDFCompressor_opt::GDFC_compress_stream(
                            d_in[s],
                            d_out[s],
                            &locCmpSize[chunkIDX[s] + 1],  // +1 因为有前保护
                            todo,
                            streams[s]);

                        // 记录尺寸事件
                        active += 1;
                        processedEle += todo;
                          
                        stage[s] = SIZE_PENDING;
                        cudaCheckError(cudaEventRecord(evSize[chunkIDX[s]], streams[s]));
                    }
                    break;
                case SIZE_PENDING:
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
                        h_cmp_offset[idx+1] = h_cmp_offset[idx] + compressedBytes;
                        of_rd[idx+1]=true;//给下一个准备好了偏移
                        chunkSizes[idx] = compressedBytes;
                        chunkElementCountsVec[idx] = chunkElementCounts[idx];
                        cudaCheckError(cudaEventRecord(evData[chunkIDX[s]], streams[s]));
                        stage[s] = DATA_PENDING;
                    }
                    break;

                case DATA_PENDING:
                    if (cudaEventQuery(evData[chunkIDX[s]]) == cudaSuccess) {
                        // 累计压缩大小
                        unsigned int compressedBytes = (locCmpSize[chunkIDX[s]+1] + 7) / 8;//+1因为有前缀保护
                        totalCmpSize += compressedBytes;
                        progress=1;
                        stage[s] = IDLE;
                        active -= 1;
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
    *data.cmpSize=totalCmpSize;
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
    cudaEvent_t global_start_event, global_end_event; //全局开始和结束事件
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

// 执行解压流程函数()
PipelineAnalysis execute_decompression_pipeline(const CompressionResult& compResult, ProcessedData &decompData,int NUM_STREAM) {
    CompressedData compData = createCompressedData(compResult, decompData);
    cudaDeviceSynchronize();
    // 创建时间线记录事件
    cudaEventCreate(&global_start_event);
    cudaEventCreate(&global_end_event);

    // 主机侧内存分配 - 用于记录每个流的处理信息
    size_t *streamChunkIds;
    cudaCheckError(cudaHostAlloc((void**)&streamChunkIds, sizeof(size_t) * NUM_STREAM, cudaHostAllocDefault));
    size_t *streamOutputOffsets;
    cudaCheckError(cudaHostAlloc((void**)&streamOutputOffsets, sizeof(size_t) * NUM_STREAM, cudaHostAllocDefault));

    // ---------- 创建流池和事件 ----------
    cudaStream_t streams[NUM_STREAM];
    cudaEvent_t evSize[NUM_STREAM]; // 压缩数据拷贝完成
    cudaEvent_t evData[NUM_STREAM]; // 解压数据拷贝完成
    Stage stage[NUM_STREAM]; // 每流状态

    for (int i = 0; i < NUM_STREAM; ++i) {
        cudaCheckError(cudaStreamCreate(&streams[i]));
        cudaCheckError(cudaEventCreateWithFlags(&evSize[i], cudaEventDisableTiming));
        cudaCheckError(cudaEventCreateWithFlags(&evData[i], cudaEventDisableTiming));
        stage[i] = IDLE;
    }

    // ---------- 为每个流分配固定设备缓冲 ----------
    unsigned char *d_in[NUM_STREAM];  // 压缩数据输入缓冲
    double *d_out[NUM_STREAM];        // 解压数据输出缓冲

    // 计算最大chunk大小用于内存分配
    size_t maxChunkSize = 0;
    size_t maxCompressedSize = 0;
    for (size_t i = 0; i < compData.totalChunks; ++i) {
        maxChunkSize = std::max(maxChunkSize, compData.chunkElementCounts[i]);
        maxCompressedSize = std::max(maxCompressedSize, compData.chunkSizes[i]);
    }

    for (int i = 0; i < NUM_STREAM; ++i) {
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
        for (int s = 0; s < NUM_STREAM; ++s) {
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
    for (int i = 0; i < NUM_STREAM; i++) {
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
    

    // ---------- 清理资源 ----------
    // 清理设备内存
    for (int i = 0; i < NUM_STREAM; i++) {
        cudaCheckError(cudaFree(d_out[i]));
        cudaCheckError(cudaFree(d_in[i]));
    }

    // 清理流和事件
    for (int i = 0; i < NUM_STREAM; i++) {
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

CompressionInfo test_streams_compression(ProcessedData data, size_t chunkSize)
{


    CompressionResult compResult = execute_pipeline(data, chunkSize, NUM_STREAM);
    cudaDeviceSynchronize(); 

    PipelineAnalysis decompAnalysis = execute_decompression_pipeline(compResult, data, NUM_STREAM);
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
