// 阶段2: 解压处理
std::cout << "\n===== 阶段2: 执行解压流水线 =====\n";

cmp_bytes_pool.reset();
MemoryPool dec_data_pool(poolSize, chunkSize * sizeof(double));

cudaEvent_t decomp_start, decomp_end;
cudaEventCreate(&decomp_start);
cudaEventCreate(&decomp_end);
cudaEventRecord(decomp_start);

// Create a structure to track decompression operations
struct DecompressionTask {
    int streamIdx;
    double* d_decData;
    unsigned char* d_cmpBytes;
    size_t original_offset;
    size_t original_size;
};

std::vector<DecompressionTask> active_tasks;

// Process each compressed block
for (size_t i = 0; i < compression_infos.size(); ++i) {
    const auto& info = compression_infos[i];
    int streamIdx = i % NUM_STREAMS;
    
    // Validate compressed data size
    if (info.compressed_size <= 0) {
        std::cerr << "Error: Invalid compressed size for chunk " << i 
                 << ": " << info.compressed_size << " bytes" << std::endl;
        continue;
    }
    
    std::cout << "解压数据块 " << i << " (流 " << streamIdx << "): 原始偏移 " 
              << info.original_offset << ", 压缩偏移 " << info.compressed_offset 
              << ", 大小: " << info.original_size << " 元素, 压缩大小: " 
              << info.compressed_size << " 字节" << std::endl;
    
    // Wait if all streams are busy
    if (active_tasks.size() >= NUM_STREAMS) {
        // Wait for the earliest task to complete
        int wait_stream = active_tasks[0].streamIdx;
        cudaCheckError(cudaStreamSynchronize(operators[wait_stream].get_stream()));
        
        // Record timing for the completed task
        StreamTiming timing = operators[wait_stream].calculate_decompression_timing();
        decomp_timings.push_back(timing);
        
        // Free the GPU memory for the completed task
        dec_data_pool.deallocate(active_tasks[0].d_decData);
        cmp_bytes_pool.deallocate(active_tasks[0].d_cmpBytes);
        
        // Remove the completed task
        active_tasks.erase(active_tasks.begin());
    }
    
    // Allocate GPU memory
    double* d_decData = (double*)dec_data_pool.allocate();
    unsigned char* d_cmpBytes = (unsigned char*)cmp_bytes_pool.allocate();
    
    if (!d_decData || !d_cmpBytes) {
        std::cerr << "解压阶段内存池分配失败!" << std::endl;
        break;
    }
    
    // Set the current block index
    operators[streamIdx].set_index(i);
    
    // Execute decompression (asynchronous operation)
    operators[streamIdx].decompress_chunk(
        decompressed_data,
        data.cmpBytes,
        d_decData,
        d_cmpBytes,
        info.original_size, info.compressed_size, info.compressed_offset, info.original_offset
    );
    
    // Track this task
    DecompressionTask task = {
        streamIdx,
        d_decData,
        d_cmpBytes,
        info.original_offset,
        info.original_size
    };
    active_tasks.push_back(task);
}

// Wait for all remaining decompression tasks to complete
while (!active_tasks.empty()) {
    int wait_stream = active_tasks[0].streamIdx;
    cudaCheckError(cudaStreamSynchronize(operators[wait_stream].get_stream()));
    
    // Record timing
    StreamTiming timing = operators[wait_stream].calculate_decompression_timing();
    decomp_timings.push_back(timing);
    
    // Free GPU memory
    dec_data_pool.deallocate(active_tasks[0].d_decData);
    cmp_bytes_pool.deallocate(active_tasks[0].d_cmpBytes);
    
    // Remove the completed task
    active_tasks.erase(active_tasks.begin());
}

cudaEventRecord(decomp_end);
cudaEventSynchronize(decomp_end);

float decompression_time = timer.Elapsed();
std::cout << "解压阶段完成，用时: " << decompression_time << " ms" << std::endl;</parameter>