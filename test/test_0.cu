// 修改后的核函数实现 - 增加边界检查和错误处理
__global__ void decompressKernel(
    const unsigned char* compressedData, // 压缩数据
    double* output,                      // 解压后数据输出
    const int* offsets,                  // 每个块的偏移
    int numBlocks,                       // 块数
    int numDatas,                        // 所有块的总数据个数
    int maxBlockSize,                    // 每个数据块的最大大小
    int* errorFlag                       // 错误标志位，用于在主机端检测内核错误
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= numBlocks) return;

    const unsigned char* blockData = compressedData + offsets[blockId];
    size_t bitPos = 0;

    // 计算当前块的数据大小
    int numData = min(numDatas - blockId * maxBlockSize, maxBlockSize);
    if (numData <= 0) {
        // 没有数据要处理
        return;
    }

    // 读取 bitSize 和 firstValue
    uint64_t bitSize = readBitsDevice(blockData, bitPos, 64);
    if (bitSize == 0 || bitSize > 1024 * 64) { // 设置合理的上限
        atomicExch(errorFlag, blockId + 1); // 记录错误的块ID（+1避免与0冲突）
        return;
    }

    int64_t firstValue = static_cast<int64_t>(readBitsDevice(blockData, bitPos, 64));
    
    // 读取 maxDecimalPlaces
    uint64_t maxDecimalPlacesRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char maxDecimalPlaces = static_cast<unsigned char>(maxDecimalPlacesRaw);
    
    // 读取 maxBeta (8 位)
    uint64_t maxBetaRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char maxBeta = static_cast<unsigned char>(maxBetaRaw);

    // 读取 bitCount
    uint64_t bitCountRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = static_cast<unsigned char>(bitCountRaw);

    if (bitCount == 0 || bitCount > 64) { // 增加合理的上限检查
        atomicExch(errorFlag, blockId + 1);
        return;
    }

    uint64_t flag1 = readBitsDevice(blockData, bitPos, 64);

    // 确保数据块大小在合理范围内
    int dataByte = (numData + 7) / 8;
    if (dataByte <= 0 || dataByte > 128) {
        atomicExch(errorFlag, blockId + 1);
        return;
    }

    int flag2Size = (dataByte + 7) / 8;
    if (flag2Size <= 0 || flag2Size > 16) {
        atomicExch(errorFlag, blockId + 1);
        return;
    }

    // 使用动态共享内存替代固定大小数组
    extern __shared__ uint8_t sharedMem[];
    uint8_t* result = sharedMem;
    uint8_t* flag2 = sharedMem + 64 * 128;

    // 初始化 flag2
    for (int i = 0; i < bitCount; i++) {
        for (int j = 0; j < flag2Size * 8; j++) {
            if (i < 64 && j < 128) {
                flag2[i * 128 + j] = 0;
            }
        }
    }

    // 处理每一位
    for (int i = 0; i < bitCount; i++) {
        if ((flag1 & (1ULL << i)) != 0) { // i列稀疏
            for (int z = 0; z < flag2Size * 8; z++) {
                if (i < 64 && z < 128) {
                    flag2[i * 128 + z] = readBitsDevice(blockData, bitPos, 1);
                }
            }
            
            for (int j = 0; j < dataByte; j++) {
                if (i < 64 && j < 128) {
                    if (flag2[i * 128 + j] != 0) {
                        result[i * 128 + j] = static_cast<unsigned char>(readBitsDevice(blockData, bitPos, 8));
                    } else {
                        result[i * 128 + j] = 0;
                    }
                }
            }
        } else {
            for (int j = 0; j < dataByte; j++) {
                if (i < 64 && j < 128) {
                    result[i * 128 + j] = static_cast<unsigned char>(readBitsDevice(blockData, bitPos, 8));
                }
            }
        }
    }

    // 读取 deltas
    uint64_t* deltasZigzag = new uint64_t[numData];
    memset(deltasZigzag, 0, numData * sizeof(uint64_t));
    
    for (int i = 0; i < bitCount; i++) {
        for (int j = 0; j < numData; j++) {
            int byteIndex = j / 8;
            int bitIndex = j % 8;
            if (i < 64 && byteIndex < 128) {
                uint8_t bitValue = (result[i * 128 + byteIndex] >> (7 - bitIndex)) & 1;
                deltasZigzag[j] |= (uint64_t(bitValue) << (bitCount - 1 - i));
            }
        }
    }

    // 解码 deltas
    int64_t* deltas = new int64_t[numData];
    memset(deltas, 0, numData * sizeof(int64_t));
    
    for (int i = 0; i < numData; i++) {
        deltas[i] = zigzag_decode(deltasZigzag[i]);
    }

    // 计算整数值
    int64_t* integers = new int64_t[numData];
    memset(integers, 0, numData * sizeof(int64_t));
    
    integers[0] = firstValue;
    for (int i = 1; i < numData; i++) {
        integers[i] = integers[i - 1] + deltas[i];
    }

    // 转换为 double 值并存储结果
    for (int i = 0; i < numData; i++) {
        double d;
        if (maxBeta > 15) {
            // 直接将整数位转换为 double
            uint64_t bits = static_cast<uint64_t>(integers[i]);
            memcpy(&d, &bits, sizeof(double));
        } else {
            // 使用更稳定的计算方法
            double scale = 1.0;
            for (int p = 0; p < maxDecimalPlaces; p++) {
                scale *= 10.0;
            }
            d = static_cast<double>(integers[i]) / scale;
        }
        
        // 安全地写入输出数组
        size_t outputIdx = blockId * maxBlockSize + i;
        if (outputIdx < numDatas) {
            output[outputIdx] = d;
        }
    }
    
    // 释放临时分配的内存
    delete[] deltasZigzag;
    delete[] deltas;
    delete[] integers;
}

// 改进的 GDFC_decompress 函数
void GDFDecompressor::GDFC_decompress(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream) {
    // 在设备上计算偏移
    std::vector<unsigned char> hostCmpBytes(cmpSize);
    cudaError_t err = cudaMemcpyAsync(hostCmpBytes.data(), d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in GDFC_decompress (cudaMemcpyAsync): " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // 同步流以确保数据已复制
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in GDFC_decompress (cudaStreamSynchronize): " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // 计算块偏移
    std::vector<int> offsets;
    BitReader reader(hostCmpBytes);
    
    // 添加安全检查
    size_t totalBits = cmpSize * 8;
    int maxBlocks = (nbEle + 1023) / 1024;  // 估计的最大块数
    
    while (reader.getBitPos() + 64 + 64 + 8 + 8 + 64 <= totalBits && offsets.size() < maxBlocks) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(64);
        
        // 添加安全检查
        if (bitSize < 64 || bitSize > totalBits - reader.getBitPos()) {
            std::cerr << "Error: Invalid bitSize " << bitSize << " at position " << reader.getBitPos() / 8 
                      << " (total bits: " << totalBits << ", remaining: " << totalBits - reader.getBitPos() << ")" << std::endl;
            break;
        }
        
        reader.advance(bitSize - 64);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        std::cerr << "Error: No valid blocks found in compressed data of size " << cmpSize << " bytes" << std::endl;
        return;
    }
    
    std::cout << "解压块数: " << numBlocks << ", 总元素数: " << nbEle << std::endl;

    // 分配错误标志
    int *d_errorFlag;
    cudaMalloc(&d_errorFlag, sizeof(int));
    cudaMemsetAsync(d_errorFlag, 0, sizeof(int), stream);

    // 将偏移复制到设备
    int* d_offsets;
    err = cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in GDFC_decompress (cudaMalloc): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_errorFlag);
        return;
    }
    
    err = cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in GDFC_decompress (cudaMemcpyAsync): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_offsets);
        cudaFree(d_errorFlag);
        return;
    }

    // 调用核函数解压
    int threadsPerBlock = 128;  // 减小线程数，避免资源不足
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    // 计算所需的共享内存大小
    size_t sharedMemSize = 64 * 128 * 2; // 为 result 和 flag2 分配共享内存
    
    std::cout << "启动解压内核: " << blocksPerGrid << " 块, 每块 " << threadsPerBlock << " 线程" << std::endl;
    
    decompressKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize, stream>>>(
        d_cmpBytes, d_decData, d_offsets, numBlocks, nbEle, 1024, d_errorFlag);
    
    // 检查内核启动错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in GDFC_decompress (kernel launch): " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_offsets);
        cudaFree(d_errorFlag);
        return;
    }
    
    // 检查内核执行错误
    int hostErrorFlag = 0;
    cudaMemcpyAsync(&hostErrorFlag, d_errorFlag, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // 等待错误检查完成
    
    if (hostErrorFlag > 0) {
        std::cerr << "CUDA Kernel Error: Block " << (hostErrorFlag - 1) << " encountered an error during decompression" << std::endl;
    }

    // 释放资源
    cudaFree(d_offsets);
    cudaFree(d_errorFlag);
}