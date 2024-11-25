//
// Created by lz on 24-9-26.
//
#include "GDFCompressor.cuh"
#include <iomanip> // 用于设置输出格式


#define MAX_BITCOUNT 64
#define MAX_BITSIZE_PER_BLOCK (64 + 64 + 8 + 8 + (BLOCK_SIZE_G - 1) * MAX_BITCOUNT)
#define MAX_BYTES_PER_BLOCK ((MAX_BITSIZE_PER_BLOCK + 7) / 8)

// 核函数
__device__ static unsigned long zigzag_encode_cuda(long value) {
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}

__device__ static int getDecimalPlaces(double value) {
    double trac = value + POW_NUM_G - POW_NUM_G;
    double temp = value;
    int digits = 0;
    int64_t int_temp, trac_temp;
    memcpy(&int_temp, &temp, sizeof(double));
    memcpy(&trac_temp, &trac, sizeof(double));
    while (llabs(trac_temp - int_temp) > 1.0 && digits < 16) { // 使用 llabs 替代 std::abs
        digits++;
        double td = pow(10, digits);
        temp = value * td;
        memcpy(&int_temp, &temp, sizeof(double));
        trac = temp + POW_NUM_G - POW_NUM_G;
        memcpy(&trac_temp, &trac, sizeof(double));
    }
    return digits;
}



// 辅助函数：打印缓冲区的指定范围，以十六进制格式显示
__device__ void print_bytes(const unsigned char* buffer, size_t start, size_t length, const char* label) {
    printf("%s: ", label);
    for (size_t i = start; i < start + length; ++i) {
        // 打印每个字节的十六进制表示
        printf("%02x ", buffer[i]);
    }
    printf("\n");
}

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ inline int device_min(int a, int b) {
    return (a < b) ? a : b;
}

__device__ inline int device_max(int a, int b) {
    return (a > b) ? a : b;
}

__device__ inline uint64_t device_min_uint64(uint64_t a, uint64_t b) {
    return (a < b) ? a : b;
}

__device__ inline uint64_t device_max_uint64(uint64_t a, uint64_t b) {
    return (a > b) ? a : b;
}

// 核函数实现
/*
input：输入的完整数据
blockSize:块大小
totalSize:输入数据的完整大小
output:输出数据
bitSizes:每个块的bitSize
*/
__global__ void compressBlockKernel(const double* input, int blockSize, int totalSize, unsigned char* output, uint64_t* bitSizes) {
    int blockIdxGlobal = blockIdx.x; // 块序列号
    int startIdx = blockIdxGlobal * blockSize; // 当前块的数据开始位置
    int endIdx = min(startIdx + blockSize, totalSize); // 当前块的数据结束位置

    // 块内数据同步
    extern __shared__ unsigned char sharedMem[]; // 动态分配共享内存
    int* decimalPlaces = (int*)sharedMem; // 小数位数数组
    long* integers = (long*)&decimalPlaces[blockSize];
    uint64_t* deltas = (uint64_t*)&integers[blockSize];
    __shared__ int maxDecimalPlaces; // 最大小数位
    __shared__ long firstValue; // 第一位
    __shared__ int bitCount; // 差分值的最大位宽

    int tid = threadIdx.x; // 线程id

    // 1. 每个线程计算其小数位数
    if (startIdx + tid < endIdx) {
        decimalPlaces[tid] = getDecimalPlaces(input[startIdx + tid]);
    } else {
        decimalPlaces[tid] = 0;
    }
    __syncthreads();

    // 归约找到最大的小数位数
    if (tid == 0) {
        maxDecimalPlaces = 0;
        for (int i = 0; i < blockSize && startIdx + i < endIdx; i++) {
            maxDecimalPlaces = device_max(maxDecimalPlaces, decimalPlaces[i]);
        }
        // 限制 maxDecimalPlaces 不超过 16
        maxDecimalPlaces = device_min(maxDecimalPlaces, 16);
    }
    __syncthreads();

    // 2. 转换浮点数为整数
    if (startIdx + tid < endIdx) {
        if (maxDecimalPlaces > 15) {
            unsigned long long bits = __double_as_longlong(input[startIdx + tid]);
            integers[tid] = static_cast<long>(bits);
        }
        else {
            integers[tid] = static_cast<long>(round(input[startIdx + tid] * pow(10, maxDecimalPlaces)));
        }
    }
    __syncthreads();

    // 3. 计算Delta序列
    if (tid == 0) {
        firstValue = integers[0];
    }
    __syncthreads();

    if (startIdx + tid < endIdx) {
        if (tid != 0) {
            deltas[tid - 1] = integers[tid] - integers[tid - 1];
        }
    }
    __syncthreads();

    // 4. ZigZag编码
    if (tid < (endIdx - startIdx - 1)) { // 确保不越界
        deltas[tid] = zigzag_encode_cuda(deltas[tid]); // 编码已经计算好的Delta
        // 调试输出每个 Delta
        // printf("Block %d, Delta %d: %llu\n", blockIdxGlobal, tid, deltas[tid]); // 可选
    }
    __syncthreads();

    // 5. 输出位数计算，用第一个线程进行处理
    if (tid == 0) {
        bitCount = 0;
        uint64_t maxDelta = 0;
        for (int i = 0; i < endIdx - startIdx - 1; i++) {
            maxDelta = device_max_uint64(maxDelta, deltas[i]);
        }
        while (maxDelta > 0) {
            maxDelta >>= 1;
            bitCount++;
        }
        
        // 防止 bitCount 为 0（所有 deltas 都为 0）
        if (bitCount == 0) {
            bitCount = 1;
        }

        // 限制 bitCount 不超过 MAX_BITCOUNT
        bitCount = device_min(bitCount, MAX_BITCOUNT);
    }
    __syncthreads();

    // 6. 写入 output
    if (tid == 0) {
        // 计算 bitSize，确保使用 uint64_t 防止溢出
        uint64_t bitSize = 64ULL + 64ULL + 8ULL + 8ULL + ((uint64_t)(endIdx - startIdx - 1)) * ((uint64_t)bitCount);
        bitSizes[blockIdxGlobal] = bitSize;

        // 检查 bitSize 是否合理
        if (bitSize > MAX_BITSIZE_PER_BLOCK) {
            // 设置 bitSize 为 0，表示该块无效
            bitSizes[blockIdxGlobal] = 0;
            // 可以在此处添加更多的错误处理逻辑
            return;
        }

        int outputIdx = blockIdxGlobal * MAX_BYTES_PER_BLOCK; // 使用 MAX_BYTES_PER_BLOCK 作为偏移量

        // 写入 bitSize (8 字节)
        for (int i = 0; i < 8; i++) {
            output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;
        }

        // 写入 firstValue (8 字节)
        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        for (int i = 0; i < 8; i++) {
            output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;
        }

        // 写入 maxDecimalPlaces 和 bitCount (各1字节)
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(bitCount);

        // 写入 Delta 差分值序列
        int deltaStartIdx = outputIdx + 18; // Delta 数据起始位置
        int bitOffset = 0; // 当前字节已使用的位数
        unsigned char currentByte = 0; // 当前字节缓冲区

        for(int i = 0; i < endIdx - startIdx -1; i++) {
            uint64_t deltaValue = deltas[i] & ((1ULL << bitCount) -1);

            // 将 Delta 填充到当前字节的剩余空间
            currentByte |= (deltaValue << bitOffset);
            bitOffset += bitCount;

            // 如果当前字节已满（8 位），写入 output，并开始下一个字节
            while(bitOffset >=8) {
                output[deltaStartIdx++] = currentByte;
                bitOffset -=8;
                // 将剩余的 Delta 位数放入新字节
                currentByte = (deltaValue >> (bitCount - bitOffset)) &0xFF; // 运算优先级修正
            }
        }

        // 如果有未写入的字节（即 bitOffset > 0），写入最后一个字节
        if(bitOffset >0) {
            output[deltaStartIdx++] = currentByte;
        }

        // 填充剩余的输出缓冲区
        while(deltaStartIdx < MAX_BYTES_PER_BLOCK){
            output[deltaStartIdx++] = 0x00;
        }

        // 调试输出
        //printf("Block %d: bitSize=%llu, bitCount=%d, maxDecimalPlaces=%d\n", blockIdxGlobal, bitSize, bitCount, maxDecimalPlaces);
    }
}

// 初始化设备内存
void GDFCompressor::setupDeviceMemory(const std::vector<double>& input, double*& d_input, unsigned char*& d_output, unsigned long*& d_bitSizes) {
    size_t inputSize = input.size();
    cudaCheckError(cudaMalloc((void**)&d_input, inputSize * sizeof(double))); // 初始化
    int numBlocks = (inputSize + BLOCK_SIZE_G - 1) / BLOCK_SIZE_G;
    cudaCheckError(cudaMalloc((void**)&d_output, numBlocks * MAX_BYTES_PER_BLOCK * sizeof(unsigned char)));
    cudaCheckError(cudaMalloc((void**)&d_bitSizes, numBlocks * sizeof(unsigned long)));
    
    cudaCheckError(cudaMemcpy(d_input, input.data(), inputSize * sizeof(double), cudaMemcpyHostToDevice));
}

// 释放设备内存
void GDFCompressor::freeDeviceMemory(double* d_input, unsigned char* d_output, unsigned long* d_bitSizes) {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bitSizes);
}

// 主压缩函数
void GDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output) {
    size_t inputSize = input.size();
    if (inputSize == 0) return;

    int blockSize = BLOCK_SIZE_G; // 确保 BLOCK_SIZE_G 已定义
    size_t numBlocks = (inputSize + blockSize - 1) / blockSize; // 块数量

    double* d_input;
    unsigned char* d_output;
    uint64_t* d_bitSizes; // 修改为 uint64_t*

    // 分配设备内存
    setupDeviceMemory(input, d_input, d_output, d_bitSizes);

    // 启动核函数
    size_t sharedMemSize = blockSize * sizeof(int) + blockSize * sizeof(long) + blockSize * sizeof(uint64_t); // decimalPlaces + integers + deltas
    compressBlockKernel<<<numBlocks, blockSize, sharedMemSize>>>(d_input, blockSize, inputSize, d_output, d_bitSizes);

    // 检查错误
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // 复制 bitSizes 回主机
    std::vector<uint64_t> bitSizes(numBlocks);
    cudaCheckError(cudaMemcpy(bitSizes.data(), d_bitSizes, numBlocks * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    // 计算每个块的输出偏移量
    std::vector<uint64_t> offsets(numBlocks, 0);
    uint64_t totalCompressedBits = 0;
    for (size_t i = 0; i < numBlocks; i++) {
        offsets[i] = totalCompressedBits;
        totalCompressedBits += bitSizes[i];
    }
    uint64_t totalCompressedBytes = (totalCompressedBits + 7) / 8; // 按字节对齐

    // 分配输出缓冲区
    output.resize(totalCompressedBytes, 0);

    // 复制 d_output 到主机的临时缓冲区，分配为 numBlocks * MAX_BYTES_PER_BLOCK
    std::vector<unsigned char> tempOutput(numBlocks * MAX_BYTES_PER_BLOCK);
    cudaCheckError(cudaMemcpy(tempOutput.data(), d_output, numBlocks * MAX_BYTES_PER_BLOCK * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // 将压缩数据整理到最终的 output 中
    size_t currentBitPos = 0;
    for (size_t i = 0; i < numBlocks; i++) {
        uint64_t bitSize = bitSizes[i];
        if (bitSize == 0) {
            continue; // 跳过填充块
        }
        int byteSize = (bitSize + 7) / 8;
        size_t srcIdx = i * MAX_BYTES_PER_BLOCK;
        size_t dstByteIdx = currentBitPos / 8;
        int dstBitOffsetRemainder = currentBitPos % 8;

        for (int b = 0; b < byteSize; b++) {
            if (dstBitOffsetRemainder == 0) {
                // 无偏移，直接复制
                if (dstByteIdx + b < output.size()) {
                    output[dstByteIdx + b] = tempOutput[srcIdx + b];
                }
            } else {
                // 有偏移，将数据拆分并合并
                if (dstByteIdx + b < output.size()) {
                    output[dstByteIdx + b] |= tempOutput[srcIdx + b] << dstBitOffsetRemainder;
                }
                if (dstByteIdx + b + 1 < output.size()) {
                    output[dstByteIdx + b + 1] |= tempOutput[srcIdx + b] >> (8 - dstBitOffsetRemainder);
                }
            }
        }
        currentBitPos += bitSize;
    }

    // 释放设备内存
    freeDeviceMemory(d_input, d_output, d_bitSizes);
}
