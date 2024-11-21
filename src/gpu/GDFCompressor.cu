//
// Created by lz on 24-9-26.
//
#include "GDFCompressor.cuh"


// 核函数实现
/*
input：输入的完整数据
blockSize:块大小
totalSize:输入数据的完整大小
output:输出数据
*/
__global__ void compressBlockKernel(const double* input, int blockSize, int totalSize, unsigned char* output, int* bitSizes) {
    int blockIdxGlobal = blockIdx.x;//块序列号
    int startIdx = blockIdxGlobal * blockSize;//当前块的数据开始位置
    int endIdx = min(startIdx + blockSize, totalSize);//当前块的数据结束位置
    
    //块内数据同步
    __shared__ int decimalPlaces[BLOCK_SIZE_G];//小数位数数组
    __shared__ long integers[BLOCK_SIZE_G];//整数数组
    __shared__ long deltas[BLOCK_SIZE_G];//差分数组
    __shared__ int maxDecimalPlaces;//最大小数位
    __shared__ long firstValue;//第一位
    __shared__ int bitCount; // 差分值的最大位宽


    int tid = threadIdx.x;//线程id

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
            maxDecimalPlaces = max(maxDecimalPlaces, decimalPlaces[i]);
        }
    }
    __syncthreads();

    // 2. 转换浮点数为整数
    if (startIdx + tid < endIdx) {

        if(maxDecimalPlaces>15)
        {
            uint64_t bits;
            std::memcpy(&bits, &input[startIdx + tid], sizeof(bits));
            integers[tid] = static_cast<long>(bits);
        }else
        {
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
        if (tid == 0) {
            //deltas[tid] = firstValue;
        } else {
            deltas[tid-1] = integers[tid] - integers[tid - 1];
        }
    }
    __syncthreads();

    // 4. Zigzag编码
    if (startIdx + tid + 1 < endIdx) {
        deltas[tid] = zigzag_encode_cuda(deltas[tid]);
    }
    __syncthreads();

    // 5. 输出位数计算,用第一个线程进行处理
    if (tid == 0) {
        bitCount = 0;
        long maxDelta = 0;
        for (int i = 0; i < endIdx - startIdx; i++) {
            maxDelta = max(maxDelta, deltas[i]);
        }
        while (maxDelta > 0) {
            maxDelta >>= 1;
            bitCount++;
        }
        
    }
    __syncthreads();
    // 6. 写入 output
    if (tid == 0) {
        // 写入 bitSize
        int bitSize = 64 + 64 + 8 + 8 + (endIdx - startIdx - 1) * bitCount;
        bitSizes[blockIdxGlobal] = bitSize;
        int outputIdx = blockIdxGlobal * blockSize; // output 的偏移量
        
        // 写入 bitSize (8 字节)
        for (int i = 0; i < 8; i++) {
            output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;
        }

        // 写入 firstValue (8 字节)
        for (int i = 0; i < 8; i++) {
            output[outputIdx + 8 + i] = (firstValue >> (i * 8)) & 0xFF;
        }

        // 写入 maxDecimalPlaces 和 bitCount
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(bitCount);

        // 写入 Delta 差分值序列
        int deltaStartIdx = outputIdx + 18; // Delta 数据起始位置
        int bitOffset = 0; // 当前字节已使用的位数
        unsigned char currentByte = 0; // 当前字节缓冲区

        for (int i = 0; i < endIdx - startIdx - 1; i++) {
            // 获取 Delta 的低 bitCount 位
            unsigned long deltaValue = deltas[i] & ((1 << bitCount) - 1);

            // 将 Delta 填充到当前字节的剩余空间
            currentByte |= (deltaValue << bitOffset);
            bitOffset += bitCount;

            // 如果当前字节已满（8 位），写入 output，并开始下一个字节
            if (bitOffset >= 8) {
                output[deltaStartIdx++] = currentByte;
                bitOffset -= 8;
                currentByte = (deltaValue >> (bitCount - bitOffset)) & 0xFF; // 将剩余的 Delta 位数放入新字节
            }
        }

        // 如果有未写入的字节（即 bitOffset > 0），写入最后一个字节
        if (bitOffset > 0) {
            output[deltaStartIdx++] = currentByte;
        }

    //     memcpy(&output[outputIdx], &bitSize, 8);

    //     // 写入 firstValue
    //     memcpy(&output[outputIdx + 8], &firstValue, 8);

    //     // 写入 maxDecimalPlaces 和 bitCount
    //     output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
    //     output[outputIdx + 17] = static_cast<unsigned char>(bitCount);

    //     // 写入 Delta 序列
    //     int deltaStartIdx = outputIdx + 18; // Delta 数据起始位置
    //     for (int i = 0; i < endIdx - startIdx - 1; i++) {
    //         memcpy(&output[deltaStartIdx + i * bitCount / 8], &deltas[i], bitCount / 8);
    //     }
    }
}

// 初始化设备内存
void GDFCompressor::setupDeviceMemory(const std::vector<double>& input, double*& d_input, unsigned char*& d_output, int*& d_bitSizes) {
    size_t inputSize = input.size();
    cudaMalloc((void**)&d_input, inputSize * sizeof(double));//初始化
    cudaMalloc((void**)&d_output, inputSize * sizeof(unsigned char));
    cudaMalloc((void**)&d_bitSizes, (inputSize / blockSize + 1) * sizeof(int));

    cudaMemcpy(d_input, input.data(), inputSize * sizeof(double), cudaMemcpyHostToDevice);
}

// 释放设备内存
void GDFCompressor::freeDeviceMemory(double* d_input, unsigned char* d_output, int* d_bitSizes) {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bitSizes);
}

// 主压缩函数
void GDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output) {
    size_t inputSize = input.size();
    if (inputSize == 0) return;

    double* d_input;
    unsigned char* d_output;
    int* d_bitSizes;

    setupDeviceMemory(input, d_input, d_output, d_bitSizes);

    int numBlocks = (inputSize + blockSize - 1) / blockSize;//块数量
    compressBlockKernel<<<numBlocks, blockSize>>>(d_input, blockSize, inputSize, d_output, d_bitSizes);

    std::vector<int> bitSizes(numBlocks);
    cudaMemcpy(bitSizes.data(), d_bitSizes, numBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // TODO: 将结果整理到output中
    int totalCompressedBits = 0;
    for (int i = 0; i < numBlocks; i++) {
        totalCompressedBits += bitSizes[i];
    }
    int totalCompressedBytes = (totalCompressedBits + 7) / 8; // 按字节对齐

    output.resize(totalCompressedBytes);
    cudaMemcpy(output.data(), d_output, inputSize * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    freeDeviceMemory(d_input, d_output, d_bitSizes);
}