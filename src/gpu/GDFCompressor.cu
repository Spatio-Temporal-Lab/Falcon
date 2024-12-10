//
// Created by lz on 24-9-26.
//
#include "GDFCompressor.cuh"
#include <iomanip> // 用于设置输出格式

// 定义常量
#define DATA_PER_THREAD 32
#define MAX_BITCOUNT 64
#define MAX_BITSIZE_PER_BLOCK (64 + 64 + 8 + 8 + (BLOCK_SIZE_G * DATA_PER_THREAD - 1) * MAX_BITCOUNT)
#define MAX_BYTES_PER_BLOCK ((MAX_BITSIZE_PER_BLOCK + 7) / 8)

// pow10_table 和 POW_NUM_G
__constant__ double pow10_table[17] = {
    1.0,                    // 10^0
    10.0,                   // 10^1
    100.0,                  // 10^2
    1000.0,                 // 10^3
    10000.0,                // 10^4
    100000.0,               // 10^5
    1000000.0,              // 10^6
    10000000.0,             // 10^7
    100000000.0,            // 10^8
    1000000000.0,           // 10^9
    10000000000.0,          // 10^10
    100000000000.0,         // 10^11
    1000000000000.0,        // 10^12
    10000000000000.0,       // 10^13
    100000000000000.0,      // 10^14
    1000000000000000.0,     // 10^15
    10000000000000000.0     // 10^16
};

// ZigZag 编码函数
// __device__ static uint64_t zigzag_encode_cuda(int64_t value) {
//     return (value << 1) ^ (value >> 63);
// }
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
        double td = pow10_table[digits]; // 使用查找表替代 pow
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
// 定义共享内存结构体，确保对齐
struct SharedMemory {
    int decimalPlaces[BLOCK_SIZE_G * DATA_PER_THREAD];
    // Padding to align to 8 bytes
    char padding[sizeof(int64_t) - (BLOCK_SIZE_G * DATA_PER_THREAD * sizeof(int)) % sizeof(int64_t)];
    int64_t integers[BLOCK_SIZE_G * DATA_PER_THREAD];
    uint64_t deltas[BLOCK_SIZE_G * DATA_PER_THREAD - 1];
};

// 核函数实现
/*
input：输入的完整数据
blockSize:块大小//线程数量
totalSize:输入数据的完整大小
output:输出数据
bitSizes:每个块的bitSize
*/
__global__ void compressBlockKernel(const double* input, int blockSize, int totalSize, unsigned char* output, uint64_t* bitSizes) {
    int blockIdxGlobal = blockIdx.x;// 块id
    int tid = threadIdx.x;// 线程id

    // 每个线程处理32个数据项,每个线程在所有数据中的开始和结束位置
    int startIdx = blockIdxGlobal * blockSize * DATA_PER_THREAD + tid * DATA_PER_THREAD;
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    
    int startIdxInBlock = tid * DATA_PER_THREAD;
 
    // 结构体尝试
    extern __shared__ unsigned char sharedMemRaw[];// 动态分配共享内存
    SharedMemory* sharedMem = (SharedMemory*)sharedMemRaw;
    int* decimalPlaces = sharedMem->decimalPlaces;// 小数位数数组
    long* integers = sharedMem->integers;
    uint64_t* deltas = sharedMem->deltas;

    __shared__ int maxDecimalPlaces;// 最大小数位
    __shared__ long firstValue;// 第一位
    __shared__ int bitCount;// 差分值的最大位宽

    __shared__ int numDeltas; // 新增：实际 delta 数量

    // 计算实际的 delta 数量
    if (tid == 0) {
        numDeltas = 0;
        maxDecimalPlaces = 0;//初始化
        for (int i = 0; i < blockSize * DATA_PER_THREAD && (blockIdxGlobal * blockSize * DATA_PER_THREAD + i) < totalSize; i++) {
            //if ((blockIdxGlobal * blockSize * DATA_PER_THREAD + i) > 0) {//这是所有块用一个firstvalue
            if(i>0){
                numDeltas++;
            }
        }
        //printf("ND:%d\n",numDeltas);
    }
    __syncthreads();

    // 1. 每个线程计算其32个数据项的小数位数:可能会有错误的
    for (int i = 0; i < DATA_PER_THREAD; i++) {
        int idx = startIdx + i;
        if (idx < endIdx) {
            decimalPlaces[startIdxInBlock + i] = getDecimalPlaces(input[idx]);
        } 
        // else {
        //     decimalPlaces[startIdxInBlock + i] = 0;
        // }

        //printf("i:%d,mD:%d\n",startIdx + i,decimalPlaces[startIdxInBlock + i]);
    }
    __syncthreads();

    // 归约找到最大的小数位数
    if (tid == 0) {
        for(int i=0;i<=numDeltas;i++)
        {
            maxDecimalPlaces = device_max(maxDecimalPlaces, decimalPlaces[i]);
        }
        maxDecimalPlaces = device_min(maxDecimalPlaces, 16);
        //printf("maxD:%d\n",maxDecimalPlaces);
    }
    __syncthreads();

    // 2. 转换浮点数为整数
    for (int i = 0; i < DATA_PER_THREAD; i++) {
        int idx = startIdx + i;
        if (idx < endIdx) {
            if (maxDecimalPlaces > 15) {
                unsigned long long bits = __double_as_longlong(input[idx]);
                integers[startIdxInBlock + i] = static_cast<long>(bits);//当前线程在块中的数据位置
            } else {
                integers[startIdxInBlock + i] = static_cast<long>(round(input[idx] * pow10_table[maxDecimalPlaces]));
            }
        }
        //printf("i:%d v:%lo\n",idx,integers[startIdxInBlock + i]);
    }
    __syncthreads();

    // 3. 计算Delta序列
    if (tid == 0 && startIdx < totalSize) {
        firstValue = integers[0];
    }
    __syncthreads();

    // 计算 deltas 仅限于实际的数据
    for (int i = 0; i < DATA_PER_THREAD; i++) {
        int nid = startIdxInBlock + i;//当前线程在块中的位置
        if (startIdx + i < endIdx && nid>0) {
            deltas[nid - 1] = integers[nid] - integers[nid - 1];
            //printf("i:%d Dv:%lo\n",startIdx + i,deltas[nid - 1]);
        }
        
    }
    __syncthreads();

    // 4. ZigZag编码
    for (int i = 0; i < DATA_PER_THREAD; i++) {
        int deltaIdx = startIdxInBlock + i;
        if (deltaIdx < numDeltas ) {
            deltas[deltaIdx] = zigzag_encode_cuda(deltas[deltaIdx]);
            //printf("i:%d dv:%lo\n",startIdx + i,deltas[deltaIdx]);
        }
    }
    __syncthreads();

    // 5. 计算每个块的bitCount
    if (tid == 0) {
        bitCount = 0;
        uint64_t maxDelta = 0;
        for (int i = 0; i < numDeltas && i < (blockSize * DATA_PER_THREAD - 1); i++) {
            maxDelta = device_max_uint64(maxDelta, deltas[i]);
        }
        //printf("maxDelta:%d\n",maxDelta);
        while (maxDelta > 0) {
            maxDelta >>= 1;
            bitCount++;
        }
        //printf("bitcount:%d\n",bitCount);
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
        // 计算 bitSize
        uint64_t bitSize = 64ULL + 64ULL + 8ULL + 8ULL + ((uint64_t)numDeltas) * ((uint64_t)bitCount);
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

        for(int i = 0; i < numDeltas; i++) {
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
        // 不需要填充剩余的字节，因为 bitSize 已经正确反映了实际的数据
    }
}


// __global__ void compressBlockKernel(const double* input, int blockSize, int totalSize, unsigned char* output, uint64_t* bitSizes) {
//     int blockIdxGlobal = blockIdx.x;
//     int tid = threadIdx.x;

//     // 每个线程处理32个数据项
//     int startIdx = blockIdxGlobal * blockSize * DATA_PER_THREAD + tid * DATA_PER_THREAD;
//     int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);

//     extern __shared__ unsigned char sharedMemRaw[];
//     SharedMemory* sharedMem = (SharedMemory*)sharedMemRaw;
//     int* decimalPlaces = sharedMem->decimalPlaces;
//     long* integers = sharedMem->integers;
//     uint64_t* deltas = sharedMem->deltas;

//     __shared__ int maxDecimalPlaces;
//     __shared__ long firstValue;
//     __shared__ int bitCount;

//     // 1. 每个线程计算其32个数据项的小数位数
//     for (int i = 0; i < DATA_PER_THREAD; i++) {
//         int idx = startIdx + i;
//         if (idx < endIdx) {
//             decimalPlaces[tid * DATA_PER_THREAD + i] = getDecimalPlaces(input[idx]);
//         } else {
//             decimalPlaces[tid * DATA_PER_THREAD + i] = 0;
//         }
//     }
//     __syncthreads();

//     // 归约找到最大的小数位数
//     if (tid == 0) {
//         maxDecimalPlaces = 0;
//         for (int i = 0; i < blockSize * DATA_PER_THREAD && (blockIdxGlobal * blockSize * DATA_PER_THREAD + i) < totalSize; i++) {
//             maxDecimalPlaces = device_max(maxDecimalPlaces, decimalPlaces[i]);
//         }
//         maxDecimalPlaces = device_min(maxDecimalPlaces, 16);
//     }
//     __syncthreads();

//     // 2. 转换浮点数为整数
//     for (int i = 0; i < DATA_PER_THREAD; i++) {
//         int idx = startIdx + i;
//         if (idx < endIdx) {
//             if (maxDecimalPlaces > 15) {
//                 unsigned long long bits = __double_as_longlong(input[idx]);
//                 integers[tid * DATA_PER_THREAD + i] = static_cast<long>(bits);
//             } else {
//                 integers[tid * DATA_PER_THREAD + i] = static_cast<long>(round(input[idx] * pow10_table[maxDecimalPlaces]));
//             }
//         }
//     }
//     __syncthreads();

//     // 3. 计算Delta序列
//     if (tid == 0 && blockIdxGlobal * blockSize * DATA_PER_THREAD < totalSize) {
//         firstValue = integers[0];
//     }
//     __syncthreads();

//     for (int i = 0; i < DATA_PER_THREAD; i++) {
//         int idx = startIdx + i;
//         if (idx < endIdx) {
//             if (i != 0 || (blockIdxGlobal * blockSize * DATA_PER_THREAD + idx) != 0) {
//                 if (idx > 0) {
//                     deltas[tid * DATA_PER_THREAD + i - 1] = integers[tid * DATA_PER_THREAD + i] - integers[tid * DATA_PER_THREAD + i - 1];
//                 }
//             }
//         }
//     }
//     __syncthreads();

//     // 4. ZigZag编码
//     for (int i = 0; i < DATA_PER_THREAD - 1; i++) {
//         int deltaIdx = tid * DATA_PER_THREAD + i;
//         if (deltaIdx < blockSize * DATA_PER_THREAD - 1 && (blockIdxGlobal * blockSize * DATA_PER_THREAD + deltaIdx) < totalSize - 1) {
//             deltas[deltaIdx] = zigzag_encode_cuda(deltas[deltaIdx]);
//         }
//     }
//     __syncthreads();

//     // 5. 计算每个块的bitCount
//     if (tid == 0) {
//         bitCount = 0;
//         uint64_t maxDelta = 0;
//         for (int i = 0; i < blockSize * DATA_PER_THREAD - 1 && (blockIdxGlobal * blockSize * DATA_PER_THREAD + i) < totalSize - 1; i++) {
//             maxDelta = device_max_uint64(maxDelta, deltas[i]);
//         }
//         while (maxDelta > 0) {
//             maxDelta >>= 1;
//             bitCount++;
//         }

//         if (bitCount == 0) bitCount = 1;
//         bitCount = device_min(bitCount, MAX_BITCOUNT);
//     }
//     __syncthreads();

//     // 6. 写入输出
//     if (tid == 0) {
//         // 计算 bitSize
//         uint64_t bitSize = 64ULL + 64ULL + 8ULL + 8ULL + ((uint64_t)(blockSize * DATA_PER_THREAD - 1)) * ((uint64_t)bitCount);
//         bitSizes[blockIdxGlobal] = bitSize;

//         if (bitSize > MAX_BITSIZE_PER_BLOCK) {
//             bitSizes[blockIdxGlobal] = 0;
//             return;
//         }

//         int outputIdx = blockIdxGlobal * MAX_BYTES_PER_BLOCK;

//         // 写入 bitSize
//         for (int i = 0; i < 8; i++) {
//             output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;
//         }

//         // 写入 firstValue
//         unsigned long long firstValueBits = 0;
//         memcpy(&firstValueBits, &firstValue, sizeof(long));
//         for (int i = 0; i < 8; i++) {
//             output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;
//         }

//         // 写入 maxDecimalPlaces 和 bitCount
//         output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
//         output[outputIdx + 17] = static_cast<unsigned char>(bitCount);

//         // 写入 Delta 差分值序列
//         int deltaStartIdx = outputIdx + 18;
//         int bitOffset = 0;
//         unsigned char currentByte = 0;

//         for(int i = 0; i < blockSize * DATA_PER_THREAD - 1; i++) {
//             uint64_t deltaValue = deltas[i] & ((1ULL << bitCount) -1);

//             currentByte |= (deltaValue << bitOffset);
//             bitOffset += bitCount;

//             while(bitOffset >=8) {
//                 output[deltaStartIdx++] = currentByte;
//                 bitOffset -=8;
//                 currentByte = (deltaValue >> (bitCount - bitOffset)) &0xFF;
//             }
//         }

//         if(bitOffset >0) {
//             output[deltaStartIdx++] = currentByte;
//         }

//         while(deltaStartIdx < MAX_BYTES_PER_BLOCK){
//             output[deltaStartIdx++] = 0x00;
//         }
//     }
// }


// 初始化设备内存
void GDFCompressor::setupDeviceMemory(const std::vector<double>& input, double*& d_input, unsigned char*& d_output, uint64_t*& d_bitSizes) {
    size_t inputSize = input.size();
    cudaCheckError(cudaMalloc((void**)&d_input, inputSize * sizeof(double)));
    int numBlocks = (inputSize + BLOCK_SIZE_G * DATA_PER_THREAD - 1) / (BLOCK_SIZE_G * DATA_PER_THREAD);
    cudaCheckError(cudaMalloc((void**)&d_output, numBlocks * MAX_BYTES_PER_BLOCK * sizeof(unsigned char)));
    cudaCheckError(cudaMalloc((void**)&d_bitSizes, numBlocks * sizeof(uint64_t)));
    
    cudaCheckError(cudaMemcpy(d_input, input.data(), inputSize * sizeof(double), cudaMemcpyHostToDevice));
}

// 释放设备内存
void GDFCompressor::freeDeviceMemory(double* d_input, unsigned char* d_output, uint64_t* d_bitSizes) {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bitSizes);
}

// 主压缩函数
void GDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output) {
    size_t inputSize = input.size();
    if (inputSize == 0) return;

    int blockSize = BLOCK_SIZE_G; // 32
    size_t numBlocks = (inputSize + blockSize * DATA_PER_THREAD - 1) / (blockSize * DATA_PER_THREAD);

    double* d_input;
    unsigned char* d_output;
    uint64_t* d_bitSizes;

    // 分配设备内存
    setupDeviceMemory(input, d_input, d_output, d_bitSizes);

    // 启动核函数
    size_t sharedMemSize = sizeof(SharedMemory);
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

    // 复制 d_output 到主机的临时缓冲区
    std::vector<unsigned char> tempOutput(numBlocks * MAX_BYTES_PER_BLOCK);
    cudaCheckError(cudaMemcpy(tempOutput.data(), d_output, numBlocks * MAX_BYTES_PER_BLOCK * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // 将压缩数据整理到最终的 output 中
    size_t currentBitPos = 0;
    for (size_t i = 0; i < numBlocks; i++) {
        uint64_t bitSize = bitSizes[i];
        if (bitSize == 0) {
            continue;
        }
        int byteSize = (bitSize + 7) / 8;
        size_t srcIdx = i * MAX_BYTES_PER_BLOCK;
        size_t dstByteIdx = currentBitPos / 8;
        int dstBitOffsetRemainder = currentBitPos % 8;

        for (int b = 0; b < byteSize; b++) {
            if (dstBitOffsetRemainder == 0) {
                if (dstByteIdx + b < output.size()) {
                    output[dstByteIdx + b] = tempOutput[srcIdx + b];
                }
            } else {
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
