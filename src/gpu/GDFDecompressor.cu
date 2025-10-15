
// 优化后的 Faclon_decompressor.cu
// 主要优化：内存访问模式、并行度、位操作效率、错误处理
//

#include "GDFDecompressor.cuh"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>

// 常量定义
__constant__ double POW10_TABLE[16] = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
    100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 
    1000000000000.0, 10000000000000.0, 100000000000000.0, 1000000000000000.0
};

// ZigZag 解码 - 内联优化
__device__ __forceinline__ int64_t zigzag_decode(uint64_t n) {
    return (int64_t)(n >> 1) ^ -((int64_t)(n & 1));
}

// 优化的按位读取函数 - 减少循环开销
__device__ __forceinline__ uint64_t readBitsDevice(const unsigned char* buffer, size_t& bitPos, int n) {
    if (n == 0) return 0;
    if (n > 64) n = 64; // 防护
    
    uint64_t result = 0;
    size_t startByte = bitPos / 8;
    int startBit = bitPos % 8;
    
    // 优化：尽量按字节对齐读取
    if (startBit == 0 && n >= 8) {
        // 字节对齐情况下的快速读取
        int fullBytes = n / 8;
        for (int i = 0; i < fullBytes; i++) {
            result |= ((uint64_t)buffer[startByte + i]) << (i * 8);
        }
        int remainingBits = n % 8;
        if (remainingBits > 0) {
            uint64_t lastBits = buffer[startByte + fullBytes] & ((1 << remainingBits) - 1);
            result |= lastBits << (fullBytes * 8);
        }
    } else {
        // 非对齐情况的优化读取
        for (int i = 0; i < n; i++) {
            size_t byteIdx = (bitPos + i) / 8;
            int bitIdx = (bitPos + i) % 8;
            uint64_t bit = (buffer[byteIdx] >> bitIdx) & 1ULL;
            result |= (bit << i);
        }
    }
    
    bitPos += n;
    return result;
}

__device__ double decodeDoubleWithSignLast(uint64_t value) {
    uint64_t original = (value >> 1) ^ -((int64_t)(value & 1));
    union {
        uint64_t u;
        double d;
    } val;

    val.u = original;
    return val.d;
}

// 优化的核函数 - 使用更好的内存访问模式和共享内存
__global__ void decompressKernelOptimized(
    const unsigned char* __restrict__ compressedData,
    double* __restrict__ output,
    const int* __restrict__ offsets,
    // int numBlocks,
    int numDatas
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    // if (blockId >= numBlocks) return;
    int numData = min(numDatas - blockId * 1024, 1024);
    
    if (numData <= 0) return;
    // 使用寄存器变量减少全局内存访问
    const unsigned char* blockData = compressedData + offsets[blockId];
    size_t bitPos = 0;


    // 读取头部信息
    uint64_t bitSize = readBitsDevice(blockData, bitPos, 64);
    int64_t firstValue = (int64_t)readBitsDevice(blockData, bitPos, 64);
    unsigned char maxDecimalPlaces = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char maxBeta = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = (unsigned char)readBitsDevice(blockData, bitPos, 8);

    if (bitCount == 0 || bitCount > 64) {
        // 填充默认值而不是返回
        for (int i = 0; i < numData; i++) {
            output[blockId * 1024 + i] = 0.0;
        }
        return;
    }

    uint64_t flag1 = readBitsDevice(blockData, bitPos, 64);
    int dataByte = (numData + 7) / 8;
    int flag2Size = (dataByte + 7) / 8;
    
    // 使用栈上数组，减少内存分配开销
    uint8_t result[64][128];
    uint8_t flag2[64][128];
    
    // 优化的内存初始化
    #pragma unroll
    for (int i = 0; i < 64; i++) {
        #pragma unroll 4
        for (int j = 0; j < 128; j += 4) {
            *((uint32_t*)&flag2[i][j]) = 0;
        }
    }

    // 读取压缩数据 - 优化循环结构
    for (int i = 0; i < bitCount; i++) {
        bool isSparse = (flag1 & (1ULL << i)) != 0;
        
        if (isSparse) {
            // 读取稀疏标志
            for (int z = 0; z < flag2Size * 8; z++) {
                flag2[i][z] = (uint8_t)readBitsDevice(blockData, bitPos, 1);
            }
            
            // 根据稀疏标志读取数据
            for (int j = 0; j < dataByte; j++) {
                if (flag2[i][j] != 0) {
                    result[i][j] = (uint8_t)readBitsDevice(blockData, bitPos, 8);
                } else {
                    result[i][j] = 0;
                }
            }
        } else {
            // 非稀疏情况，直接读取所有字节
            for (int j = 0; j < dataByte; j++) {
                result[i][j] = (uint8_t)readBitsDevice(blockData, bitPos, 8);
            }
        }
    }

    // 重构delta解码 - 使用更高效的位操作
    uint64_t deltasZigzag[1024];
    #pragma unroll 4
    for (int j = 0; j < numData; j++) {
        uint64_t delta = 0;
        int byteIndex = j / 8;
        int bitIndex = 7 - (j % 8); // 预计算位索引
        
        for (int i = 0; i < bitCount; i++) {
            uint8_t bitValue = (result[i][byteIndex] >> bitIndex) & 1;
            delta |= ((uint64_t)bitValue << (bitCount - 1 - i));
        }
        deltasZigzag[j] = delta;
    }

    // 解码和前缀求和 - 合并到一个循环中
    int64_t prevValue = firstValue;
    double scale = (maxBeta > 15) ? 1.0 : 
        (maxDecimalPlaces < 16 ? POW10_TABLE[maxDecimalPlaces] : pow(10.0, maxDecimalPlaces));
    bool useDirectConversion = (maxBeta > 15);
    
    if (useDirectConversion) {
        uint64_t bits = (uint64_t)prevValue;
        output[blockId * 1024] = decodeDoubleWithSignLast(bits);
    } else {
        output[blockId * 1024] = (double)prevValue / scale;
    }
    // output[blockId * 1024] = useDirectConversion ? 
    //     *reinterpret_cast<double*>(&firstValue) : 
    //     (double)firstValue / scale;
    
    for (int i = 1; i < numData; i++) {
        int64_t delta = zigzag_decode(deltasZigzag[i]);
        prevValue += delta;
        
        if (useDirectConversion) {
            uint64_t bits = (uint64_t)prevValue;
            output[blockId * 1024 + i] = decodeDoubleWithSignLast(bits);
        } else {
            output[blockId * 1024 + i] = (double)prevValue / scale;
        }
    }
}

// 优化的主机端解压缩函数，包括CPU-GPU数据传输，主机偏移计算
void GDFDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output, int numDatas) {
    size_t dataSize = compressedData.size();
    if (dataSize == 0 || numDatas <= 0) {
        output.clear();
        return;
    }

    cudaEvent_t kernal_start_event,kernal_end_event;
    cudaEventCreate(&kernal_start_event);
    cudaEventCreate(&kernal_end_event);

    // 预分配offsets向量以减少重分配
    std::vector<int> offsets;
    offsets.reserve((numDatas + 1024) / 1025); // 预估块数
    
    BitReader reader(compressedData);
    size_t totalBits = dataSize * 8;
    
    // 优化的偏移计算 - 与原始代码保持兼容
    while (reader.getBitPos() + 64 + 64 + 8 + 8 + 64 <= totalBits) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(64);
        
        if (bitSize < 64) break;
        
        // 检查是否有足够的位可以跳过
        if (reader.getBitPos() + bitSize - 64 > totalBits) break;
        
        reader.advance(bitSize - 64);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        output.assign(numDatas, 0.0);
        return;
    }

    // 使用CUDA内存池或预分配内存（如果可能）
    unsigned char* d_compressedData;
    double* d_output;
    int* d_offsets;

    // 异步内存分配
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc(&d_compressedData, compressedData.size());
    cudaMalloc(&d_output, numDatas * sizeof(double));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));

    // 异步内存传输
    cudaMemcpyAsync(d_compressedData, compressedData.data(), compressedData.size(), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 优化的网格配置
    int threadsPerBlock = 128; // 减少线程数以增加寄存器使用
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;



    cudaEventRecord(kernal_start_event,stream);
    // 使用优化的核函数
    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_compressedData, d_output, d_offsets,numDatas);
    cudaEventRecord(kernal_end_event,stream);

    // 等待所有操作完成
    cudaEventSynchronize(kernal_end_event);
    float totalTime;
    cudaEventElapsedTime(&totalTime, kernal_start_event, kernal_end_event);
    printf("\n解压核函数运行时间：%f\n",totalTime);

    // 检查核函数执行错误
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "CUDA Kernel Error: " << cudaGetErrorString(kernelErr) << std::endl;
    }

    output.resize(numDatas);
    cudaMemcpyAsync(output.data(), d_output, numDatas * sizeof(double), cudaMemcpyDeviceToHost, stream);
    
    // 同步流
    cudaStreamSynchronize(stream);
    
    // 清理资源
    cudaFree(d_compressedData);
    cudaFree(d_output);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
    
    // std::cout << "Decompressed " << numBlocks << " blocks, " << numDatas << " elements" << std::endl;
}

// 优化的GDFC_decompress函数(流式，主机偏移计算，仅压缩无主机数据传输）
void GDFDecompressor::GDFC_decompress(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;

    cudaEvent_t kernal_start_event,kernal_end_event;
    cudaEventCreate(&kernal_start_event);
    cudaEventCreate(&kernal_end_event);

    // 使用固定大小的临时缓冲区避免动态分配
    static thread_local std::vector<unsigned char> hostCmpBytes;
    hostCmpBytes.resize(cmpSize);

    // 异步内存传输
    cudaError_t err = cudaMemcpyAsync(hostCmpBytes.data(), d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    cudaStreamSynchronize(stream);

    // 优化的偏移计算 - 使用更少的边界检查
    std::vector<int> offsets;
    offsets.reserve((nbEle + 1024) / 1025);

    BitReader reader(hostCmpBytes);
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 64 + 64 + 8 + 8 + 64; // 192 bits

    while (reader.getBitPos() + minHeaderSize <= totalBits && offsets.size() * 1025 < nbEle) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(64);

        if (bitSize < 64) break;

        size_t nextPos = reader.getBitPos() + bitSize - 64;
        if (nextPos > totalBits) break;

        reader.advance(bitSize - 64);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        // 清零输出数据
        cudaMemsetAsync(d_decData, 0, nbEle * sizeof(double), stream);
        return;
    }

    // 使用临时设备内存
    int* d_offsets;
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 优化的核函数调用
    int threadsPerBlock = 128;
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    cudaEventRecord(kernal_start_event,stream);
    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets,  nbEle);

    cudaEventRecord(kernal_end_event,stream);

    // 等待所有操作完成
    cudaEventSynchronize(kernal_end_event);
    float totalTime;
    cudaEventElapsedTime(&totalTime, kernal_start_event, kernal_end_event);
    printf("\n解压核函数运行时间：%f\n",totalTime);
    // 异步清理 - 使用同步版本以保证兼容性
    cudaFree(d_offsets);

    // 错误检查（非阻塞）
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
    cudaEventDestroy(kernal_start_event);
    cudaEventDestroy(kernal_end_event);
}

// 设备端偏移计算核函数
__global__ void calculateOffsetsKernel(const unsigned char* d_cmpBytes,
                                      int* d_offsets,
                                      int* d_numBlocks,
                                      size_t cmpSize,
                                      size_t nbEle) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return; // 单线程执行

    size_t bitPos = 0;
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 216;//192; // 64 + 64 + 8 + 8 + 64 bits
    int offsetCount = 0;
    int maxBlocks = (nbEle + 1024) / 1025;

    while (bitPos + minHeaderSize <= totalBits && offsetCount < maxBlocks) {
        d_offsets[offsetCount] = (bitPos+7) / 8;

        uint64_t bitSize = readBitsDevice(d_cmpBytes, bitPos, 64);

        if (bitSize < 64) break;
        // printf("解压 Chunk %d: offset=%d, size= %dbytes\n",offsetCount,d_offsets[offsetCount],(bitSize+7)/8);
        size_t nextPos = bitPos + bitSize - 64;
        if (nextPos > totalBits) break;

        bitPos = nextPos;
        offsetCount++;
    }

    *d_numBlocks = offsetCount;
}
// 优化后的流式解压接口，采用核函数计算偏移
void GDFDecompressor::GDFC_decompress_stream_optimized(double* d_decData,
                                     unsigned char* d_cmpBytes,
                                     size_t nbEle,
                                     size_t cmpSize,
                                     cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;

    // 预估最大块数
    int maxBlocks = (nbEle + 1024) / 1025;
    if(maxBlocks<=0)
    {
        return;
    }
    // 分配设备内存用于偏移和块数
    int* d_offsets;
    int* d_numBlocks;

    cudaError_t err1 = cudaMallocAsync(&d_offsets, (maxBlocks) * sizeof(int), stream);
    cudaError_t err2 = cudaMallocAsync(&d_numBlocks, sizeof(int), stream);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "CUDA malloc error" << std::endl;
        if (err1 == cudaSuccess) cudaFreeAsync(d_offsets, stream);
        if (err2 == cudaSuccess) cudaFreeAsync(d_numBlocks, stream);
        return;
    }

    // 初始化块数为0
    cudaMemsetAsync(d_numBlocks, 0, sizeof(int), stream);

    // 在GPU上计算偏移
    calculateOffsetsKernel<<<1, 1, 0, stream>>>(
        d_cmpBytes, d_offsets, d_numBlocks, cmpSize, nbEle);

    // 检查核函数执行
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        // std::cerr << "Offset calculation kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaFreeAsync(d_offsets, stream);
        cudaFreeAsync(d_numBlocks, stream);
        return;
    }
    // 调用解压核函数
    int threadsPerBlock = 128;
    int blocksPerGrid = (maxBlocks+ threadsPerBlock - 1) / threadsPerBlock;

    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, nbEle);

    // 异步清理内存
    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_numBlocks, stream);

    // 错误检查
    kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Decompression kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
}


// 将小端字节数组转换为 uint64_t
uint64_t bytesToULong(const unsigned char* bytes) {
    uint64_t val = 0;
    for(int i = 0; i < 8; i++) {
        val |= ((uint64_t)bytes[i]) << (i * 8);
    }
    return val;
}

// 将小端字节数组转换为 int64_t
int64_t bytesToLong(const unsigned char* bytes) {
    int64_t val = 0;
    for(int i = 0; i < 8; i++) {
        val |= ((int64_t)bytes[i]) << (i * 8);
    }
    return val;
}

// ZigZag 解码函数
int64_t zigzag_decode1(uint64_t n) {
    return (n >> 1) ^ -(n & 1);
}

// 将位表示转换为双精度浮点数（主机端）
double bitsToDoubleHost(uint64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}
// 采用偏移量计算 + 块级解压的简化方案

// 无打包格式的GPU解压核函数 - 修正为字节+位混合读取
__global__ void decompressKernelNoPack(
    const unsigned char* __restrict__ compressedData,
    double* __restrict__ output,
    const int* __restrict__ offsets,
    // int numBlocks,
    int numDatas
) {
    // int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    // if (blockId >= numBlocks) return;

    // // 每个线程处理一个压缩块
    // const unsigned char* blockData = compressedData + offsets[blockId];
    
    // // 按字节读取头部（小端格式）
    // uint64_t bitSize = 0;
    // for (int i = 0; i < 8; i++) {
    //     bitSize |= ((uint64_t)blockData[i]) << (i * 8);
    // }
    
    // int64_t firstValue = 0;
    // for (int i = 0; i < 8; i++) {
    //     firstValue |= ((int64_t)blockData[8 + i]) << (i * 8);
    // }
    
    // unsigned char maxDecimalPlaces = blockData[16];
    // unsigned char maxBeta = blockData[17];
    // unsigned char bitCount = blockData[18];

    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    // if (blockId >= numBlocks) return;
    int numData = min(numDatas - blockId * 1024, 1024);
    
    if (numData <= 0) return;
    // 使用寄存器变量减少全局内存访问
    const unsigned char* blockData = compressedData + offsets[blockId];
    size_t bitPos = 0;


    // 读取头部信息
    uint64_t bitSize = readBitsDevice(blockData, bitPos, 64);
    int64_t firstValue = (int64_t)readBitsDevice(blockData, bitPos, 64);
    unsigned char maxDecimalPlaces = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char maxBeta = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = (unsigned char)readBitsDevice(blockData, bitPos, 8);
    

    if (bitCount == 0 || bitCount > 64) {
        for (int i = 0; i < numData; i++) {
            output[blockId * 1024 + i] = 0.0;
        }
        return; // 无效的bitCount，跳过这个块
    }
    // 计算delta信息

    // 计算输出起始位置
    // uint64_t deltasZigzag[1024];
    size_t outputOffset = blockId * 1024;
    // if (outputOffset >= totalElements) return;
    
    // int actualElements = min(numDeltas + 1, (int)(totalElements - outputOffset));
    // actualElements=min(actualElements,1025);
    // 计算解码参数
    double scale = (maxBeta > 15) ? 1.0 : 
        (maxDecimalPlaces < 16 ? POW10_TABLE[maxDecimalPlaces] : pow(10.0, maxDecimalPlaces));
    bool useDirectConversion = (maxBeta > 15);
    
    // 输出第一个值
    if (useDirectConversion) {
        uint64_t bits = (uint64_t)firstValue;
        output[outputOffset] = decodeDoubleWithSignLast(bits);
    } else {
        output[outputOffset] = (double)firstValue / scale;
    }
    
    // 111111
    // 重建并输出剩余值
    int64_t prevValue = firstValue;
    for (int i = 1; i < numData; i++) {
        // 从delta起始位置读取delta值
        uint64_t deltaZigzag = readBitsDevice(blockData, bitPos, bitCount);
        int64_t delta = zigzag_decode(deltaZigzag);
        
        // 重建值
        prevValue += delta;
        
        // 转换并输出
        if (useDirectConversion) {
            uint64_t bits = (uint64_t)prevValue;
            output[outputOffset + i] = decodeDoubleWithSignLast(bits);
        } else {
            output[outputOffset + i] = (double)prevValue / scale;
        }
    }
}

// 主机端实现 - 采用您建议的逻辑
void GDFDecompressor::GDFC_decompress_no_pack(double* d_decData,
                                     unsigned char* d_cmpBytes,
                                     size_t nbEle,
                                     size_t cmpSize,
                                     cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;

    // 预估最大块数
    int maxBlocks = (nbEle + 1023) / 1024;
    if(maxBlocks<=0)
    {
        return;
    }
    // 分配设备内存用于偏移和块数
    int* d_offsets;
    int* d_numBlocks;

    cudaError_t err1 = cudaMallocAsync(&d_offsets, (maxBlocks) * sizeof(int), stream);
    cudaError_t err2 = cudaMallocAsync(&d_numBlocks, sizeof(int), stream);

    if (err1 != cudaSuccess || err2 != cudaSuccess) {
        std::cerr << "CUDA malloc error" << std::endl;
        if (err1 == cudaSuccess) cudaFreeAsync(d_offsets, stream);
        if (err2 == cudaSuccess) cudaFreeAsync(d_numBlocks, stream);
        return;
    }

    // 初始化块数为0
    cudaMemsetAsync(d_numBlocks, 0, sizeof(int), stream);

    // 在GPU上计算偏移
    calculateOffsetsKernel<<<1, 1, 0, stream>>>(
        d_cmpBytes, d_offsets, d_numBlocks, cmpSize, nbEle);

    // 检查核函数执行
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        // std::cerr << "Offset calculation kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
        cudaFreeAsync(d_offsets, stream);
        cudaFreeAsync(d_numBlocks, stream);
        return;
    }
    // 调用解压核函数
    int threadsPerBlock = 128;
    int blocksPerGrid = (maxBlocks+ threadsPerBlock - 1) / threadsPerBlock;

    decompressKernelNoPack<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, nbEle);

    // 异步清理内存
    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_numBlocks, stream);

    // 错误检查
    kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Decompression kernel error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
}

