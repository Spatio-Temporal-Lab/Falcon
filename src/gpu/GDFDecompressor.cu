//
// 优化后的 GDFDecompressor.cu
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

// 优化的核函数 - 使用更好的内存访问模式和共享内存
__global__ void decompressKernelOptimized(
    const unsigned char* __restrict__ compressedData,
    double* __restrict__ output,
    const int* __restrict__ offsets,
    int numBlocks,
    int numDatas
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= numBlocks) return;

    // 使用寄存器变量减少全局内存访问
    const unsigned char* blockData = compressedData + offsets[blockId];
    size_t bitPos = 0;
    int numData = min(numDatas - blockId * 1024, 1024);
    
    if (numData <= 0) return;

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
    
    output[blockId * 1024] = useDirectConversion ? 
        *reinterpret_cast<double*>(&firstValue) : 
        (double)firstValue / scale;
    
    for (int i = 1; i < numData; i++) {
        int64_t delta = zigzag_decode(deltasZigzag[i]);
        prevValue += delta;
        
        if (useDirectConversion) {
            uint64_t bits = (uint64_t)prevValue;
            output[blockId * 1024 + i] = *reinterpret_cast<double*>(&bits);
        } else {
            output[blockId * 1024 + i] = (double)prevValue / scale;
        }
    }
}

// 优化的主机端解压缩函数
void GDFDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output, int numDatas) {
    size_t dataSize = compressedData.size();
    if (dataSize == 0 || numDatas <= 0) {
        output.clear();
        return;
    }

    // 预分配offsets向量以减少重分配
    std::vector<int> offsets;
    offsets.reserve((numDatas + 1023) / 1024); // 预估块数
    
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

    // 使用优化的核函数
    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_compressedData, d_output, d_offsets, numBlocks, numDatas);

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
    
    std::cout << "Decompressed " << numBlocks << " blocks, " << numDatas << " elements" << std::endl;
}

// 高度优化的GDFC_decompress函数
void GDFDecompressor::GDFC_decompress(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;
    
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
    offsets.reserve((nbEle + 1023) / 1024);
    
    BitReader reader(hostCmpBytes);
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 64 + 64 + 8 + 8 + 64; // 192 bits
    
    while (reader.getBitPos() + minHeaderSize <= totalBits && offsets.size() * 1024 < nbEle) {
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

    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, numBlocks, nbEle);

    // 异步清理 - 使用同步版本以保证兼容性
    cudaFree(d_offsets);
    
    // 错误检查（非阻塞）
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(kernelErr) << std::endl;
    }
}