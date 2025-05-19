//
// Created by lz on 24-9-26.
// cuCompressor/src/gpu/GDFDecompressor.cu
//

// GDFDecompressor.cu
#include "GDFDecompressor.cuh"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>


// ZigZag 解码
__device__ int64_t zigzag_decode(uint64_t n) {
    return (n >> 1) ^ -(n & 1);
}

// 设备端按位读取函数
__device__ uint64_t readBitsDevice(const unsigned char* buffer, size_t& bitPos, int n) {
    uint64_t value = 0;
    for (int i = 0; i < n; ++i) {
        size_t byteIdx = bitPos / 8;
        size_t bitIdx = bitPos % 8;
        uint8_t bit = (buffer[byteIdx] >> bitIdx) & 1;
        value |= (static_cast<uint64_t>(bit) << i);
        bitPos++;
    }
    return value;
}

// 核函数实现
__global__ void decompressKernel(
    const unsigned char* compressedData, // 压缩数据
    double* output,                      // 解压后数据输出
    const int* offsets,                  // 每个块的偏移
    int numBlocks,                       // 块数
    int numDatas                         // 所有块的总数据个数，每个数据块最大为1024
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= numBlocks) return;

    const unsigned char* blockData = compressedData + offsets[blockId];
    size_t bitPos = 0;

    int numData = min(numDatas-blockId*1024, 1024);

    //读取bitSize和firstValue
    uint64_t bitSize = readBitsDevice(blockData, bitPos, 64);
    int64_t firstValue = static_cast<int64_t>(readBitsDevice(blockData, bitPos, 64));//get
    
    //读取maxDecimalPlaces
    uint64_t maxDecimalPlacesRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char maxDecimalPlaces = static_cast<unsigned char>(maxDecimalPlacesRaw);
    
    //读取 maxBeta (8 位)
    uint64_t maxBetaRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char maxBeta = static_cast<unsigned char>(maxBetaRaw);

    //读取bitCount
    uint64_t bitCountRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = static_cast<unsigned char>(bitCountRaw);

    if (bitCount == 0) {
        printf("Error: bitCount is zero in block %d.\n", blockId);
        return;
    }

    uint64_t flag1 = readBitsDevice(blockData, bitPos, 64);


    int dataByte = (numData + 7) / 8;
    int flag2Size = (dataByte + 7) / 8;
    
    uint8_t result[64][128];
    uint8_t flag2[64][128];
    memset(flag2, 0, sizeof(flag2));
    for (int i = 0; i < bitCount; i++) {// 循环判断每一列
        if ((flag1 & (1ULL << i)) != 0) {// i列稀疏

            for (int z = 0; z < flag2Size * 8; z++) {
                flag2[i][z] = readBitsDevice(blockData, bitPos, 1);
            }
            for (int j = 0; j < dataByte; j++) {
                if (i >= 64 || j >= 128) {
                    printf("Index out of bounds: i=%d,j=%d\n",i,j);
                    return;
                    continue;
                }
                if (flag2[i][j] != 0) {
                    result[i][j] = static_cast<unsigned char>(readBitsDevice(blockData, bitPos, 8));
                } else {
                    result[i][j] = 0;
                }
            }
        } else {
            for (int j = 0; j < dataByte; j++) {
                if (i >= 64 || j >= 128) {
                    printf("Index out of bounds: i=%d,j=%d\n",i,j);
                    return;
                    continue;
                }
                result[i][j] = static_cast<unsigned char>(readBitsDevice(blockData, bitPos, 8));
            }
        }
    }

    // 读取deltas
    uint64_t deltasZigzag[1024] = {0};
    for (int i = 0; i < bitCount; i++) {
        for (int j = 0; j < numData; j++) {
            int byteIndex = j / 8;
            int bitIndex = j % 8;
            uint8_t bitValue = (result[i][byteIndex] >> (7 - bitIndex)) & 1;
            deltasZigzag[j] |= (uint64_t(bitValue) << (bitCount - 1 - i));
        }
    }

    int64_t deltas[1024] = {0};
    for (int i = 0; i < numData; i++) {
        deltas[i] = zigzag_decode(deltasZigzag[i]);
    }

    // 得到整数数组进行还原
    int64_t integers[1024] = {0};
    integers[0] = firstValue;
    for (int i = 1; i < numData; i++) {
        integers[i] = integers[i - 1] + deltas[i];
    }

    for (int i = 0; i < numData; i++) {
        double d;
        // if (maxDecimalPlaces > 15) {
        //     uint64_t bits = static_cast<uint64_t>(integers[i]);
        //     memcpy(&d, &bits, sizeof(double));
        // } else {
        //     d = static_cast<double>(integers[i]) / pow(10.0, maxDecimalPlaces);
        // }
        if(maxBeta >15) {
            // 直接将整数位转换为 double
            uint64_t bits = static_cast<uint64_t>(integers[i]);
            memcpy(&d, &bits, sizeof(double));
        }
        else {
            d = static_cast<double>(integers[i]) / std::pow(10.0, static_cast<double>(maxDecimalPlaces));
        }
        output[blockId * 1024 + i] = d;
    }
}

// 主机端解压缩函数
void GDFDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output, int numDatas) {
    size_t dataSize = compressedData.size();

    // 计算每个块的偏移
    std::vector<int> offsets;
    BitReader reader(compressedData);
    while (reader.getBitPos() + 64 + 64 + 8 + 8 + 64 <= dataSize * 8) {
        
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(64);
        reader.advance(bitSize - 64);
    }

    int numBlocks = offsets.size();
    unsigned char* d_compressedData;
    double* d_output;
    int* d_offsets;

    cudaMalloc(&d_compressedData, compressedData.size());
    cudaMalloc(&d_output, numDatas * sizeof(double));
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));

    cudaMemcpy(d_compressedData, compressedData.data(), compressedData.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;

    decompressKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_compressedData, d_output, d_offsets, numBlocks, numDatas);

    output.resize(numDatas);
    std::cout << "endCuda numBlocks: " << numBlocks << std::endl;
    cudaMemcpy(output.data(), d_output, numDatas * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_compressedData);
    cudaFree(d_output);
    cudaFree(d_offsets);
}


// 实现GDFC_decompress函数 - 使用CUDA流进行解压缩
void GDFDecompressor::GDFC_decompress(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream) {
    // 在设备上计算偏移
    std::vector<unsigned char> hostCmpBytes(cmpSize);
    cudaError_t err = cudaMemcpyAsync(hostCmpBytes.data(), d_cmpBytes, cmpSize, cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in GDFC_decompress (cudaMemcpyAsync): " << cudaGetErrorString(err) << std::endl;
        return;
    }
    err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error in GDFC_decompress (cudaStreamSynchronize): " << cudaGetErrorString(err) << std::endl;
        return;
    }
    // 计算块偏移
    printf("计算块偏移\n");
    std::vector<int> offsets;
    BitReader reader(hostCmpBytes);

    size_t totalBits = cmpSize * 8;
    int maxBlocks = (nbEle + 1023) / 1024;  // 估计的最大块数
    
    while (reader.getBitPos() + 64 + 64 + 8 + 8 + 64 <= cmpSize * 8 && offsets.size() < maxBlocks) {
        offsets.push_back(reader.getBitPos() / 8);
        uint64_t bitSize = reader.readBits(64);
        if (bitSize < 64 || bitSize -64 > totalBits - reader.getBitPos()) {
            std::cerr << "Error: Invalid bitSize " << bitSize << " at position " << reader.getBitPos() / 8 
                      << " (total bits: " << totalBits << ", remaining: " << totalBits - reader.getBitPos() << ")" << std::endl;
            break;
        }
        reader.advance(bitSize - 64);
    }

    int numBlocks = offsets.size();
    if (numBlocks == 0) {
        std::cerr << "Error: No valid blocks found in compressed data." << std::endl;
        return;
    }
    std::cout << "解压块数: " << numBlocks << ", 总元素数: " << nbEle << std::endl;

    // 将偏移复制到设备
    int* d_offsets;
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 调用核函数解压
    int threadsPerBlock = 256;
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;

    decompressKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, numBlocks, nbEle);

    // 异步释放内存
    cudaFree(d_offsets);
    
    // 注意: 调用此函数后不需要同步，因为我们使用了异步操作
    // 调用方应该在适当的时候同步流
}