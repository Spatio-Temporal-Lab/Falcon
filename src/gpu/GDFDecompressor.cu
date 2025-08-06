
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
    size_t minHeaderSize = 192; // 64 + 64 + 8 + 8 + 64 bits
    int offsetCount = 0;
    int maxBlocks = (nbEle + 1023) / 1024;

    while (bitPos + minHeaderSize <= totalBits && offsetCount < maxBlocks) {
        d_offsets[offsetCount] = bitPos / 8;

        uint64_t bitSize = readBitsDevice(d_cmpBytes, bitPos, 64);

        if (bitSize < 64) break;

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
// 没有位打包的解压
void GDFDecompressor::decompress_nopack(const std::vector<unsigned char>& compressedData, std::vector<double>& output) {
    BitReader reader(compressedData);
    size_t dataSize = compressedData.size() * 8; // 总位数
    int blockNumber = 0;

    while(reader.getBitPos() + 64 + 64 + 8 + 8 <= dataSize) { // 确保有足够的元数据
        // 1. 读取 bitSize (64 位)
        uint64_t bitSize = reader.readBits(64);

        // 2. 读取 firstValue (64 位)
        int64_t firstValue = static_cast<int64_t>(reader.readBits(64));

        // 3. 读取 maxDecimalPlaces (8 位)
        uint64_t maxDecimalPlacesRaw = reader.readBits(8);
        unsigned char maxDecimalPlaces = static_cast<unsigned char>(maxDecimalPlacesRaw);

        // 4. 读取 bitCount (8 位)
        uint64_t bitCountRaw = reader.readBits(8);
        unsigned char bitCount = static_cast<unsigned char>(bitCountRaw);

        // 检查 bitCount 是否为零
        if(bitCount ==0) {
            std::cerr << "Error: bitCount is zero, invalid compressed data in block " << blockNumber << "." << std::endl;
            break;
        }

        // 计算 Delta 序列的位数和数量
        uint64_t delta_bits = bitSize - 64ULL - 64ULL - 8ULL - 8ULL;
        if(delta_bits <=0) {
            std::cerr << "Warning: delta_bits calculated as non-positive. Interpreting bitSize as delta_bits in block " << blockNumber << "." << std::endl;
            delta_bits = bitSize;
        }

        int numDeltas = static_cast<int>(delta_bits / bitCount);
        if(numDeltas <=0) {
            std::cerr << "Warning: numDeltas <= 0 in block " << blockNumber << "." << std::endl;
            blockNumber++;
            continue;
        }

        // 调试输出
        // std::cout << "Block " << blockNumber++ << " Metadata:" << std::endl;
        // std::cout << "bitSize: " << bitSize << std::endl;
        // std::cout << "firstValue: " << firstValue << std::endl;
        // std::cout << "maxDecimalPlaces: " << static_cast<int>(maxDecimalPlaces) << std::endl;
        // std::cout << "bitCount: " << static_cast<int>(bitCount) << std::endl;
        // std::cout << "delta_bits: " << delta_bits << std::endl;
        // std::cout << "numDeltas: " << numDeltas << std::endl;

        // 5. 读取 Delta 序列
        std::vector<uint64_t> deltasZigzag(numDeltas, 0);
        for(int i =0; i < numDeltas; i++) {
            if(reader.getBitPos() + bitCount > dataSize) {
                std::cerr << "Error: Not enough data for Delta " << i << " in block " << blockNumber-1 << "." << std::endl;
                break;
            }
            deltasZigzag[i] = reader.readBits(bitCount);
        }

        // 6. ZigZag 解码
        std::vector<int64_t> deltas(numDeltas, 0);
        for(int i =0; i < numDeltas; i++) {
            deltas[i] = zigzag_decode1(deltasZigzag[i]);
        }

        // 7. 重建整数序列
        std::vector<int64_t> integers;
        integers.reserve(numDeltas +1);
        integers.push_back(firstValue);
        for(int i =0; i < numDeltas; i++) {
            integers.push_back(integers.back() + deltas[i]);
        }

        // 8. 转换回双精度浮点数
        for(auto val : integers) {
            double d;
            if(maxDecimalPlaces >15) {
                // 直接将整数位转换为 double
                uint64_t bits = static_cast<uint64_t>(val);
                d = bitsToDoubleHost(bits);
            }
            else {
                d = static_cast<double>(val) / std::pow(10.0, static_cast<double>(maxDecimalPlaces));
            }
            output.push_back(d);
        }

        //std::cout << "Block " << (blockNumber-1) << " decompressed successfully." << std::endl;
    }

    // 处理剩余位（如果有）
    if(reader.getBitPos() < dataSize) {
        //std::cerr << "Warning: " << (dataSize - reader.getBitPos()) << " remaining bits not processed." << std::endl;
    }
}

// 采用偏移量计算 + 块级解压的简化方案

// 无打包格式的GPU解压核函数 - 修正为字节+位混合读取
__global__ void decompressKernelNoPack(
    const unsigned char* __restrict__ compressedData,
    double* __restrict__ output,
    const int* __restrict__ offsets,
    int numBlocks,
    size_t totalElements
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= numBlocks) return;

    // 每个线程处理一个压缩块
    const unsigned char* blockData = compressedData + offsets[blockId];
    
    // 按字节读取头部（小端格式）
    uint64_t bitSize = 0;
    for (int i = 0; i < 8; i++) {
        bitSize |= ((uint64_t)blockData[i]) << (i * 8);
    }
    
    int64_t firstValue = 0;
    for (int i = 0; i < 8; i++) {
        firstValue |= ((int64_t)blockData[8 + i]) << (i * 8);
    }
    
    unsigned char maxDecimalPlaces = blockData[16];
    unsigned char maxBeta = blockData[17];
    unsigned char bitCount = blockData[18];

    if (bitCount == 0 || bitCount > 64) {
        return; // 无效的bitCount，跳过这个块
    }

    // 计算delta信息
    uint64_t header_bits = 152; // 64+64+8+8+8
    if (bitSize < header_bits) return;
    
    uint64_t delta_bits = bitSize - header_bits;
    int numDeltas = (int)(delta_bits / bitCount);
    if (numDeltas <= 0) return;

    // 计算输出起始位置
    size_t outputOffset = (size_t)blockId * 1024;
    if (outputOffset >= totalElements) return;
    
    int actualElements = min(numDeltas + 1, (int)(totalElements - outputOffset));
    actualElements=min(actualElements,1024);
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
    
    // delta数据从第19字节开始，按位读取
    size_t deltaBitPos = 19 * 8; // 19字节 = 152位
    
    // 重建并输出剩余值
    int64_t currentValue = firstValue;
    for (int i = 1; i < actualElements && i <= numDeltas; i++) {
        // 从delta起始位置读取delta值
        uint64_t deltaZigzag = readBitsDevice(blockData, deltaBitPos, bitCount);
        int64_t delta = zigzag_decode(deltaZigzag);
        
        // 重建值
        currentValue += delta;
        
        // 转换并输出
        if (useDirectConversion) {
            uint64_t bits = (uint64_t)currentValue;
            output[outputOffset + i] = decodeDoubleWithSignLast(bits);
        } else {
            output[outputOffset + i] = (double)currentValue / scale;
        }
    }
}

// 主机端实现 - 采用您建议的逻辑
void GDFDecompressor::GDFC_decompress_no_pack(
    double* d_decData,          // 解压输出缓冲
    unsigned char* d_cmpBytes,  // 压缩输入缓冲（设备端）
    size_t nbEle,               // 原始元素数量
    size_t cmpSize,             // 压缩数据大小
    cudaStream_t stream         // CUDA流
) {
    if (nbEle == 0 || cmpSize == 0) return;

    cudaEvent_t kernal_start_event, kernal_end_event;
    cudaEventCreate(&kernal_start_event);
    cudaEventCreate(&kernal_end_event);

    // std::cout << "开始无打包解压，数据: " << nbEle << " 元素, " << cmpSize << " 字节" << std::endl;

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

    // 修正的偏移计算 - 头部按字节存储，只有delta按位存储
    std::vector<int> offsets;
    offsets.reserve((nbEle + 1023) / 1024);

    size_t totalBytes = cmpSize;
    size_t minHeaderBytes = 19; // 8+8+1+1+1 = 19字节头部
    
    // std::cout << "解析偏移量，最小头大小: " << minHeaderBytes << " 字节..." << std::endl;

    int blockCount = 0;
    size_t totalParsedElements = 0;
    size_t currentBytePos = 0;
    
    while (currentBytePos + minHeaderBytes <= totalBytes && totalParsedElements < nbEle) {
        offsets.push_back(currentBytePos);
        
        // 按字节读取头部（小端格式）
        uint64_t bitSize = 0;
        for (int i = 0; i < 8; i++) {
            bitSize |= ((uint64_t)hostCmpBytes[currentBytePos + i]) << (i * 8);
        }
        
        int64_t firstValue = 0;
        for (int i = 0; i < 8; i++) {
            firstValue |= ((int64_t)hostCmpBytes[currentBytePos + 8 + i]) << (i * 8);
        }
        
        unsigned char maxDecimalPlaces = hostCmpBytes[currentBytePos + 16];
        unsigned char maxBeta = hostCmpBytes[currentBytePos + 17];
        unsigned char bitCount = hostCmpBytes[currentBytePos + 18];
        
        // std::cout << "块 " << blockCount << " - 字节位置: " << currentBytePos 
        //           << ", bitSize: " << bitSize << std::endl;
        
        if (bitCount == 0 || bitCount > 64) {
            std::cerr << "无效的bitCount: " << (int)bitCount 
                      << "，在字节位置: " << (currentBytePos + 18) << std::endl;
            break;
        }
        
        // 计算这个块的元素数量
        // bitSize包含头部152位，delta部分需要减去头部
        uint64_t header_bits = 152; // 64+64+8+8+8
        if (bitSize < header_bits) {
            std::cerr << "bitSize太小: " << bitSize << " < " << header_bits << std::endl;
            break;
        }
        
        uint64_t delta_bits = bitSize - header_bits;
        int numDeltas = (int)(delta_bits / bitCount);
        int blockElements = numDeltas + 1; // +1 for firstValue
        
        // std::cout << "  bitCount: " << (int)bitCount 
        //           << ", maxDecimal: " << (int)maxDecimalPlaces
        //           << ", maxBeta: " << (int)maxBeta
        //           << ", delta_bits: " << delta_bits
        //           << ", 元素数: " << blockElements << std::endl;
        
        totalParsedElements += blockElements;
        
        // 跳到下一个块 - bitSize是总位数，转换为字节数
        size_t blockSizeBytes = (bitSize + 7) / 8; // 向上取整
        size_t nextBytePos = currentBytePos + blockSizeBytes;
        
        if (nextBytePos > totalBytes) {
            std::cerr << "块超出范围: nextBytePos=" << nextBytePos << " > totalBytes=" << totalBytes << std::endl;
            break;
        }
        
        // 更新位置到下一个块
        currentBytePos = nextBytePos;
        blockCount++;
        
        // 防止无限循环
        if (blockCount > (nbEle / 100 + 1000)) {
            std::cerr << "块数量异常，可能存在解析错误" << std::endl;
            break;
        }
    }

    int numBlocks = offsets.size();
    // std::cout << "找到 " << numBlocks << " 个块，解析元素总数: " << totalParsedElements << std::endl;
    
    if (numBlocks == 0) {
        std::cerr << "未找到有效块，清零输出" << std::endl;
        cudaMemsetAsync(d_decData, 0, nbEle * sizeof(double), stream);
        return;
    }

    // 使用设备内存传输偏移
    int* d_offsets;
    cudaMalloc(&d_offsets, offsets.size() * sizeof(int));
    cudaMemcpyAsync(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 核函数调用
    int threadsPerBlock = 128;
    int blocksPerGrid = (numBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    // std::cout << "启动GPU核函数: " << blocksPerGrid << " 网格, " 
    //           << threadsPerBlock << " 线程/块" << std::endl;
    
    cudaEventRecord(kernal_start_event, stream);
    decompressKernelNoPack<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, numBlocks, nbEle);
    cudaEventRecord(kernal_end_event, stream);

    // 等待完成
    cudaEventSynchronize(kernal_end_event);
    float totalTime;
    cudaEventElapsedTime(&totalTime, kernal_start_event, kernal_end_event);
    // printf("无打包解压核函数运行时间：%f ms\n", totalTime);
    
    // 清理
    cudaFree(d_offsets);
    cudaEventDestroy(kernal_start_event);
    cudaEventDestroy(kernal_end_event);

    // 错误检查
    cudaError_t kernelErr = cudaGetLastError();
    if (kernelErr != cudaSuccess) {
        std::cerr << "Kernel execution error: " << cudaGetErrorString(kernelErr) << std::endl;
    } else {
        // std::cout << "无打包解压完成!" << std::endl;
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------
// 优化方案1: 批量并行偏移计算

__global__ void calculateOffsetsBatchKernel(const unsigned char* __restrict__ d_cmpBytes,
                                           int* __restrict__ d_offsets,
                                           int* __restrict__ d_numBlocks,
                                           size_t cmpSize,
                                           size_t nbEle,
                                           int batchSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程处理一个潜在的起始位置
    if (tid >= batchSize) return;
    
    size_t startBitPos = tid * 8; // 8位对齐的起始位置
    size_t totalBits = cmpSize * 8;
    size_t minHeaderSize = 192; // 64 + 64 + 8 + 8 + 64 bits
    
    __shared__ int localOffsets[256];
    __shared__ int localCount;
    
    if (threadIdx.x == 0) {
        localCount = 0;
    }
    __syncthreads();
    
    // 验证这个位置是否是有效的块起始
    if (startBitPos + minHeaderSize <= totalBits) {
        size_t bitPos = startBitPos;
        uint64_t bitSize = readBitsDevice(d_cmpBytes, bitPos, 64);
        
        // 简单验证：bitSize应该合理
        if (bitSize >= 64 && bitSize < totalBits && 
            startBitPos + bitSize <= totalBits) {
            
            int localIdx = atomicAdd(&localCount, 1);
            if (localIdx < 256) {
                localOffsets[localIdx] = startBitPos / 8;
            }
        }
    }
    
    __syncthreads();
    
    // 写回全局内存
    if (threadIdx.x == 0 && localCount > 0) {
        int globalStart = atomicAdd(d_numBlocks, localCount);
        for (int i = 0; i < localCount && i < 256; i++) {
            d_offsets[globalStart + i] = localOffsets[i];
        }
    }
}

// 优化方案2: 基于模式识别的快速偏移计算
__global__ void calculateOffsetsPatternKernel(const unsigned char* __restrict__ d_cmpBytes,
                                             int* __restrict__ d_offsets,
                                             int* __restrict__ d_numBlocks,
                                             size_t cmpSize,
                                             size_t nbEle) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    __shared__ int sharedOffsets[128];
    __shared__ int sharedCount;
    
    if (threadIdx.x == 0) {
        sharedCount = 0;
    }
    __syncthreads();
    
    // 每个线程检查多个8字节对齐的位置
    size_t totalBytes = cmpSize;
    size_t minHeaderBytes = 24; // 192 bits = 24 bytes minimum
    
    for (size_t bytePos = tid * 8; bytePos + minHeaderBytes < totalBytes; bytePos += stride * 8) {
        // 快速模式匹配：查找可能的块头部特征
        // 大多数GDF块的bitSize会在合理范围内
        if (bytePos + 8 < totalBytes) {
            uint64_t possibleBitSize = *((uint64_t*)(d_cmpBytes + bytePos));
            
            // 快速验证：bitSize应该在合理范围内
            if (possibleBitSize >= 192 && possibleBitSize < totalBytes * 8) {
                size_t expectedNextBlock = bytePos + (possibleBitSize + 7) / 8;
                
                // 检查这个偏移是否合理
                if (expectedNextBlock < totalBytes) {
                    int localIdx = atomicAdd(&sharedCount, 1);
                    if (localIdx < 128) {
                        sharedOffsets[localIdx] = bytePos;
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // 合并结果到全局内存
    if (threadIdx.x == 0 && sharedCount > 0) {
        int globalStart = atomicAdd(d_numBlocks, sharedCount);
        int maxBlocks = (nbEle + 1023) / 1024;
        
        for (int i = 0; i < min(sharedCount, 128) && globalStart + i < maxBlocks; i++) {
            d_offsets[globalStart + i] = sharedOffsets[i];
        }
    }
}

// 优化方案5: 流水线式偏移计算
__global__ void pipelineOffsetCalculation(const unsigned char* __restrict__ d_cmpBytes,
                                         int* __restrict__ d_offsets,
                                         int* __restrict__ d_numBlocks,
                                         size_t cmpSize,
                                         size_t nbEle,
                                         int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 分阶段处理，每个阶段处理数据的不同部分
    size_t phaseSize = cmpSize / 4;
    size_t startPos = phase * phaseSize;
    size_t endPos = min(startPos + phaseSize, cmpSize);
    
    if (startPos >= endPos) return;
    
    __shared__ int sharedOffsets[64];
    __shared__ int sharedCount;
    
    if (threadIdx.x == 0) {
        sharedCount = 0;
    }
    __syncthreads();
    
    // 每个线程处理这个阶段内的多个位置
    for (size_t pos = startPos + tid * 8; pos + 24 < endPos; pos += blockDim.x * 8) {
        uint64_t possibleBitSize = *((uint64_t*)(d_cmpBytes + pos));
        
        if (possibleBitSize >= 192 && possibleBitSize < cmpSize * 8) {
            int localIdx = atomicAdd(&sharedCount, 1);
            if (localIdx < 64) {
                sharedOffsets[localIdx] = pos;
            }
        }
    }
    
    __syncthreads();
    
    // 写回全局内存
    if (threadIdx.x == 0 && sharedCount > 0) {
        int globalStart = atomicAdd(d_numBlocks, sharedCount);
        for (int i = 0; i < min(sharedCount, 64); i++) {
            d_offsets[globalStart + i] = sharedOffsets[i];
        }
    }
}

// 最终优化的接口函数
void GDFDecompressor::GDFC_decompress_ultra_optimized(double* d_decData,
                                                     unsigned char* d_cmpBytes,
                                                     size_t nbEle,
                                                     size_t cmpSize,
                                                     cudaStream_t stream) {
    if (nbEle == 0 || cmpSize == 0) return;
    
    int maxBlocks = (nbEle + 1023) / 1024;
    if (maxBlocks <= 0) return;
    
    // 分配内存
    int* d_offsets;
    int* d_numBlocks;
    
    cudaMallocAsync(&d_offsets, maxBlocks * sizeof(int), stream);
    cudaMallocAsync(&d_numBlocks, sizeof(int), stream);
    cudaMemsetAsync(d_numBlocks, 0, sizeof(int), stream);
    
    // 选择最适合的偏移计算策略
    if (cmpSize < 1024 * 1024) { // 小文件用单线程
        calculateOffsetsKernel<<<1, 1, 0, stream>>>(
            d_cmpBytes, d_offsets, d_numBlocks, cmpSize, nbEle);
    } else if (cmpSize < 16 * 1024 * 1024) { // 中等文件用模式匹配
        int blocks = min(256, (int)((cmpSize + 8191) / 8192));
        calculateOffsetsPatternKernel<<<blocks, 128, 0, stream>>>(
            d_cmpBytes, d_offsets, d_numBlocks, cmpSize, nbEle);
    } else { // 大文件用流水线处理
        for (int phase = 0; phase < 4; phase++) {
            pipelineOffsetCalculation<<<32, 128, 0, stream>>>(
                d_cmpBytes, d_offsets, d_numBlocks, cmpSize, nbEle, phase);
        }
    }
    
    // 调用解压核函数
    int threadsPerBlock = 128;
    int blocksPerGrid = (maxBlocks + threadsPerBlock - 1) / threadsPerBlock;
    
    decompressKernelOptimized<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_cmpBytes, d_decData, d_offsets, nbEle);
    
    // 清理
    cudaFreeAsync(d_offsets, stream);
    cudaFreeAsync(d_numBlocks, stream);
}