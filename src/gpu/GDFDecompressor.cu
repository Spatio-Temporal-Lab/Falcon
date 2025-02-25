// //
// // Created by lz on 24-9-26.
// // cuCompressor/src/gpu/GDFDecompressor.cu
// //

// // GDFDecompressor.cu
// #include "GDFDecompressor.cuh"
// #include <vector>
// #include <cmath>
// #include <cstdint>
// #include <cstring>
// #include <iostream>
// #include <iomanip>


// // 将小端字节数组转换为 uint64_t
// uint64_t bytesToULong(const unsigned char* bytes) {
//     uint64_t val = 0;
//     for(int i = 0; i < 8; i++) {
//         val |= ((uint64_t)bytes[i]) << (i * 8);
//     }
//     return val;
// }

// // 将小端字节数组转换为 int64_t
// int64_t bytesToLong(const unsigned char* bytes) {
//     int64_t val = 0;
//     for(int i = 0; i < 8; i++) {
//         val |= ((int64_t)bytes[i]) << (i * 8);
//     }
//     return val;
// }

// // ZigZag 解码函数
// int64_t zigzag_decode(uint64_t n) {
//     return (n >> 1) ^ -(n & 1);
// }

// // 将位表示转换为双精度浮点数（主机端）
// double bitsToDoubleHost(uint64_t bits) {
//     double d;
//     std::memcpy(&d, &bits, sizeof(d));
//     return d;
// }

// void GDFDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output,int numDatas) {
//     BitReader reader(compressedData);
//     size_t dataSize = compressedData.size() * 8; // 总位数
//     //printf("dataSize:%d\n",dataSize);
//     int blockNumber = 0;

//     while(reader.getBitPos() + 64 + 64 + 8 + 8 +64 <= dataSize) { // 确保有足够的元数据
//         if(reader.getBitPos()%8!=0)
//         {
//             printf("Warning！！！\n");
//             break;
//         }
//         // printf("reader.getBitPos():%d\n",reader.getBitPos());
//         int numData = min(numDatas, 1024);//剩余数据
//         numDatas -= 1024;
//         // 1. 读取 bitSize (64 位)
//         uint64_t bitSize = reader.readBits(64);

//         // 2. 读取 firstValue (64 位)
//         int64_t firstValue = static_cast<int64_t>(reader.readBits(64));

//         // 3. 读取 maxDecimalPlaces (8 位)
//         uint64_t maxDecimalPlacesRaw = reader.readBits(8);
//         unsigned char maxDecimalPlaces = static_cast<unsigned char>(maxDecimalPlacesRaw);

//         // 4. 读取 bitCount (8 位)
//         uint64_t bitCountRaw = reader.readBits(8);
//         unsigned char bitCount = static_cast<unsigned char>(bitCountRaw);
//         // if(bitCount>64)
//         // {
//         //     break;
//         // }

//         // 检查 bitCount 是否为零
//         if(bitCount ==0) {
//             std::cerr << "Error: bitCount is zero, invalid compressed data in block " << blockNumber << "." << std::endl;
//             break;
//         }

//         // 5. 读取 flag1 (64 位)
//         uint64_t flag1 = reader.readBits(64);


//         // 计算 flag2 + data 序列的位数
//         // uint64_t delta_bits = bitSize - 64ULL - 64ULL - 8ULL - 8ULL -64ULL;
//         // if(delta_bits <=0) {
//         //     std::cerr << "Warning: delta_bits calculated as non-positive. Interpreting bitSize as delta_bits in block " << blockNumber << "." << std::endl;
//         //     delta_bits = bitSize;
//         // }
//         // 调试输出
//         // std::cout << "Block " << blockNumber++ << " Metadata:" << std::endl;
//         // printf("bitSize:%d\n",bitSize);
//         // printf("firstValue:%d\n",firstValue);
//         // printf("maxDecimalPlaces:%d\n",maxDecimalPlaces);
//         // printf("bitCount:%d\n",bitCount);
//         // printf("flag1:%016llx\n",flag1);

//         // 6. 得到result数组
//         uint8_t result[64][128];
//         int dataByte = (numData+7)/8;    //data里面有多少byte
//         int flag2Size = (dataByte+7)/8;   //flag2占多少byte
//         uint8_t flag2[64][128];
//         memset(flag2, 0, sizeof(flag2));
//         for(int i=0;i<bitCount;i++)//循环判断每一列，进行复原
//         {
            
//             if((flag1 & (1ULL << i)) != 0)//第i列是稀疏的
//             {
//                 //读取flag2
//                 //printf("read\n");
//                 for(int z=0;z<flag2Size*8;z++)
//                 {
//                     flag2[i][z] = reader.readBits(1);
//                 }
//                 //printf("flag2[%d]:0x%llx\n",i,flag2);
//                 for(int j=0;j<dataByte;j++)
//                 {
//                     if (i >= 64 || j >= 128) {
//                         std::cerr << "Index out of bounds: i=" << i << ", j=" << j << std::endl;
//                         return;
//                         continue;
//                     }
//                     if(flag2[i][j] != 0)//第j个bit不为0
//                     {
//                         uint64_t temp=reader.readBits(8);
//                         result[i][j]=static_cast<unsigned char>(temp);
//                     }
//                     else
//                     {
//                         result[i][j]=0;
//                     }
//                 }
//             }
//             else//非稀疏，直接读
//             {
//                 for(int j=0;j<dataByte;j++)
//                 {
//                     if (i >= 64 || j >= 128) {
//                         std::cerr << "Index out of bounds: i=" << i << ", j=" << j << std::endl;
//                         return;
//                         continue;
//                     }
//                     uint64_t temp=reader.readBits(8);
//                     result[i][j]=static_cast<unsigned char>(temp);
//                 }
//             }
//         }

//         // std::cout << "delta_bits: " << delta_bits << std::endl;
//         // std::cout << "numDeltas: " << numDeltas << std::endl;

//         // 7. 通过result数组还原数据(可能有问题)
//         uint64_t deltasZigzag[numData]; // 存放还原后的数据
//         memset(deltasZigzag, 0, sizeof(deltasZigzag)); // 初始化为 0

//         // 遍历每个 bit（行方向）
//         for (int i = 0; i < bitCount; ++i) {
//             for (int j = 0; j < numData; ++j) {
//                 int byteIndex = j / 8;  // 当前数据的 bit 位属于 result 的哪一字节
//                 int bitIndex = j % 8;   // 当前数据的 bit 位在字节中的偏移量

//                 // 提取 result[i][byteIndex] 的第 bitIndex 位
//                 uint8_t bitValue = (result[i][byteIndex] >> (7 - bitIndex)) & 1;

//                 // 将提取的 bitValue 加到 deltas[j] 的第 i 位
//                 deltasZigzag[j] |= (uint64_t(bitValue) << (bitCount - 1 - i));
//             }
//         }

//         // 8. ZigZag 解码
//         std::vector<int64_t> deltas(numData, 0);
//         for(int i =0; i < numData; i++) {
//             deltas[i] = zigzag_decode(deltasZigzag[i]);
//         }

//         // 7. 重建整数序列
//         std::vector<int64_t> integers;
//         integers.reserve(numData);
//         integers.push_back(firstValue);
//         for(int i = 1; i < numData; i++) {
//             integers.push_back(integers.back() + deltas[i]);
//         }

//         // 8. 转换回双精度浮点数
//         for(auto val : integers) {
//             double d;
//             if(maxDecimalPlaces >15) {
//                 // 直接将整数位转换为 double
//                 uint64_t bits = static_cast<uint64_t>(val);
//                 d = bitsToDoubleHost(bits);
//             }
//             else {
//                 d = static_cast<double>(val) / std::pow(10.0, static_cast<double>(maxDecimalPlaces));
//             }
//             output.push_back(d);
//         }

//         //std::cout << "Block " << (blockNumber-1) << " decompressed successfully." << std::endl;
//     }

//     // 处理剩余位（如果有）
//     if(reader.getBitPos() < dataSize) {
//         //std::cerr << "Warning: " << (dataSize - reader.getBitPos()) << " remaining bits not processed." << std::endl;
//     }
// }
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

    uint64_t bitSize = readBitsDevice(blockData, bitPos, 64);
    int64_t firstValue = static_cast<int64_t>(readBitsDevice(blockData, bitPos, 64));
    uint64_t maxDecimalPlacesRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char maxDecimalPlaces = static_cast<unsigned char>(maxDecimalPlacesRaw);
    uint64_t bitCountRaw = readBitsDevice(blockData, bitPos, 8);
    unsigned char bitCount = static_cast<unsigned char>(bitCountRaw);

    if (bitCount == 0) {
        printf("Error: bitCount is zero in block %d.\n", blockId);
        return;
    }

    uint64_t flag1 = readBitsDevice(blockData, bitPos, 64);

    uint8_t result[64][128];
    int dataByte = (numData + 7) / 8;
    int flag2Size = (dataByte + 7) / 8;
    uint8_t flag2[64][128] = {0};

    for (int i = 0; i < bitCount; i++) {
        if ((flag1 & (1ULL << i)) != 0) {
            for (int z = 0; z < flag2Size * 8; z++) {
                flag2[i][z] = readBitsDevice(blockData, bitPos, 1);
            }
            for (int j = 0; j < dataByte; j++) {
                if (flag2[i][j] != 0) {
                    result[i][j] = static_cast<unsigned char>(readBitsDevice(blockData, bitPos, 8));
                } else {
                    result[i][j] = 0;
                }
            }
        } else {
            for (int j = 0; j < dataByte; j++) {
                result[i][j] = static_cast<unsigned char>(readBitsDevice(blockData, bitPos, 8));
            }
        }
    }

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

    int64_t integers[1024] = {0};
    integers[0] = firstValue;
    for (int i = 1; i < numData; i++) {
        integers[i] = integers[i - 1] + deltas[i];
    }

    for (int i = 0; i < numData; i++) {
        double d;
        if (maxDecimalPlaces > 15) {
            uint64_t bits = static_cast<uint64_t>(integers[i]);
            memcpy(&d, &bits, sizeof(double));
        } else {
            d = static_cast<double>(integers[i]) / pow(10.0, maxDecimalPlaces);
        }
        output[blockId * 1024 + i] = d;
    }
}

// 主机端解压缩函数
void GDFDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output, int numDatas) {
    size_t dataSize = compressedData.size();

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
