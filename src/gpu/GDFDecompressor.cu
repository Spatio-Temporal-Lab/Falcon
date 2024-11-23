// //
// // Created by lz on 24-9-26.
// // cuCompressor/src/gpu/GDFDecompressor.cu
// //

// GDFDecompressor.cu
#include "GDFDecompressor.cuh"
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>


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
int64_t zigzag_decode(uint64_t n) {
    return (n >> 1) ^ -(n & 1);
}

// 将位表示转换为双精度浮点数（主机端）
double bitsToDoubleHost(uint64_t bits) {
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

void GDFDecompressor::decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output) {
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
            deltas[i] = zigzag_decode(deltasZigzag[i]);
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
