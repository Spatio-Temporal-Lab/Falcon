//
// Created by lz on 24-9-26.
// cuCompressor/include/GDFDeCompressor.cuh
//
#include <vector>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>


#pragma once
#include <vector>
#include <cstdint>

// 解压缩类
class GDFDecompressor {
public:
    void decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output,int numDatas);
};

class GDFDecompressor0 {
public:
    void decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output,int numDatas);
};
// 辅助类，用于按位读取压缩数据
class BitReader {
public:
    BitReader(const std::vector<unsigned char>& buffer) : buffer(buffer), bitPos(0) {}

    // 读取 n 位并作为 uint64_t 返回
    uint64_t readBits(int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = bitPos / 8;
            size_t bitIdx = bitPos % 8;
            if(byteIdx >= buffer.size()) break;
            uint8_t bit = (buffer[byteIdx] >> bitIdx) & 1;
            value |= (static_cast<uint64_t>(bit) << i);
            bitPos++;
        }
        return value;
    }
    uint64_t readBits(int begin,int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = begin / 8;
            size_t bitIdx = begin % 8;
            if(byteIdx >= buffer.size()) break;
            uint8_t bit = (buffer[byteIdx] >> bitIdx) & 1;
            value |= (static_cast<uint64_t>(bit) << i);
            begin++;
        }
        return value;
    }
    // 跳过 n 位
    void advance(int n) {
        bitPos += n;
    }

    // 获取当前位位置
    size_t getBitPos() const {
        return bitPos;
    }

private:
    const std::vector<unsigned char>& buffer;
    size_t bitPos;
};

// //
// // Created by lz on 24-9-26.
// // cuCompressor/include/GDFDecompressor.cuh
// //
// #pragma once
// #include <vector>
// #include <cstdint>

// // 主机端解压缩接口：将压缩数据解压为 double 数组
// class GDFDecompressor {
// public:
//     // 参数说明：
//     //   compressedData：压缩数据字节数组
//     //   output：解压后浮点数结果（各块数据顺序拼接）
//     //   numDatas：待解压的总数据个数
//     void decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output, int numDatas);
// };

// //─────────────────────────────────────────────
// //【设备端辅助结构与函数】

// // 设备端 BitReader，用于按位读取数据（仅在设备内核中使用）
// struct DeviceBitReader {
//     const unsigned char* buffer; // 数据指针
//     size_t size;                 // 数据大小（字节）
//     size_t bitPos;               // 当前比特位置

//     __device__ DeviceBitReader(const unsigned char* buf, size_t s, size_t initBit = 0)
//         : buffer(buf), size(s), bitPos(initBit) {}

//     __device__ uint64_t readBits(int n) {
//         uint64_t value = 0;
//         for (int i = 0; i < n; ++i) {
//             size_t byteIdx = bitPos / 8;
//             size_t bitIdx  = bitPos % 8;
//             if (byteIdx >= size) break;
//             uint8_t bit = (buffer[byteIdx] >> bitIdx) & 1;
//             value |= (uint64_t(bit) << i);
//             bitPos++;
//         }
//         return value;
//     }
    
//     __device__ void advance(int n) {
//         bitPos += n;
//     }
    
//     __device__ size_t getBitPos() const {
//         return bitPos;
//     }
// };

// // 设备端 ZigZag 解码函数
// __device__ inline int64_t zigzag_decode(uint64_t n) {
//     return (n >> 1) ^ -static_cast<int64_t>(n & 1);
// }

// // 设备端辅助函数：将 64 位二进制转换为 double（利用 union 技巧）
// __device__ inline double bitsToDouble(uint64_t bits) {
//     union {
//         uint64_t u;
//         double d;
//     } conv;
//     conv.u = bits;
//     return conv.d;
// }
