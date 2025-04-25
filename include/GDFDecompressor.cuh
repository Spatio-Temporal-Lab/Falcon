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


// 主机端 辅助类，用于按位读取压缩数据
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


