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
    void GDFC_decompress(double* d_decData, unsigned char* d_cmpBytes, size_t nbEle, size_t cmpSize, cudaStream_t stream);
    void  GDFC_decompress_stream_optimized(
            double* d_decData,          // 解压输出缓冲
            unsigned char* d_cmpBytes,  // 压缩输入缓冲（设备端）
            size_t nbEle,               // 原始元素数量
            size_t cmpSize,             // 压缩数据大小
            cudaStream_t stream) ;
    void decompress(const std::vector<unsigned char>& compressedData, std::vector<double>& output,int numDatas);
    void decompress_nopack(const std::vector<unsigned char>& compressedData, std::vector<double>& output) ;
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

class BitReader0 {
public:
    // 原有的vector构造函数 - 保持向后兼容
    BitReader0(const std::vector<unsigned char>& buffer) : 
        bufferPtr(buffer.data()), bufferSize(buffer.size()), bitPos(0), ownsBuffer(false) {}
    
    // 新增的指针构造函数 - 支持直接使用指针
    BitReader0(const unsigned char* buffer, size_t size) : 
        bufferPtr(buffer), bufferSize(size), bitPos(0), ownsBuffer(false) {}

    // 读取 n 位并作为 uint64_t 返回
    uint64_t readBits(int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = bitPos / 8;
            size_t bitIdx = bitPos % 8;
            if(byteIdx >= bufferSize) break;
            uint8_t bit = (bufferPtr[byteIdx] >> bitIdx) & 1;
            value |= (static_cast<uint64_t>(bit) << i);
            bitPos++;
        }
        return value;
    }
    
    // 从指定位置读取 n 位
    uint64_t readBits(int begin, int n) {
        uint64_t value = 0;
        for(int i = 0; i < n; ++i) {
            size_t byteIdx = begin / 8;
            size_t bitIdx = begin % 8;
            if(byteIdx >= bufferSize) break;
            uint8_t bit = (bufferPtr[byteIdx] >> bitIdx) & 1;
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
    
    // 获取缓冲区大小
    size_t getBufferSize() const {
        return bufferSize;
    }

private:
    const unsigned char* bufferPtr;  // 统一使用指针
    size_t bufferSize;               // 缓冲区大小
    size_t bitPos;                   // 当前位位置
    bool ownsBuffer;                 // 是否拥有缓冲区（用于将来可能的内存管理）
};

