//
// Created by lz on 24-9-26.
//

#ifndef CDF_COMPRESSOR_H
#define CDF_COMPRESSOR_H

#include <vector>
#include <cstddef> // for size_t

// 常量定义
const int DEFAULT_BLOCK_SIZE = 32;      // 每个压缩块的元素数
const int BYTE_SIZE = 8;                // 每字节的位数
const int MAX_BIT_RATE = 64;            // 最大位宽，支持long类型数据的位宽

// 获取最大值的比特数，用于确定压缩位率
int get_bit_num(long maxQuant);

// Zigzag 解码，将无符号整数还原为带符号整数
long zigzag_decode(unsigned long value);


// 解压缩函数：将压缩数据解码还原为原始数据
void CDFDecompressor(const std::vector<unsigned char>& cmpData,
                     const std::vector<unsigned int>& cmpOffset,
                     std::vector<long>& decompressedData,
                     int bit_rate, size_t nbEle);

#endif // CDF_COMPRESSOR_H