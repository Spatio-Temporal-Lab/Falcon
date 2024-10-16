//
// Created by lz on 24-9-26.
//

#ifndef CDFCOMPRESSOR_H
#define CDFCOMPRESSOR_H

#include "output_bit_stream.h"
#include <vector>

class CDFCompressor {
public:
    CDFCompressor()
    {

    }

    // 压缩给定输入数据
    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);

private:
    // 压缩数据块
    void compressBlock(const std::vector<double>& block, OutputBitStream& bitStream , int& bitSize);

    // 对数据块进行采样，获取最小值、最大值和最大的小数点后位数
    int sampleBlock(const std::vector<double>& block, std::vector<int>& integers, int& minValue, int& maxValue);

    // 计算给定值的小数点后位数
    int getDecimalPlaces(double value);

    // 刷新位写入操作
    void flushBits(std::vector<unsigned char>& output, int& totalBitsWritten);

    // Zigzag 编码，将带符号整数转为无符号整数
    unsigned long zigzag_encode(long value);
};

#endif // CDFCOMPRESSOR_H