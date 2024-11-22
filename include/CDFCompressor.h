//
// Created by lz on 24-9-26.
//

#ifndef CDF_COMPRESSOR_H
#define CDF_COMPRESSOR_H

#include <vector>
#include <cstdint>
#include "output_bit_stream.h"

class CDFCompressor {
public:
    CDFCompressor();

    double POW_NUM;

    size_t BLOCK_SIZE = 1024;

    // 压缩给定输入数据
    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);

    int get_max(){return bitWight;}
private:
    // 压缩数据块

    void compressBlock(const std::vector<double>& block, OutputBitStream& bitStream, int& totalBitsWritten);

    // 对数据块进行采样，获取最大的小数位数
    void sampleBlock(const std::vector<double>& block, std::vector<long>& longs, long& firstValue,
                                int& maxDecimalPlaces);

    // 计算给定值的小数点后位数
    int getDecimalPlaces(double value);

    // 刷新位写入操作
    void flushBits(std::vector<unsigned char>& output, OutputBitStream& bitStream, int& totalBitsWritten);

    // Zigzag 编码，将带符号整数转为无符号整数
    unsigned long zigzag_encode(long value);

    int bitWight;
};

#endif // CDF_COMPRESSOR_H
