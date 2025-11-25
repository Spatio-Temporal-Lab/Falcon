//
// Created by lz on 24-9-26.
//

#ifndef CDF_COMPRESSOR_H
#define CDF_COMPRESSOR_H

#include <vector>
#include <cstdint>
#include <thread> 
#include "output_bit_stream.h"

class Falcon_basic_compressor {
public:
    Falcon_basic_compressor();
    size_t BLOCK_SIZE = 1025;
    double POW_NUM = ((1L << 51) + (1L << 52));
    // 压缩给定输入数据
    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);
private:
    // 压缩数据块
    void compressBlock(const std::vector<double>& block, OutputBitStream& bitStream, int& totalBitsWritten);

    // 对数据块进行采样，获取最大的小数位数
    void sampleBlock(const std::vector<double>& block, std::vector<long>& longs, long& firstValue,
                                int& maxDecimalPlaces, int& isOk, int& maxBeta);

    // 计算给定值的小数点后位数
    int getDecimalPlaces(double value, int sp);

    // Zigzag 编码，将带符号整数转为无筦号整数
    unsigned long zigzag_encode(long value);
    
    // 与GPU版本一致的double编码函数
    long encodeDoubleWithSignLast(double x);
    
    double pow10_table[17] = {
        1.0,                    // 10^0
        10.0,                   // 10^1
        100.0,                  // 10^2
        1000.0,                 // 10^3
        10000.0,                // 10^4
        100000.0,               // 10^5
        1000000.0,              // 10^6
        10000000.0,             // 10^7
        100000000.0,            // 10^8
        1000000000.0,           // 10^9
        10000000000.0,          // 10^10
        100000000000.0,         // 10^11
        1000000000000.0,        // 10^12
        10000000000000.0,       // 10^13
        100000000000000.0,      // 10^14
        1000000000000000.0,     // 10^15
        10000000000000000.0     // 10^16
    };
};

#endif // CDF_COMPRESSOR_H