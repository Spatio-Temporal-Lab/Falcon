//
// Created by lz on 24-9-26.
//

#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "CDFCompressor.h"
#include <cstring>
#include <cstdint>

CDFCompressor::CDFCompressor()
{
    // OutputBitStream bitStream(BLOCK_SIZE * 8); // 初始化输出位流，假设缓冲区大小为 1024 * 8
    POW_NUM = pow(2, 51) + pow(2, 52);
}

void CDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output)
{
    OutputBitStream bitStream(BLOCK_SIZE * 8); // 初始化输出位流，假设缓冲区大小为 1024 * 8
    int totalBitSize = 0;
    bitStream.Write(input.size(),64);
    bitStream.Flush();
    Array<uint8_t> buffer1 = bitStream.GetBuffer(8);

    // 将压缩数据复制到输出
    for (size_t j = 0; j < buffer1.length(); ++j)
    {
        output.push_back(buffer1[j]); // 确保类型转换
    }
    bitStream.Refresh();
    // 对数据进行分块压缩
    for (int i = 0; i < input.size(); i += BLOCK_SIZE)
    {
        int perBitSize = 0;
        size_t currentBlockSize = std::min(BLOCK_SIZE, input.size() - i);
        std::vector<double> block(input.begin() + i, input.begin() + i + currentBlockSize);
        compressBlock(block, bitStream, perBitSize);
        // std::cout << std::endl << "Block size: " << block.size() << std::endl;

        // 将位流中的数据更新到输出缓冲区中
        bitStream.Flush();
        // std::cout << "perBitSize " << perBitSize << std::endl;
        // std::cout << "perBitSize " << (perBitSize + 31) / 32 * 4 << std::endl;
        Array<uint8_t> buffer = bitStream.GetBuffer((perBitSize + 31) / 32 * 4);

        // 将压缩数据复制到输出
        for (size_t j = 0; j < buffer.length(); ++j)
        {
            output.push_back(buffer[j]); // 确保类型转换
        }
        // std::cout << output.size() << std::endl;
        // std::cout << std::endl << "Block size: " << block.size() << std::endl;
        bitStream.Refresh();
    }
}

void CDFCompressor::compressBlock(const std::vector<double>& block, OutputBitStream& bitStream, int& bitSize)
{
    // 调用采样方法
    std::vector<long> longs;
    long firstValue;
    int maxDecimalPlaces = 0;
    sampleBlock(block, longs, firstValue, maxDecimalPlaces);
    // std::cout << std::endl << "sample " << block.size() << std::endl;


    long lastValue = firstValue;
    std::vector<long> deltaList;
    long maxDelta = 0;

    for (int i = 1; i < longs.size(); i++)
    {
        // double num = block[i];
        // double longnum = longs[i];
        long delta = zigzag_encode(longs[i] - lastValue);
        deltaList.push_back(delta);
        maxDelta = std::max(delta, maxDelta);
        lastValue = longs[i];
    }


    // 计算所有差值，并找出所需的最大 bit 位数
    int bitCount = 0;
    // std::cout << " maxDelta " << maxDelta << std::endl;
    while (maxDelta > 0)
    {
        maxDelta >>= 1;
        bitCount++;
    }

    // std::cout << bitCount << std::endl;

    // 将 bit 位数写入输出
    // 将firstValue和位数写入输出
    bitSize = 64 + 64 + 8 + 8 + (block.size() - 1) * bitCount;
    // std::cout << " bitSize " << bitSize << std::endl;
    bitStream.WriteLong(bitSize, 64);
    bitStream.WriteLong(firstValue, 64);
    bitStream.WriteInt(maxDecimalPlaces, 8);
    // std::cout << "maxDecimalPlaces " <<maxDecimalPlaces<< std::endl;
    bitStream.WriteInt(bitCount, 8);
    // 按照计算的 bit 位数，将所有差值进行存储
    for (long delta : deltaList)
    {
        bitStream.WriteLong(delta, bitCount);
    }

    // std::cout << "deltasize "<<deltaList.size() << std::endl;
}

void CDFCompressor::sampleBlock(const std::vector<double>& block, std::vector<long>& longs, long& firstValue,
                                int& maxDecimalPlaces)
{
    // 将所有数转换为整数，并选取最小值和最大值，同时计算最大的小数点后位数

    for (double val : block)
    {
        // 计算当前值的小数点后位数
        int decimalPlaces = getDecimalPlaces(val);
        if (decimalPlaces > maxDecimalPlaces)
        {
            maxDecimalPlaces = decimalPlaces;
        }

    }
    // std::cout << std::dec << "maxDecimalPlaces " << maxDecimalPlaces << std::endl;

    //感觉可以优化计算方法
    firstValue = static_cast<long>(block[0] * std::pow(10, maxDecimalPlaces));

    if(maxDecimalPlaces>15)
    {
        uint64_t bits;
        std::memcpy(&bits, &block[0], sizeof(bits));
        firstValue = static_cast<long>(bits);
        for (double val : block)
        {
            uint64_t uint64;
            std::memcpy(&uint64, &val, sizeof(uint64));
            longs.push_back(static_cast<long>(uint64));
        }
    }else
    {
        firstValue = static_cast<long>(std::round(block[0] * std::pow(10, maxDecimalPlaces)));
        for (double val : block)
        {
            // double num =val * std::pow(10, maxDecimalPlaces);
            longs.push_back(static_cast<long>(std::round (val * std::pow(10, maxDecimalPlaces))));
        }
    }

}

int CDFCompressor::getDecimalPlaces(double value)
{
    double trac = value + POW_NUM - POW_NUM;
    double temp = value;
    int digits = 0;
    int64_t int_temp;
    int64_t trac_temp;
    std::memcpy(&int_temp, &temp, sizeof(double));
    std::memcpy(&trac_temp, &trac, sizeof(double));
    double log10v = log10(value);
    int sp = floor(log10v);
    while (std::abs(trac_temp - int_temp) >1 && digits < 16-sp-1)
    {
        digits++;
        double td = pow(10, digits);
        temp =value*td;
        std::memcpy(&int_temp, &temp, sizeof(double));
        trac = temp + POW_NUM - POW_NUM;
        std::memcpy(&trac_temp, &trac, sizeof(double));

    }
    return digits;
}

// 获取最大值的比特数，用于确定压缩位率
int get_bit_num(long maxQuant)
{
    int bits = 0;
    while (maxQuant)
    {
        maxQuant >>= 1;
        bits++;
    }
    return bits;
}

// Zigzag 编码，将带符号整数转为无符号整数
unsigned long CDFCompressor::zigzag_encode(const long value)
{
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}
