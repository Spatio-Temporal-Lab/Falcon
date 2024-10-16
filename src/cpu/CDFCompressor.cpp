//
// Created by lz on 24-9-26.
//

#include "CDFCompressor.h"

#include <iostream>

#include "output_bit_stream.h"

CDFCompressor::CDFCompressor()
{
    OutputBitStream bitStream(1024 * 8); // 初始化输出位流，假设缓冲区大小为 1024 * 8
}

void CDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output)
{
    OutputBitStream bitStream(1024 * 8); // 初始化输出位流，假设缓冲区大小为 1024 * 8
    int totalBitSize = 0;


    // 对数据进行分块压缩
    for (size_t i = 0; i < input.size(); i += BLOCK_SIZE) {
        int perBitSize = 0;
        size_t currentBlockSize = std::min(BLOCK_SIZE, input.size() - i);
        std::vector<double> block(input.begin() + i, input.begin() + i + currentBlockSize);
        compressBlock(block, bitStream, perBitSize);

        // 将位流中的数据更新到输出缓冲区中
        bitStream.Flush();
        output.insert(output.end(), bitStream.GetBuffer((perBitSize + 7) / 8).begin(),
                      bitStream.GetBuffer((perBitSize + 7) / 8).end());
        bitStream.Refresh();
    }
}

void CDFCompressor::compressBlock(const std::vector<double>& block, OutputBitStream& bitStream, int& bitSize)
{
    // 调用采样方法
    std::vector<long> longs;
    long firstValue;
    int maxDecimalPlaces;
    sampleBlock(block, longs, firstValue, maxDecimalPlaces);


    // 将firstValue和位数写入输出
    bitSize += bitStream.WriteLong(firstValue, 64);
    bitSize += bitStream.WriteInt(maxDecimalPlaces, 8);

    long lastValue = firstValue;
    std::vector<long> deltaList;
    long maxDelta = 0;
    for (int i = 1; i < longs.size(); i++)
    {
        deltaList[i] = zigzag_encode(longs[i] - lastValue);
        maxDelta = std::max(deltaList[i], maxDelta);
    }

    // 计算所有差值，并找出所需的最大 bit 位数
    int bitCount = 0;
    while (maxDelta > 0)
    {
        maxDelta >>= 1;
        bitCount++;
    }

    // 将 bit 位数写入输出
    bitSize += bitStream.WriteInt(bitCount, 8);

    // 按照计算的 bit 位数，将所有差值进行存储
    for (long delta : deltaList)
    {
        bitSize += bitStream.WriteLong(delta, bitCount);
    }
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

    //感觉可以优化计算方法
    firstValue = static_cast<long>(block[0] * maxDecimalPlaces);

    for (double val : block)
    {
        longs.push_back(static_cast<long>(val * maxDecimalPlaces));
    }
}

int CDFCompressor::getDecimalPlaces(double value)
{
    int decimalPlaces = 0;
    while (value != static_cast<int>(value))
    {
        value *= 10;
        decimalPlaces++;
    }
    return decimalPlaces;
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
unsigned long zigzag_encode(long value)
{
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}
