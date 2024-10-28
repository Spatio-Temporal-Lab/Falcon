//
// Created by lz on 24-9-26.
//

#include "CDFDecompressor.h"
#include "output_bit_stream.h"
#include <cmath>
#include <iostream>

#define bit8 64

// Zigzag 解码，将无符号整数还原为带符号整数
long CDFDecompressor::zigzag_decode(unsigned long value)
{
    return (value >> 1) ^ -(value & 1);
}

void CDFDecompressor::decompressBlock(InputBitStream& bitStream, std::vector<long>& integers, int& totalBitsRead,
                                      size_t blockSize, int& maxDecimalPlaces)
{
    int blocksRead = 0;
    // 读取第一个整数
    long firstValue = bitStream.ReadLong(64);
    blocksRead += 128;
    // std::cout << "First integer read (hex): " << firstValue << std::endl;
    // 读取每个数据的最大位数
    maxDecimalPlaces = static_cast<int>(bitStream.ReadInt(8));
    // std::cout << "解压缩：最大小数位数 = " << maxDecimalPlaces << std::endl;
    blocksRead += 16;
    int bitWight = static_cast<int>(bitStream.ReadInt(8));

    // std::cout << "解压缩：最大数据位数 " << bitWight << std::endl;

    integers.push_back(firstValue);
    // 读取后续的Delta编码数据
    for (size_t i = 1; i < blockSize; i++)
    {
        long encodedDelta = bitStream.ReadLong(bitWight);
        // std::cout << encodedDelta<<" ";
        long delta = zigzag_decode(encodedDelta);
        integers.push_back(integers[i - 1] + delta);

        blocksRead += bitWight;
    }
    // std::cout << blocksRead << std::endl;
}

// 解压缩数据主函数
void CDFDecompressor::decompress(const std::vector<unsigned char>& input, std::vector<double>& output)
{
    if (input.empty())
    {
        std::cerr << "Error: Input data is empty." << std::endl;
        return;
    }

    InputBitStream bitStream;
    bitStream.SetBuffer(input);
    long totalValues = bitStream.ReadLong(64);

    // std::cout << "总大小 " << totalValues << std::endl;
    size_t blockSize = 1024;
    int i = 0;
    // 解压缩数据块
    size_t numBlocks = (totalValues + blockSize - 1) / blockSize;
    for (size_t i = 0; i < numBlocks; ++i)
    {
        size_t currentBlock = std::min(blockSize, totalValues - i * blockSize);
        std::vector<long> integers; // 为块大小分配内存
        int totalBitsRead = bitStream.ReadLong(64);
        // std::cout << "总大小 " << totalBitsRead << std::endl;
        int maxDecimalPlaces = 0;
        // 定义块大小
        decompressBlock(bitStream, integers, totalBitsRead, currentBlock, maxDecimalPlaces);
        // 将解压后的整数转换为浮点数
        if(maxDecimalPlaces>14)
        {
            for (long intValue : integers)
            {
                double d;
                memcpy(&d, &intValue, sizeof(double));
                // std::cout << " " << value;
                output.push_back(d);
            }
        }else
        {
            long po = std::pow(10, maxDecimalPlaces);
            for (long intValue : integers)
            {
                double value = static_cast<double>(intValue) / po;
                output.push_back(value);
            }
        }

        // std::cout << "\n size :" << integers.size() << std::endl;
        bitStream.ReadLong((totalBitsRead + 31) / 32 * 32 - totalBitsRead);
        // std::cout << "kongbai "<<(totalBitsRead+31)/32*32-totalBitsRead;
    }
}
