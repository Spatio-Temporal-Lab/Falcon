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

#include "../../../../../../usr/local/cuda/targets/x86_64-linux/include/driver_types.h"

CDFCompressor::CDFCompressor()
{
    // OutputBitStream bitStream(BLOCK_SIZE * 8); // 初始化输出位流，假设缓冲区大小为 1024 * 8
    POW_NUM = pow(2, 51) + pow(2, 52);
}

void CDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output)
{
    OutputBitStream bitStream(BLOCK_SIZE * 8); // 初始化输出位流，假设缓冲区大小为 1024 * 8
    int totalBitSize = 0;
    bitStream.Write(input.size(), 64);
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
        int perBlockBitSize = 0;
        size_t currentBlockSize = std::min(BLOCK_SIZE, input.size() - i);
        std::vector<double> block(input.begin() + i, input.begin() + i + currentBlockSize);
        compressBlock(block, bitStream, perBlockBitSize);
        // std::cout << std::endl << "Block size: " << block.size() << std::endl;

        // 将位流中的数据更新到输出缓冲区中
        bitStream.Flush();
        // std::cout << "perBitSize " << perBitSize << std::endl;
        // std::cout << "perBitSize " << (perBitSize + 31) / 32 * 4 << std::endl;
        Array<uint8_t> buffer = bitStream.GetBuffer((perBlockBitSize + 31) / 32 * 4);

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
    size_t currentBlockSize = block.size();
    // std::cout << std::endl << "sample " << block.size() << std::endl;

    // std::cout << "blockSize: "<<currentBlockSize << std::endl;

    long lastValue = firstValue;
    std::vector<long> deltaList;
    long maxDelta = 0;
    std::vector<int> bitCounts(64, 0);
    deltaList.push_back(0);

    // std::cout << "0"<<":"<<deltaList[0] << " ";
    for (int i = 1; i < longs.size(); i++)
    {
        // double num = block[i];
        // double longnum = longs[i];
        long delta = zigzag_encode(longs[i] - lastValue);
        deltaList.push_back(delta);
        // std::cout << i<<":"<<deltaList[i] << " ";
        maxDelta = std::max(delta, maxDelta);
        for (int i = 0; i < 64; i++)
        {
            bitCounts[i] += (delta >> i) & 1;
        }
        lastValue = longs[i];
    }

    // std::cout << std::endl;

    // 计算所有差值，并找出所需的最大 bit 位数
    int bitWeight = 0;

    // std::cout<<maxDelta<<std::endl;
    // std::cout << " maxDelta " << maxDelta << std::endl;
    while (maxDelta != 0)
    {
        maxDelta >>= 1;
        bitWeight++;
    }

    // std::cout << bitCount << std::endl;

    // 稀疏列的判断：从 bitWeight 向下寻找稀疏性
    int bestPoint = bitWeight;
    // std::cout<<bitWeight<<std::endl;
    for (int i = bitWeight - 1; i >= 0; --i)
    {
        if (currentBlockSize / 8 + bitCounts[i] * 8 >= currentBlockSize)
        {
            break;
        }
        bestPoint = i;
    }
    // std::cout<<bestPoint<<std::endl;

    // 1. 非稀疏列处理：从 0 到 bestPoint
    int numNonSparseCols = bestPoint;
    int nonSparseColSize = (currentBlockSize + 63) / 64; // 每列的大小（以 uint64_t 为单位）
    int nonSparseSize = numNonSparseCols * nonSparseColSize;
    std::vector<uint64_t> transposedNonSparse(nonSparseSize, 0);
    // std::cout << nonSparseSize << std::endl;
    // std::cout<<bitWeight<<" "<<bestPoint<<std::endl;
    for (int j = 0; j < bestPoint; ++j)
    {
        int baseIndex = j * nonSparseColSize;
        for (int i = 0; i < currentBlockSize; ++i)
        {
            if (deltaList[i] & (1ULL << j))
            {

                transposedNonSparse[baseIndex + i / 64] |= (1ULL << (i % 64));
            }
        }
    }

    // 2. 稀疏列处理：从 bestPoint + 1 到 bitWeight - 1
    int numSparseCols = bitWeight - bestPoint;
    int sparseColSize = (currentBlockSize+7) / 8; // 每列的大小（以字节为单位）
    int sparseSize = numSparseCols * sparseColSize;
    std::vector<uint8_t> sparseTransposed(sparseSize, 0);

    // 计算 flag 数组的位数
    int numFlagBits = sparseSize; // 每个字节对应一个标志位
    int flagArraySize = (numFlagBits + 7) / 8; // 将位数转换为字节数
    std::vector<uint8_t> flag(flagArraySize, 0);

    // 稀疏列转置并标记非零字节
    for (int j = bestPoint; j < bitWeight; ++j)
    {
        int colIndex = j - bestPoint;
        int baseIndex = colIndex * sparseColSize;
        for (int i = 0; i < currentBlockSize; ++i)
        {
            if (deltaList[i] & (1ULL << j))
            {
                sparseTransposed[baseIndex + i / 8] |= (1 << (i % 8));
            }
        }
    }

    int num1Value = 0;

    // 判断稀疏矩阵中的每个字节是否为零，并设置 flag 中对应的位
    for (int idx = 0; idx < sparseSize; ++idx)
    {
        if (sparseTransposed[idx] != 0)
        {
            num1Value++;
            int byteIndex = idx / 8;
            int bitIndex = idx % 8;
            flag[byteIndex] |= (1 << bitIndex); // 设置 flag 中对应的位为 1，表示该字节为非零
        }
    }


    // 将 bit 位数写入输出
    // 将firstValue和位数写入输出
    bitSize = 64 + 64 + 8 + 8 + 8 + flagArraySize * 8 + num1Value * 8 + nonSparseSize * 64;

    // std::cout << "firstValue: " << firstValue << std::endl;
    // std::cout << "BWBP: " << bitWeight << "   " << bestPoint << std::endl;
    // std::cout << "numFlagBits: " << numFlagBits << std::endl;
    // std::cout << "nonSparseSize: " << nonSparseSize << std::endl;
    // std::cout << "flag array size: " << flagArraySize << std::endl;
    // std::cout << " bitSize " << bitSize << std::endl;
    // std::cout << " flagArraySize " << flagArraySize << std::endl;
    // std::cout << " num1Value " << num1Value << std::endl;
    // std::cout << " nonSparseSize " << nonSparseSize << std::endl;
    bitStream.WriteLong(bitSize, 64);
    bitStream.WriteLong(firstValue, 64);
    bitStream.WriteInt(maxDecimalPlaces, 8);
    bitStream.WriteInt(bitWeight, 8);
    bitStream.WriteInt(bestPoint, 8);
    // std::cout << "maxDecimalPlaces " <<maxDecimalPlaces<< std::endl;
    for (int i = 0; i < flagArraySize; i++)
    {
        bitStream.WriteByte(flag[i]);

        // std::cout << static_cast<int>(flag[i]) << " ";
    }
    // std::cout << "BWBP " << bitWeight << " " << bestPoint << std::endl;
    // std::cout << std::endl;
    for (int i = 0; i < sparseTransposed.size(); i++)
    {
        if (sparseTransposed[i] != 0)
        {
            bitStream.WriteByte(sparseTransposed[i]);
            // std::cout << static_cast<int>(sparseTransposed[i]) << " ";
        }
    }
    // std::cout << std::endl;
    for (int i = 0; i < nonSparseSize; i++)
    {
        bitStream.WriteLong(transposedNonSparse[i], 64);

        // std::cout << transposedNonSparse[i] <<"："<<i<< " ";
    }
    // std::cout << std::endl;

    // std::cout << "deltasize "<<deltaList.size() << std::endl;
}

void CDFCompressor::sampleBlock(const std::vector<double>& block, std::vector<long>& longs, long& firstValue,
                                int& maxDecimalPlaces)
{
    // 将所有数转换为整数，并选取最小值和最大值，同时计算最大的小数点后位数
    int k = 0;
    for (double val : block)
    {
        // 计算当前值的小数点后位数
        int decimalPlaces = getDecimalPlaces(val);
        if (decimalPlaces > maxDecimalPlaces)
        {
            maxDecimalPlaces = decimalPlaces;
        }
        k++;
        // std::cout<<k<<" "<<val<<" "<<decimalPlaces<<std::endl;
    }
    // std::cout << std::dec << "maxDecimalPlaces " << maxDecimalPlaces << std::endl;

    //感觉可以优化计算方法
    firstValue = static_cast<long>(block[0] * std::pow(10, maxDecimalPlaces));

    if (maxDecimalPlaces > 15)
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
    }
    else
    {
        firstValue = static_cast<long>(std::round(block[0] * std::pow(10, maxDecimalPlaces)));
        for (double val : block)
        {
            // double num =val * std::pow(10, maxDecimalPlaces);
            longs.push_back(static_cast<long>(std::round(val * std::pow(10, maxDecimalPlaces))));
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
    double log10v = log10(std::abs(value));
    int sp = floor(log10v);
    while (std::abs(trac_temp - int_temp) > 1 && digits < 16 - sp - 1)
    {
        digits++;
        double td = pow(10, digits);
        temp = value * td;
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
