//
// Created by lz on 24-9-26.
//

#include <cmath>
#include <iostream>
#include "Falcon_basic_compressor.h"
#include <cstring>
#include <cstdint>
#include <gtest/internal/gtest-internal.h>

Falcon_basic_compressor::Falcon_basic_compressor()
{
    // POW_NUM = pow(2, 52);
}

std::vector<int> static decCounts(64, 0);

void Falcon_basic_compressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output)
{
    OutputBitStream bitStream(BLOCK_SIZE * 8); // 初始化输出位流，假设缓冲区大小为 1024 * 8
    int totalBitSize = 0;
    bitStream.Write(input.size(), 64);
    bitStream.Flush();
    Array<uint8_t> buffer1 = bitStream.GetBuffer(8);

    // 将压缩数据复制到输出，输入的数据位数
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
        // 将位流中的数据更新到输出缓冲区中
        bitStream.Flush();
        Array<uint8_t> buffer = bitStream.GetBuffer((perBlockBitSize + 31) / 32 * 4);

        // 将压缩数据复制到输出
        for (size_t j = 0; j < buffer.length(); ++j)
        {
            output.push_back(buffer[j]); // 确保类型转换
        }
        bitStream.Refresh();
    }
}

void Falcon_basic_compressor::compressBlock(const std::vector<double>& block, OutputBitStream& bitStream, int& bitSize)
{
    // 调用采样方法
    std::vector<long> longs;
    long firstValue;
    int maxDecimalPlaces = 0;
    int maxBeta = 0;
    int isOk;
    sampleBlock(block, longs, firstValue, maxDecimalPlaces, isOk, maxBeta);
    size_t currentBlockSize = block.size();

    //进行delta序列转化（与GPU版本一致）
    std::vector<uint64_t> deltas(currentBlockSize - 1);
    long prevQuant = firstValue;
    uint64_t maxDelta = 0;

    for (int i = 0; i < currentBlockSize - 1; i++)
    {
        long currQuant = longs[i + 1];
        long deltaValue = currQuant - prevQuant;
        deltas[i] = zigzag_encode(deltaValue);
        maxDelta = std::max(maxDelta, deltas[i]);
        prevQuant = currQuant;
    }

    // 计算bitCount（与GPU版本完全一致）
    int bitCount = (maxDelta > 0) ? (64 - __builtin_clzll(maxDelta)) : 1;
    bitCount = std::min(bitCount, 64); // 与GPU中的MAX_BITCOUNT一致

    // 转置处理（与GPU版本一致的算法）
    int numByte = (currentBlockSize - 1 + 7) / 8;
    std::vector<std::vector<uint8_t>> result_matrix(bitCount, std::vector<uint8_t>(numByte, 0));

    // 位转置（与GPU版本完全一致的逻辑）
    for (int i = 0; i < bitCount; ++i) {
        for (int j = 0; j < (int)(currentBlockSize - 1); ++j) {
            int byteIndex = j / 8;
            int bitIndex = j % 8;
            uint8_t bitVal = ((deltas[j] >> (bitCount - 1 - i)) & 1);
            result_matrix[i][byteIndex] |= bitVal << (7 - bitIndex);
        }
    }

    // 稀疏性判断（与GPU版本一致）
    uint64_t flag1 = 0;
    std::vector<std::vector<uint8_t>> flag2_matrix(bitCount);
    
    for (int i = 0; i < bitCount; i++) {
        int flag2Size = (numByte + 7) / 8;
        flag2_matrix[i].resize(flag2Size, 0);
        
        int b0 = 0, b1 = 0;
        for (int j = 0; j < numByte; j++) {
            if (result_matrix[i][j] == 0) {
                b0++;
            } else {
                b1++;
                int flag2_byte_idx = j / 8;
                int flag2_bit_idx = j % 8;
                flag2_matrix[i][flag2_byte_idx] |= (1 << flag2_bit_idx);
            }
        }
        
        // 与GPU版本完全一致的稀疏性判断
        uint64_t is_sparse = (uint64_t)(((numByte + 7) / 8 + b1) < numByte);
        if (is_sparse) {
            flag1 |= (1ULL << i);
        } else {
            flag1 &= ~(1ULL << i); // 确保非稀疏列的标志位为0
        }
    }

    // 计算bitSize（与GPU版本一致）
    bitSize = 64 + 64 + 8 + 8 + 8 + 64; // 基础元数据
    
    for (int i = 0; i < bitCount; i++) {
        if ((flag1 & (1ULL << i)) != 0) { // 稀疏列
            int flag2Size = (numByte + 7) / 8;
            int nonZeroCount = 0;
            for (int j = 0; j < numByte; j++) {
                if (result_matrix[i][j] != 0) {
                    nonZeroCount++;
                }
            }
            bitSize += (flag2Size + nonZeroCount) * 8;
        } else { // 非稀疏列
            bitSize += numByte * 8;
        }
    }

    // 写入数据（与GPU版本格式一致）
    bitStream.WriteLong(bitSize, 64);
    bitStream.WriteLong(firstValue, 64);
    bitStream.WriteInt(maxDecimalPlaces, 8);
    bitStream.WriteInt(maxBeta, 8);
    bitStream.WriteInt(bitCount, 8);
    bitStream.WriteLong(flag1, 64);
    
    // 写入每一列的数据
    for (int i = 0; i < bitCount; i++) {
        if ((flag1 & (1ULL << i)) != 0) { // 稀疏列
            int flag2Size = (numByte + 7) / 8;
            for (int j = 0; j < flag2Size; j++) {
                bitStream.WriteByte(flag2_matrix[i][j]);
            }
            for (int j = 0; j < numByte; j++) {
                if (result_matrix[i][j] != 0) {
                    bitStream.WriteByte(result_matrix[i][j]);
                }
            }
        } else { // 非稀疏列
            for (int j = 0; j < numByte; j++) {
                bitStream.WriteByte(result_matrix[i][j]);
            }
        }
    }
}

void Falcon_basic_compressor::sampleBlock(const std::vector<double>& block, std::vector<long>& longs, long& firstValue,
                                int& maxDecimalPlaces, int& isOk, int& maxBeta)
{
    // 将所有数转换为整数，并选取最小值和最大值，同时计算最大的小数点后位数
    int k = 0;
    int maxSp = -99;  // 与GPU版本一致的初始值
    
    // 第一次遍历：计算maxSp和maxDecimalPlaces（与GPU版本一致）
    for (double val : block)
    {
        //计算起始位置sp
        double log10v = log10(std::abs(val));
        int sp = static_cast<int>(floor(log10v));
        maxSp = std::max(maxSp, sp);
        
        // 计算当前值的小数点后位数
        int decimalPlaces = getDecimalPlaces(val, sp);
        maxDecimalPlaces = std::max(maxDecimalPlaces, decimalPlaces);
        
        k++;
    }
    
    // 与GPU版本一致的maxBeta计算方式
    maxBeta = maxSp + maxDecimalPlaces + 1;


    //感觉可以优化计算方法，伪10进制转化为整数
    // 使用pow10_table而不是std::pow，与GPU版本一致

    // 与GPU版本一致的条件判断: maxBeta > 15 || maxDecimalPlaces > 15
    if (maxBeta > 15 || maxDecimalPlaces > 15)
    {
        isOk = 0;
        firstValue = encodeDoubleWithSignLast(block[0]);
        for (double val : block)
        {
            longs.push_back(encodeDoubleWithSignLast(val));
        }
    }
    else
    {
        isOk = 1;
        double multiplier = (maxDecimalPlaces < 17) ? pow10_table[maxDecimalPlaces] : std::pow(10, maxDecimalPlaces);
        firstValue = static_cast<long>(std::round(block[0] * multiplier));
        for (double val : block)
        {
            longs.push_back(static_cast<long>(std::round(val * multiplier)));
        }
    }
}


int Falcon_basic_compressor::getDecimalPlaces(double value, int sp) //得到小数位数
{
    double trac = value + POW_NUM - POW_NUM;
    double temp = value;

    int digits = 0;
    double td = 1;
    double deltaBound = abs(value) * pow(2, -52);
    // double deltaBound = pow(2,ilogb(temp)-52);
    while (abs(temp - trac) >= deltaBound * td && digits < 16 - sp - 1)
    {
        digits++;
        td = pow10_table[digits];
        temp = value * td;
        // double deltaBound = pow(2,ilogb(temp)-52);
        trac = temp + POW_NUM - POW_NUM;
    }
    if(round(temp)/td!=value)
    {
        digits=23;
    }
    return digits;
}


// Zigzag 编码，将带符号整数转为无符号整数
unsigned long Falcon_basic_compressor::zigzag_encode(const long value)
{
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}

// 与GPU版本一致的double编码函数
long Falcon_basic_compressor::encodeDoubleWithSignLast(double x) {
    union {
        double d;
        long u;
    } val;
    val.d = x;
    return (val.u << 1) ^ (val.u >> (sizeof(long) * 8 - 1));
}