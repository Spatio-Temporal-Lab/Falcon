//
// Created by lz on 24-9-26.
//

#include "Falcon_basic_decompressor.h"
#include "output_bit_stream.h"
#include <cmath>
#include <iostream>

#define bit8 64
#define BLOCK_SIZE 1024

// Zigzag 解码，将无符号整数还原为带符号整数
long Falcon_basic_decompressor::zigzag_decode(unsigned long value)
{
    return (value >> 1) ^ -(value & 1);
}

// 与GPU版本一致的double解码函数
double Falcon_basic_decompressor::decodeDoubleWithSignLast(uint64_t value) {
    uint64_t original = (value >> 1) ^ -((int64_t)(value & 1));
    union {
        uint64_t u;
        double d;
    } val;
    val.u = original;
    return val.d;
}

void Falcon_basic_decompressor::decompressBlock(InputBitStream& bitStream, std::vector<long>& originalData, int& totalBitsRead,
                                      size_t blockSize, int& maxDecimalPlaces, int& isOk)
{
    // 读取头部信息（与GPU版本格式一致）
    uint64_t firstValue = bitStream.ReadLong(64);
    maxDecimalPlaces = bitStream.ReadInt(8);
    int maxBeta = bitStream.ReadInt(8);
    int bitCount = bitStream.ReadInt(8);
    uint64_t flag1 = bitStream.ReadLong(64);
    
    // 判断使用哪种解码方式
    isOk = (maxBeta <= 15 && maxDecimalPlaces <= 15) ? 1 : 0;

    if (bitCount == 0 || bitCount > 64) {
        std::cerr << "无效的bitCount: " << bitCount << std::endl;
        return;
    }

    int numByte = (blockSize - 1 + 7) / 8;
    int flag2Size = (numByte + 7) / 8;
    
    // 重建位矩阵
    std::vector<std::vector<uint8_t>> result_matrix(bitCount, std::vector<uint8_t>(numByte, 0));
    
    // 读取每一列的数据
    for (int i = 0; i < bitCount; i++) {
        if ((flag1 & (1ULL << i)) != 0) { // 稀疏列
            // 读取flag2
            std::vector<uint8_t> flag2(flag2Size);
            for (int j = 0; j < flag2Size; j++) {
                flag2[j] = bitStream.ReadByte(8);
            }
            
            // 读取非零字节
            for (int j = 0; j < numByte; j++) {
                int flag2_byte_idx = j / 8;
                int flag2_bit_idx = j % 8;
                if (flag2[flag2_byte_idx] & (1 << flag2_bit_idx)) {
                    result_matrix[i][j] = bitStream.ReadByte(8);
                } else {
                    result_matrix[i][j] = 0;
                }
            }
        } else { // 非稀疏列
            for (int j = 0; j < numByte; j++) {
                result_matrix[i][j] = bitStream.ReadByte(8);
            }
        }
    }
    
    // 重建delta序列
    std::vector<uint64_t> deltas(blockSize - 1, 0);
    
    for (int j = 0; j < blockSize - 1; j++) {
        for (int i = 0; i < bitCount; i++) {
            int byteIndex = j / 8;
            int bitIndex = j % 8;
            uint8_t bitVal = (result_matrix[i][byteIndex] >> (7 - bitIndex)) & 1;
            deltas[j] |= ((uint64_t)bitVal << (bitCount - 1 - i));
        }
    }
    
    // 恢复原始数据
    originalData.clear();
    originalData.push_back(firstValue);
    
    long prevValue = firstValue;
    for (int i = 0; i < blockSize - 1; i++) {
        long deltaDecoded = zigzag_decode(deltas[i]);
        long currentValue = prevValue + deltaDecoded;
        originalData.push_back(currentValue);
        prevValue = currentValue;
    }
}

// 解压缩数据主函数
void Falcon_basic_decompressor::decompress(const std::vector<unsigned char>& input, std::vector<double>& output)
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
    size_t blockSize = 1025;  // 修改为与压缩器一致的1025
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
        int isOk;
        // 定义块大小
        // std::cout << "numBlocks "<<i << std::endl;
        // std::cout << currentBlock << std::endl;

        // std::cout<<"totalbitsize: "<<totalBitsRead<<std::endl;
        decompressBlock(bitStream, integers, totalBitsRead, currentBlock, maxDecimalPlaces, isOk);
        // 将解压后的整数转换为浮点数
        if(isOk==0)
        {
            for (long intValue : integers)
            {
                // intValue已经是正确的位表示，不需要再进行ZigZag解码
                double d = decodeDoubleWithSignLast(static_cast<uint64_t>(intValue));
                output.push_back(d);
            }
        }else
        {
            // 与GPU版本一致的除数计算
            double divisor = (maxDecimalPlaces < 16) ? pow(10.0, maxDecimalPlaces) : pow(10.0, maxDecimalPlaces);
            for (long intValue : integers)
            {
                double value = static_cast<double>(intValue) / divisor;
                output.push_back(value);
            }
        }

        // std::cout << "\n size :" << integers.size() << std::endl;
        // std::cout<<"error: "<<(totalBitsRead + 31) / 32 * 32 - totalBitsRead<<std::endl;
        bitStream.ReadLong((totalBitsRead + 31) / 32 * 32 - totalBitsRead);
        // std::cout << "kongbai "<<(totalBitsRead+31)/32*32-totalBitsRead;
    }
}