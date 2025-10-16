//
// Created by lz on 24-9-26.
//

#include "CDFDecompressor.h"
#include "output_bit_stream.h"
#include <cmath>
#include <iostream>

#define bit8 64
#define BLOCK_SIZE 1024

// Zigzag 解码，将无符号整数还原为带符号整数
long CDFDecompressor::zigzag_decode(unsigned long value)
{
    return (value >> 1) ^ -(value & 1);
}

void CDFDecompressor::decompressBlock(InputBitStream& bitStream, std::vector<long>& originalData, int& totalBitsRead,
                                      size_t blockSize, int& maxDecimalPlaces,int& isOk)
{
    // std::cout<<"BLOCK START"<<std::endl;
    int blocksRead = 0;

    uint64_t firstValue = bitStream.ReadLong(64);
    isOk = bitStream.ReadInt(8);
    maxDecimalPlaces = bitStream.ReadInt(8);
    uint32_t bitWeight = bitStream.ReadInt(8);
    uint32_t bestPoint = bitStream.ReadInt(8);
    blocksRead += 128;
    blocksRead += 32;
    // 2. 计算尺寸
    int numNonSparseCols = bestPoint;
    int nonSparseColSize = (blockSize + 63) / 64; // 每列的大小（以 uint64_t 为单位）
    int nonSparseSize = numNonSparseCols * nonSparseColSize;

    int numSparseCols = bitWeight - bestPoint;
    int sparseColSize = (blockSize+7) / 8; // 每列的大小（以字节为单位）
    // std::cout << "blockSize: "<<blockSize << std::endl;
    int sparseSize = numSparseCols * sparseColSize;

    int numFlagBits = sparseSize; // 每个字节对应一个标志位
    int flagArraySize = (numFlagBits + 7) / 8; // 将位数转换为字节数
    // 3. 读取 flag 数组

    // std::cout << "firstValue: "<<firstValue<<std::endl;
    // std::cout << "BWBP: "<<bitWeight<<"   "<<bestPoint<<std::endl;
    // std::cout << "numFlagBits: "<<numFlagBits << std::endl;
    //
    // std::cout << "nonSparseSize: "<<nonSparseSize << std::endl;
    // std::cout<<"flag array size: "<<flagArraySize<<std::endl;
    std::vector<uint8_t> flag(flagArraySize);
    for (int i = 0; i < flagArraySize; ++i)
    {
        flag[i] = bitStream.ReadByte(8);
        // std::cout << static_cast<int>(flag[i]) << " ";
        blocksRead += 8;

    }
    // std::cout << std::endl;

    // 4. 重建稀疏矩阵 sparseTransposed
    std::vector<uint8_t> sparseTransposed(sparseSize, 0);
    // std::cout << "sparseTransposed:" <<sparseSize<< std::endl;
    for (int idx = 0; idx < sparseSize; ++idx)
    {
        int byteIndex = idx / 8;
        int bitIndex = idx % 8;
        if (flag[byteIndex] & (1 << bitIndex))
        {
            // std::cout << "1";
            // 该字节在编码时是非零的，需要从 bitStream 中读取
            sparseTransposed[idx] = bitStream.ReadByte(8);
            // std::cout << static_cast<int>(sparseTransposed[idx]) << " ";
            blocksRead += 8;
        }
        else
        {
            // std::cout << "0";
            // 该字节在编码时为零，无需读取，保持为 0
            sparseTransposed[idx] = 0;
        }
    }

    // 5. 读取非稀疏矩阵 transposedNonSparse
    std::vector<uint64_t> transposedNonSparse(nonSparseSize);
    // std::cout << nonSparseSize << std::endl;
    for (int i = 0; i < nonSparseSize; ++i)
    {
        transposedNonSparse[i] = bitStream.ReadLong(64);
        // std::cout << transposedNonSparse[i] <<"："<<i<< " ";
        blocksRead += 64;
    }


    std::vector<long> deltaList;

    // std::cout  << std::endl;
    // 6. 重建原始的 deltaList
    deltaList.resize(blockSize);
    for (int i = 0; i < blockSize; ++i)
    {
        deltaList[i] = 0; // 初始化
        // 处理非稀疏列
        for (int j = 0; j < bestPoint; ++j)
        {
            int baseIndex = j * nonSparseColSize;
            if (transposedNonSparse[baseIndex + i / 64] & (1ULL << (i % 64)))
            {
                deltaList[i] |= (1ULL << j);

            }
        }

        // 处理稀疏列
        for (int j = bestPoint; j < bitWeight; ++j)
        {
            int colIndex = j - bestPoint;
            int baseIndex = colIndex * sparseColSize;
            int idx = baseIndex + i / 8;
            if (sparseTransposed[idx] & (1 << (i % 8)))
            {
                deltaList[i] |= (1ULL << j);
            }
        }
        // std::cout <<i<<":"<< deltaList[i] << " ";
    }

    // 7. 如果需要，处理 firstValue 和 maxDecimalPlaces
    // 例如，恢复原始数据列表（假设 deltaList 存储的是差分值）
    originalData.push_back(firstValue);
    for (int i = 1; i < blockSize; ++i)
    {
        originalData.push_back(originalData[i - 1] + zigzag_decode(deltaList[i]));
    }
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
                // std::cout << std::dec<<value << std::endl;
                output.push_back(value);
            }
        }

        // std::cout << "\n size :" << integers.size() << std::endl;
        // std::cout<<"error: "<<(totalBitsRead + 31) / 32 * 32 - totalBitsRead<<std::endl;
        bitStream.ReadLong((totalBitsRead + 31) / 32 * 32 - totalBitsRead);
        // std::cout << "kongbai "<<(totalBitsRead+31)/32*32-totalBitsRead;
    }
}
