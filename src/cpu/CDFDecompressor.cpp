//
// Created by lz on 24-9-26.
//

#include "CDFDecompressor.h"
#include "output_bit_stream.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

#define bit8 64

// Zigzag 解码，将无符号整数还原为带符号整数
long CDFDecompressor::zigzag_decode(unsigned long value)
{
    return (value >> 1) ^ -(value & 1);
}

// 解压缩一个数据块
// void CDFDecompressor::decompressBlock(InputBitStream& bitStream, std::vector<long>& integers, int& totalBitsRead) {
//     size_t blockSize = integers.size();
//     std::cout << "in\n";
//     if (blockSize == 0) return;

//     integers.resize(blockSize);

//     // 检查 bits_in_buffer_
//     std::cout << "Before reading first integer: bits_in_buffer_ = " << bitStream.bits_in_buffer_ << std::endl;

//     // 读取第一个整数
//     if (bitStream.isEnd() || (totalBitsRead + bit8) > bitStream.data_.size() * 8) {
//         return ;
//         throw std::runtime_error("Unexpected end of input stream before reading first integer.");
//     }
//     integers[0] = bitStream.ReadLong(bit8);
//     if (isLittleEndian()) {
//         integers[0] = swapEndian(integers[0]);
//     }

//     totalBitsRead += bit8;

//     // 检查读取的值
//     std::cout << "First integer read: " << integers[0] << std::endl;

//     // 读取后续 Delta 编码值
//     for (size_t i = 1; i < blockSize; ++i) {
//         if (bitStream.isEnd() || totalBitsRead + 64 > bitStream.data_.size() * 8) {
//             throw std::runtime_error("Unexpected end of input stream.");
//         }
//         //std::cout << i <<" Cursor: " << bitStream.cursor_ << ", Bits in buffer: " << bitStream.data_.size() << std::endl;

//         long encodedDelta = bitStream.ReadLong(bit8);
//         integers[i] = encodedDelta;
//         totalBitsRead += bit8;

//         // long encodedDelta = bitStream.ReadLong(64);  // 读取 64 位
//         // long delta = zigzag_decode(encodedDelta);    // 解码 ZigZag 值
//         // integers[i] = delta;

//         // 检查读取的值
//         //std::cout << "Delta value read: " << encodedDelta << std::endl;
//     }

//     // 使用 Delta 解码恢复整数序列
//     std::vector<long> decoded;
//     deltaDecode(integers, decoded);
//     integers = std::move(decoded);
// }
//

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
        // if (bitStream.isEnd() || (bitStream.cursor_ * 8) + bitStream.bits_in_buffer_ < bitWight)
        // {
        //     throw std::runtime_error("Unexpected end of input stream.");
        // }
        long encodedDelta = bitStream.ReadLong(bitWight);
        // std::cout << encodedDelta<<" ";
        long delta = zigzag_decode(encodedDelta);
        integers.push_back(integers[i - 1] + delta);

        blocksRead += bitWight;
        //std::cout<< integers[i] << " int : " << i <<std::endl;
        // // 检查是否有足够的位可以读取
        // if ((bitStream.data_.size() - bitStream.cursor_) * 8 + bitStream.bits_in_buffer_ < bitWight)
        // {
        //     std::cout<< (bitStream.data_.size() - bitStream.cursor_) * 8 + bitStream.bits_in_buffer_ << " < " << bitWight <<std::endl;
        //     //throw std::runtime_error("Not enough bits to read.");
        //     bitStream.bits_in_buffer_ = 0;
        //     return;
        // }

        // if (bitStream.bits_in_buffer_ + bitStream.data_.size() - bitStream.cursor_ < bitWight)
        // {
        //     bitStream.cursor_ = bitStream.bits_in_buffer_ + bitStream.data_.size();
        //     bitStream.bits_in_buffer_ = 0;
        //     return;
        // }
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
        // if (bitStream.bits_in_buffer_ + bitStream.data_.size() - bitStream.cursor_ < 8)
        // {
        //     std::cerr << "Error: Not enough bits to read." << std::endl;
        //     break; // 退出循环
        // }
        // if (total > blockSize)
        // {
        //     total -= blockSize;
        // }
        // else
        // {
        //     blockSize = total;
        // }
        size_t currentBlock = std::min(blockSize, totalValues - i * blockSize);
        std::vector<long> integers; // 为块大小分配内存
        int totalBitsRead = bitStream.ReadLong(64);
        // std::cout << "总大小 " << totalBitsRead << std::endl;
        int maxDecimalPlaces = 0;
        // 定义块大小
        decompressBlock(bitStream, integers, totalBitsRead, currentBlock, maxDecimalPlaces);
        // 将解压后的整数转换为浮点数
        long po = std::pow(10, maxDecimalPlaces);
        for (long intValue : integers)
        {
            double value = static_cast<double>(intValue) / po;
            // std::cout << " " << value;
            output.push_back(value);
        }
        // std::cout << "\n size :" << integers.size() << std::endl;
        bitStream.ReadLong((totalBitsRead + 31) / 32 * 32 - totalBitsRead);
        // std::cout << "kongbai "<<(totalBitsRead+31)/32*32-totalBitsRead;
    }
}
