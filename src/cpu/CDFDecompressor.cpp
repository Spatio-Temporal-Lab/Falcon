//
// Created by lz on 24-9-26.
//

#include "CDFDecompressor.h"
#include "output_bit_stream.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

#define bit8  64

bool isLittleEndian() {
    uint16_t value = 0x1;
    return *(reinterpret_cast<uint8_t*>(&value)) == 0x1;
}
uint64_t swapEndian(uint64_t value) {
    return ((value & 0xFF00000000000000) >> 56) |
           ((value & 0x00FF000000000000) >> 40) |
           ((value & 0x0000FF0000000000) >> 24) |
           ((value & 0x000000FF00000000) >> 8)  |
           ((value & 0x00000000FF000000) << 8)  |
           ((value & 0x0000000000FF0000) << 24) |
           ((value & 0x000000000000FF00) << 40) |
           ((value & 0x00000000000000FF) << 56);
}


// Zigzag 解码，将无符号整数还原为带符号整数
long CDFDecompressor::zigzag_decode(unsigned long value) {
    return (value >> 1) ^ -(value & 1);
}

// 使用 Delta 解码恢复整数序列
void CDFDecompressor::deltaDecode(const std::vector<long>& encoded, std::vector<long>& decoded) {
    if (encoded.empty()) return;

    decoded.reserve(encoded.size());
    decoded.push_back(encoded[0]);

    for (size_t i = 1; i < encoded.size(); ++i) {
        long delta = zigzag_decode(encoded[i]);
        decoded.push_back(decoded[i - 1] + delta);
    }
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

void CDFDecompressor::decompressBlock(InputBitStream& bitStream, std::vector<long>& integers, int& totalBitsRead) {
    size_t blockSize = integers.size();
    if (blockSize == 0) return;

    // 读取第一个整数
    long firstValue = bitStream.ReadLong(64);
    if (isLittleEndian()) {
        firstValue = swapEndian(firstValue);
    }
    integers[0] = firstValue;
    totalBitsRead += 64;

    std::cout << "First integer read (hex): " << std::hex << integers[0] << std::endl;

    // 读取其余的Delta编码数据
    for (size_t i = 1; i < blockSize; ++i) {
        long encodedDelta = bitStream.ReadLong(64);
        if (isLittleEndian()) {
            encodedDelta = swapEndian(encodedDelta);
        }
        long delta = zigzag_decode(encodedDelta);
        integers[i] = integers[i - 1] + delta;
        totalBitsRead += 64;
    }
}


// 解压缩数据主函数
void CDFDecompressor::decompress(const std::vector<unsigned char>& input, std::vector<double>& output) {
    if (input.empty()) {
        std::cerr << "Error: Input data is empty." << std::endl;
        return;
    }

    InputBitStream bitStream(input);
    
    int totalBitsRead = 0;

    // 读取最大小数位数
    if (bitStream.isEnd()) {
        std::cerr << "Error: Input stream is too short to contain the decimal places." << std::endl;
        return;
    }

    int maxDecimalPlaces = static_cast<int>(bitStream.Read(8));
    totalBitsRead += 8;
    std::cout << "解压缩：最大小数位数 = " << maxDecimalPlaces << std::endl;
    // 定义块大小
    long total=(input.size()-totalBitsRead)/8;
    std::cout << "总大小 = " << input.size()<< std::endl;
    size_t blockSize = 256;

int i=0;
    // 解压缩数据块
    while (!(bitStream.isEnd() || (totalBitsRead + bit8) > bitStream.data_.size() * 8)) {

        if(total>blockSize)
        {
            total-=blockSize;
        }
        else{
            blockSize=total;
        }
        std::vector<long> integers(blockSize); // 为块大小分配内存

        // 读取块数据
        decompressBlock(bitStream, integers, totalBitsRead);
        
        std::cout << i <<" Cursor: " << bitStream.cursor_ << ", Bits in buffer: " << bitStream.bits_in_buffer_ << std::endl;

        // 将解压后的整数转换为浮点数
        for (long intValue : integers) {
            double value = static_cast<double>(intValue) / std::pow(10, maxDecimalPlaces);
            output.push_back(value);
        }
    }
}


