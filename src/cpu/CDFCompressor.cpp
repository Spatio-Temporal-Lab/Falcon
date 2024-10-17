//
// Created by lz on 24-9-26.
//

#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include "CDFCompressor.h"

// Zigzag 编码，将带符号整数转为无符号整数
unsigned long CDFCompressor::zigzag_encode(long value)
{
    return (value << 1) ^ (value >> 63);
}

// 刷新位写入操作
void CDFCompressor::flushBits(std::vector<unsigned char> &output, OutputBitStream &bitStream, int &totalBitsWritten)
{
    bitStream.Flush();
    size_t bufferSize = bitStream.GetBufferSize();
    output.resize(bufferSize);
    Array<uint8_t> buffer = bitStream.GetBuffer(bufferSize);

    for (size_t i = 0; i < buffer.length(); ++i)
    {
        output[i] = static_cast<unsigned char>(buffer[i]);
    }
}

// 计算给定值的小数点后位数
//  int CDFCompressor::getDecimalPlaces(double value) {
//      std::ostringstream oss;
//      oss << std::fixed << std::setprecision(16) << value;  // 限制高精度小数

//     std::string str = oss.str();
//     size_t dotPosition = str.find('.');
//     if (dotPosition == std::string::npos) return 0;

//     // 去掉末尾的无效 0
//     str.erase(str.find_last_not_of('0') + 1, std::string::npos);
//     if (str.back() == '.') str.pop_back();  // 如果最后是小数点，删除它

//     return str.length() - dotPosition - 1;  // 返回有效的小数位数
// }

int CDFCompressor::getDecimalPlaces(double value)
{
    int decimalPlaces = 0;
    while (value != static_cast<long>(value) && decimalPlaces < 64)
    {
        value *= 10;
        decimalPlaces++;
    }
    return decimalPlaces;
}

// 压缩数据块
void CDFCompressor::compressBlock(const std::vector<long> &block, OutputBitStream &bitStream, int &totalBitsWritten)
{
    if (block.empty())
        return;

    // // 处理第一个元素
    // bitStream.Write(static_cast<uint64_t>(block[0]), 64);
    // std::cout<<"第一个元素："<<block[0]<<std::endl;
    // totalBitsWritten += 64;

    // for (size_t i = 1; i < block.size(); ++i) {
    //     long delta = block[i] - block[i - 1];
    //     long encodedDelta = zigzag_encode(delta);
    //     bitStream.WriteLong(encodedDelta, 64);//64位
    //     totalBitsWritten += 64;
    // }

    std::vector<long> deltaList;
    deltaList.push_back(zigzag_encode(block[0]));
    std::cout << "第一个元素：" << block[0] << std::endl;

    long maxDelta = block[0];
    for (size_t i = 1; i < block.size(); ++i)
    {
        long delta = block[i] - block[i - 1];
        long encodedDelta = zigzag_encode(delta);

        deltaList.push_back(encodedDelta);

        maxDelta=std::max(encodedDelta, maxDelta);
    }

    bitWight = std::ceil(log2(maxDelta));

    totalBitsWritten += bitStream.Write(bitWight, 8);
    
    for (long delta : deltaList)
    {
        totalBitsWritten += bitStream.Write(delta, bitWight);
    }
}

void CDFCompressor::sampleBlock(const std::vector<double> &block, std::vector<long> &integers, int &maxDecimalPlaces)
{
    integers.reserve(block.size());

    for (double value : block)
    {
        long intValue = static_cast<long>(value * std::pow(10, maxDecimalPlaces)); // 将浮点数转换为长整型
        integers.push_back(intValue);
    }
}

void CDFCompressor::compress(const std::vector<double> &input, std::vector<unsigned char> &output)
{
    if (input.empty())
        return;

    const int blockSize = 1024;
    OutputBitStream bitStream(input.size() * sizeof(double)); // 估算缓冲区大小
    int totalBitsWritten = 0;

    // 第二次遍历进行压缩
    for (size_t i = 0; i < input.size(); i += blockSize)
    {
        std::vector<double> block(input.begin() + i, input.begin() + std::min(input.size(), i + blockSize));
        std::vector<long> integers;

        // 第一次遍历获取全局最大小数位数
        int overallMaxDecimalPlaces = 0;
        for (double value : input)
        {

            overallMaxDecimalPlaces = std::max(overallMaxDecimalPlaces, getDecimalPlaces(value));
        }

        std::cout << overallMaxDecimalPlaces << ":最大小数位数\n";
        // 将最大小数位数写入输出数据的前几个字节
        bitStream.Write(static_cast<uint64_t>(overallMaxDecimalPlaces), 8); // 假设使用8位存储最大小数位数
        totalBitsWritten += 8;

        sampleBlock(block, integers, overallMaxDecimalPlaces); // 进行采样并获取整数值
        compressBlock(integers, bitStream, totalBitsWritten);  // 压缩块数据
    }

    // 刷新位写入操作
    flushBits(output, bitStream, totalBitsWritten);
}
