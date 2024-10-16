//
// Created by lz on 24-9-26.
//

#include "CDFCompressor copy.h"
#include "output_bit_stream.h"
#include <vector>
#include <climits>
#include <cmath>
#include <iostream>
void CDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output) {
    const int blockSize = 1024;

    // 估算缓冲区大小，假设 1 byte ≈ 8 bits, 通过预期的压缩率来计算
    // 假设压缩比是 2:1（压缩后数据大小约为原来的 1/2）
    double expectedCompressionRatio = 2.0;
    size_t estimatedSize = static_cast<size_t>(input.size() * sizeof(double) / expectedCompressionRatio);


    // 确保输出缓冲区的大小足够，避免越界
    OutputBitStream bitStream(estimatedSize);

    int bitSize = 0, totalBitsWritten = 0;

    for (size_t i = 0; i < input.size(); i += blockSize) {
        std::vector<double> block(input.begin() + i, 
                                  input.begin() + std::min(input.size(), i + blockSize));

        int bitSize = 0; // 每个块的位数
        compressBlock(block, bitStream, bitSize); // 压缩块并更新 bitSize
    }

    // 刷新比特流
    bitStream.Flush();

    // 从比特流中获取压缩数据到输出向量
    size_t bufferSize = bitStream.GetBufferSize();
    output.resize(bufferSize); // 设置输出大小
    Array<uint8_t> buffer = bitStream.GetBuffer(bufferSize);

    // 将压缩数据复制到输出
    for (size_t i = 0; i < buffer.length(); ++i) {
        output[i] = static_cast<unsigned char>(buffer[i]); // 确保类型转换
    }
}

void CDFCompressor::compressBlock(const std::vector<double>& block, OutputBitStream& bitStream, int& bitSize) {
    std::vector<int> integers; // 存储整数值
    int minValue, maxValue;

    // 对数据块进行采样，浮点数转化为整数，获取整数表示的最小值和最大值
    int maxDecimalPlaces = sampleBlock(block, integers, minValue, maxValue);
    std::cout<< block[0] <<" : "<< integers[0] << " : ." << maxDecimalPlaces <<std::endl;
    // 将最小值进行Zigzag编码
    unsigned long encodedMin = zigzag_encode(minValue);

    // 写入编码后的最小值和最大的小数点位数到比特流 这是for编码方式
    bitStream.WriteLong(encodedMin, 32); // 假设使用32位表示
    bitStream.Write(maxDecimalPlaces, 8); // 假设使用8位表示最大的小数点后位数

    // 编码每个整数值
    for (int integerValue : integers) {
        unsigned long encodedValue = zigzag_encode(integerValue - minValue);
        bitStream.WriteLong(encodedValue, 32); // 假设使用32位表示
    }

    // 更新比特大小
    bitSize += integers.size() * 32 + 32 + 8; // 每个整数占32位，加上最小值和最大小数位数
    std::cout << "Bit size for current block: " << bitSize << std::endl; // 调试信息
}

int CDFCompressor::sampleBlock(const std::vector<double>& block, std::vector<int>& integers, 
                               int& minValue, int& maxValue) {
    minValue = INT_MAX;  // 初始化最小值为整型最大值
    maxValue = INT_MIN;  // 初始化最大值为整型最小值
    int maxDecimalPlaces = 0; // 初始化小数位数的最大值

    // 遍历数据块中的每个浮点数
    for (double value : block) {
        // 获取当前值的小数位数
        int decimalPlaces = getDecimalPlaces(value);
        maxDecimalPlaces = std::max(maxDecimalPlaces, decimalPlaces); // 更新最大小数位数

        // 将浮点数转换为整数
        int integerValue = static_cast<int>(value * std::pow(10, decimalPlaces));
        integers.push_back(integerValue); // 将转换后的整数添加到整数向量中
        // 更新最小值和最大值
        minValue = std::min(minValue, integerValue);
        maxValue = std::max(maxValue, integerValue);
    }
    return maxDecimalPlaces; // 返回最大的小数位数
}

int CDFCompressor::getDecimalPlaces(double value) {
    int places = 0; // 初始化小数位数计数器
    // 当浮点数与其四舍五入的值不相等时，继续循环
    while (std::fabs(value - std::round(value)) > 1e-9) {
        value *= 10; // 将值乘以 10
        places++; // 增加小数位数计数
    }
    return places; // 返回小数位数
}

unsigned long CDFCompressor::zigzag_encode(long value) {
    // Zigzag 编码，将负数转换为正数以便于压缩
    return (value << 1) ^ (value >> 63);
}

void CDFCompressor::flushBits(std::vector<unsigned char>& output, int& totalBitsWritten) {
    // 调试输出
    std::cout << "Total bits written before resizing: " << totalBitsWritten << std::endl;
    
    // 计算所需的字节数，向上取整
    if (totalBitsWritten < 0) {
        std::cerr << "Error: totalBitsWritten is negative!" << std::endl;
        return; // 早期返回以避免后续错误
    }
    int byteCount = (totalBitsWritten + 7) / 8;  
    output.resize(byteCount); // 调整输出向量的大小以适应写入的位
    //std::cout<<output[0]<<std::endl;

    
}

