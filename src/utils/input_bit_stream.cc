#include "input_bit_stream.h"
#include <iostream>

InputBitStream::InputBitStream(const std::vector<uint8_t>& data) 
    : data_(data), cursor_(0), bits_in_buffer_(0), buffer_(0) {}

uint64_t InputBitStream::Read(int numBits) {
    uint64_t result = 0;

    if (numBits > 64) {
        throw std::out_of_range("Cannot read more than 64 bits at once");
    }

    while (numBits > 0) {
        if (bits_in_buffer_ == 0) {
            FillBuffer();  // 填充缓冲区
            if (bits_in_buffer_ == 0) {
                throw std::runtime_error("Unexpected end of input stream");
            }
        }

        int bits_to_read = std::min(numBits, bits_in_buffer_);
        result <<= bits_to_read;  // 为新位腾出空间
        result |= (buffer_ >> (bits_in_buffer_ - bits_to_read)) & ((1ULL << bits_to_read) - 1);  // 提取位

        bits_in_buffer_ -= bits_to_read;
        numBits -= bits_to_read;
    }

    return result;
}

long InputBitStream::ReadLong(int numBits) {
    std::cout << "ReadLong called with numBits: " << numBits << std::endl;

    // 如果缓冲区为空，则调用 FillBuffer()
    if (bits_in_buffer_ < numBits) {
        std::cout << "Calling FillBuffer..." << std::endl; // 添加调试输出
        FillBuffer();
    }

    // 确保有足够的位可供读取
    if (bits_in_buffer_ < numBits) {
        throw std::runtime_error("Not enough bits to read.");
    }

    // 读取 bits_in_buffer_ 中的位
    long value = (buffer_ >> (bits_in_buffer_ - numBits)) & ((1LL << numBits) - 1);
    buffer_ >>= numBits;  // 移动缓冲区中的位
    bits_in_buffer_ -= numBits; // 减少缓冲区的位数

    return value;
}


bool InputBitStream::isEnd() const {
    return cursor_ >= data_.size() && bits_in_buffer_ == 0;
}

void InputBitStream::FillBuffer() {
    // 打印数据的状态
    std::cout << "Before FillBuffer: Cursor: " << cursor_ << ", Data size: " << data_.size() << std::endl;

    // 重置缓冲区
    buffer_ = 0;  

    // 确保游标在数据范围内
    if (cursor_ >= data_.size()) {
        bits_in_buffer_ = 0; // 数据结束
        return;
    }

    // 从数据中读取一个字节并填充到缓冲区
    for (int i = 0; i < 8 && cursor_ < data_.size(); ++i) {
        buffer_ |= static_cast<uint64_t>(data_[cursor_++]) << (8 * (7 - i)); // 高位优先存储
    }

    bits_in_buffer_ = 8 * std::min(8, static_cast<int>(data_.size() - cursor_));  // 更新缓冲区中的位数

    // 打印当前缓冲区状态
    std::cout << "After FillBuffer: Cursor: " << cursor_ << ", Bits in buffer: " << bits_in_buffer_ << std::endl;
}
