#include "input_bit_stream.h"
#include <iostream>

InputBitStream::InputBitStream(const std::vector<uint8_t>& data) 
    : data_(data), cursor_(0), bits_in_buffer_(0), buffer_(0) {}

uint64_t InputBitStream::Read(int numBits) {
    if (numBits < 1 || numBits > 64) {
        throw std::invalid_argument("numBits must be between 1 and 64");
    }

    while (bits_in_buffer_ < numBits) {
        FillBuffer();
        // 这里检查是否到达输入流的末尾
        if (bits_in_buffer_ < numBits && isEnd()) {
            //throw std::runtime_error("Unexpected end of input stream.");
            break;
        }
    }

    uint64_t result = buffer_ & ((1ULL << numBits) - 1);
    buffer_ >>= numBits;
    bits_in_buffer_ -= numBits;
    return result;
}


long InputBitStream::ReadLong(int numBits) {
    if (numBits < 1 || numBits > 64) {
        throw std::invalid_argument("numBits must be between 1 and 64");
    }

    return static_cast<long>(Read(numBits)); // 直接调用 Read 函数
}

bool InputBitStream::isEnd() const {
    return cursor_ >= data_.size() && bits_in_buffer_ == 0;
}


void InputBitStream::FillBuffer() {
    if (cursor_ >= data_.size()) {
        bits_in_buffer_ = 0; // 没有更多数据可供读取
        return;
    }

    buffer_ = 0;
    bits_in_buffer_ = 0;

    for (int i = 0; i < 8 && cursor_ < data_.size(); ++i) {
        buffer_ |= static_cast<uint64_t>(data_[cursor_++]) << (i * 8);
        bits_in_buffer_ += 8;
    }
}

