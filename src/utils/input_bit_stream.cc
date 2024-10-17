#include "input_bit_stream.h"
#include <iostream>

InputBitStream::InputBitStream(const std::vector<uint8_t> &data)
    : data_(data), cursor_(0), bits_in_buffer_(0), buffer_(0) {}

//@
// 长度
uint64_t InputBitStream::Read(int numBits)
{
    if (numBits < 1 || numBits > 64)
    {
        throw std::invalid_argument("numBits must be between 1 and 64");
    }

    // 确保缓冲区中有足够的数据
    while (bits_in_buffer_ < numBits)
    {
        FillBuffer();
        if (bits_in_buffer_ < numBits && isEnd())
        {
            throw std::runtime_error("READ Unexpected end of input stream.");
        }
    }

    /*小端序*/
    // //使用掩码正确提取 numBits 位的数据
    // uint64_t mask = (numBits == 64) ? ~0ULL : ((1ULL << numBits) - 1);
    // uint64_t result = buffer_ & mask;  // 提取所需位

    // // 打印调试信息
    // std::cout << "Buffer: " << std::hex << buffer_ << ", Mask: " << mask
    //           << ", Result: " << result << ", Bits in Buffer: " << bits_in_buffer_ << std::endl;

    // //更新缓冲区
    // buffer_ >>= numBits;//莫名其妙在numBits=64时会执行失败，可能是没有这么多位，得/8？
    //                         //在后面用其他方式实现了
    // bits_in_buffer_ -= numBits;

    /*大端序*/
    // **读取最高的 numBits 位**（注意我们采用大端序拼接）
    uint64_t shift_amount = bits_in_buffer_ - numBits;
    uint64_t result = (buffer_ >> shift_amount) & ((1ULL << numBits) - 1);

    // 打印调试信息
    // std::cout << "Buffer: " << std::hex << buffer_
    //           << ", Shift Amount: " << shift_amount
    //           << ", Result: " << result
    //           << ", Cursor: " << cursor_
    //           << ", data: " << data_.size()
    //           << ", Bits in Buffer: " << bits_in_buffer_ << std::endl;

    // 更新缓冲区
    bits_in_buffer_ -= numBits;

    return result;
}

long InputBitStream::ReadLong(int numBits)
{
    if (numBits < 1 || numBits > 64)
    {
        throw std::invalid_argument("numBits must be between 1 and 64");
    }

    return static_cast<long>(Read(numBits)); // 直接调用 Read 函数
}

bool InputBitStream::isEnd() const
{
    return cursor_ >= data_.size() && bits_in_buffer_ <= 0;
}

void InputBitStream::FillBuffer()
{
    if (cursor_ >= data_.size())
    {
        return; // 没有更多数据
    }

    if (bits_in_buffer_ <= 0)
    {
        bits_in_buffer_ = 0;
        buffer_ = 0;
    }
    // //尽可能填充新数据，最多填满 64 位（小端序）
    //     while (bits_in_buffer_ <= 56 && cursor_ < data_.size()) {
    //     uint64_t new_byte = static_cast<uint64_t>(data_[cursor_++]);
    //     buffer_ |= (new_byte << bits_in_buffer_);  // 拼接到低位
    //     bits_in_buffer_ += 8;
    // }

    // **大端序**：高字节优先填充（确保拼接顺序正确）
    while (bits_in_buffer_ <= 56 && cursor_ < data_.size())
    {
        buffer_ = (buffer_ << 8) | data_[cursor_++]; // 高字节优先
        bits_in_buffer_ += 8;
    }
    // std::cout << "After FillBuffer: buffer_ = " << std::hex << buffer_
    //           << ", bits_in_buffer_ = " << bits_in_buffer_ << std::endl;
}

// // **大端序**：高字节优先填充（确保拼接顺序正确）
// while (bits_in_buffer_ <= 56 && cursor_ < data_.size()) {
//     buffer_ = (buffer_ << 8) | data_[cursor_++];  // 高字节优先
//     bits_in_buffer_ += 8;
// }
// std::cout << "After FillBuffer: buffer_ = " << std::hex << buffer_
//           << ", bits_in_buffer_ = " << bits_in_buffer_ << std::endl;
