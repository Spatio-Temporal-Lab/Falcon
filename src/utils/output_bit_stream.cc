#include "output_bit_stream.h"
# include<iostream>
//@初始化data_
//缓冲区大小
OutputBitStream::OutputBitStream(uint32_t buffer_size) {
    data_ = Array<uint32_t>(buffer_size / 4 + 1);
    buffer_ = 0;
    cursor_ = 0;
    bit_in_buffer_ = 0;
}

//@写入
//内容+长度
uint32_t OutputBitStream::Write(uint64_t content, uint32_t len) {
    if (len > 64) {
        std::cerr << "Error: Attempt to write more than 64 bits." << std::endl;
        return 0; // 防止过写
    }
    
    // 如果 buffer_ 的空间不够
    if (cursor_ >= data_.length()) {
        int newsize=(cursor_ + (len + 7) / 8);
        if (newsize > data_.length()) 
        {
            Array<uint32_t> newData(newsize);
            std::copy(data_.begin(), data_.end(), newData.begin());
            data_ = std::move(newData);
        }
        std::cerr << "Error: cursor exceeds data array size." << std::endl;
    }

    content <<= (64 - len);                  // 左移，有效位对齐最高
    buffer_ |= (content >> bit_in_buffer_); // 写入buffer
    bit_in_buffer_ += len;                   // 更新更新长度

    // 检查 bit_in_buffer_ 是否超过 32
    if (bit_in_buffer_ >= 32) {             
        data_[cursor_++] = (buffer_ >> 32); // buffer高32位存入data_
        buffer_ <<= 32;                     // buffer左移
        bit_in_buffer_ -= 32;               // 更新长度
    }
    return len;
}

// uint32_t OutputBitStream::Write(uint64_t content, uint32_t len) {
//     content <<= (64 - len);                 //左移，有效位对齐最高
//     buffer_ |= (content >> bit_in_buffer_); //写入buffer
//     bit_in_buffer_ += len;                  //更新更新长度
//     if (bit_in_buffer_ >= 32) {             //缓冲区满存入data_
//         data_[cursor_++] = (buffer_ >> 32); //buffer高32位存入data_
//         buffer_ <<= 32;                     //buffer左移
//         bit_in_buffer_ -= 32;               //更新长度
//     }
//     return len;
// }

//@长整型写入
//内容+长度
uint32_t OutputBitStream::WriteLong(uint64_t content, uint64_t len) {
    if (len == 0) return 0;
    if (len > 32) {//大于32，分批写
        Write(content >> (len - 32), 32);
        Write(content, len - 32);
        return len;
    }
    return Write(content, len);
}

//@整型写入
//内容+长度
uint32_t OutputBitStream::WriteInt(uint32_t content, uint32_t len) {
    return Write(static_cast<uint64_t>(content), len);
}

//@位写入
//
uint32_t OutputBitStream::WriteBit(bool bit) {
    return Write(static_cast<uint64_t>(bit), 1);
}

//@位写入
//
uint32_t OutputBitStream::WriteByte(uint8_t bit) {
    return Write(static_cast<uint64_t>(bit), 8);
}

//@获取data中指定长度（字节数）的数据
//返回字节数组
Array<uint8_t> OutputBitStream::GetBuffer(uint32_t len) {
    Array<uint8_t> ret(len);
    for (auto &blk : data_) blk = htobe32(blk);//针对每一个32位数据库，转化为大端字节序
    __builtin_memcpy(ret.begin(), data_.begin(), len);//数据放入
    return ret;
}

//@存储并清空buffer
//
void OutputBitStream::Flush() {
    if (bit_in_buffer_) {
        data_[cursor_++] = buffer_ >> 32;
        buffer_ = 0;
        bit_in_buffer_ = 0;
    }
}
//@清空刷新
//
void OutputBitStream::Refresh() {
    cursor_ = 0;
    bit_in_buffer_ = 0;
    buffer_ = 0;
}
