#ifndef SERF_OUTPUT_BIT_STREAM_H
#define SERF_OUTPUT_BIT_STREAM_H

#include <cstdint>

#include "array.h"

//@输出bit流
//  大端字节序
//  data_：存储编码后的数据块
//  cursor_: 当前写入位置的光标
//  bit_in_buffer_: 当前缓冲区中已写入的位数
//  buffer_: 临时缓冲区，用于逐位写入
class OutputBitStream {
 public:
    explicit OutputBitStream(uint32_t buffer_size);

    uint32_t Write(uint64_t content, uint32_t len);

    uint32_t WriteLong(uint64_t content, uint64_t len);

    uint32_t WriteInt(uint32_t content, uint32_t len);

    uint32_t WriteBit(bool bit);

    uint32_t WriteByte(uint8_t bit);

    void Flush();

    Array<uint8_t> GetBuffer(uint32_t len);

    void Refresh();

   uint32_t GetBufferSize() const {
    return cursor_;
   }
 private:
    Array<uint32_t> data_;    // 存储编码后的数据块
    uint32_t cursor_;         // data中当前写入位置的光标，
    uint32_t bit_in_buffer_;  // 当前缓冲区中已写入的位数
    uint64_t buffer_;         // 临时缓冲区，用于逐位写入
};

#endif  // SERF_OUTPUT_BIT_STREAM_H
