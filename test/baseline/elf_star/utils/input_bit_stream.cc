#include "utils/input_bit_stream.h"

InputBitStream::InputBitStream(uint8_t *raw_data, size_t size) {
  data_ = Array<uint32_t>(std::ceil(static_cast<double>(size) / sizeof(uint32_t)));
  __builtin_memcpy(data_.begin(), raw_data, size);
  for (auto &blk : data_) blk = be32toh(blk);
  buffer_ = (static_cast<uint64_t>(data_[0])) << 32;
  cursor_ = 1;
  bit_in_buffer_ = 32;
}

uint64_t InputBitStream::Peek(size_t len) {
  return buffer_ >> (64 - len);
}

void InputBitStream::Forward(size_t len) {
  bit_in_buffer_ -= len;
  buffer_ <<= len;
  uint64_t load_more_bits = (bit_in_buffer_ < 32) * ((cursor_ < data_.length()) ? 1 : 0);
  uint64_t next_bits = load_more_bits * data_[cursor_];
  buffer_ |= next_bits << (32 - bit_in_buffer_);
  bit_in_buffer_ += load_more_bits * 32;
  cursor_ += load_more_bits;
}

uint64_t InputBitStream::ReadLong(size_t len) {
  uint64_t ret = (len > 32) * Peek(32);     // Peek 32 位并与 ret 的高位部分相加
  Forward((len > 32) * 32);                 // 如果 len > 32，则前进 32 位
  len -= (len > 32) * 32;                   // len 减去 32，避免使用分支
  ret = (ret << len) | Peek(len);           // 处理剩下的位数，直接移位和拼接
  Forward(len);                             // 前进剩余的位
  return ret;
}

uint32_t InputBitStream::ReadInt(size_t len) {
  uint32_t ret = Peek(len);
  Forward(len);
  return ret;
}

uint32_t InputBitStream::ReadBit() {
  uint32_t ret = Peek(1);
  Forward(1);
  return ret;
}

void InputBitStream::SetBuffer(const Array<uint8_t> &new_buffer) {
  data_ = Array<uint32_t>(std::ceil(static_cast<double>(new_buffer.length()) / sizeof(uint32_t)));
  __builtin_memcpy(data_.begin(), new_buffer.begin(), new_buffer.length());
  for (auto &blk : data_) blk = be32toh(blk);
  buffer_ = (static_cast<uint64_t>(data_[0])) << 32;
  cursor_ = 1;
  bit_in_buffer_ = 32;
}

void InputBitStream::SetBuffer(const std::vector<uint8_t> &new_buffer) {
  data_ = Array<uint32_t>(std::ceil(static_cast<double>(new_buffer.size()) / sizeof(uint32_t)));
  __builtin_memcpy(data_.begin(), new_buffer.data(), new_buffer.size());
  for (auto &blk : data_) blk = be32toh(blk);
  buffer_ = (static_cast<uint64_t>(data_[0])) << 32;
  cursor_ = 1;
  bit_in_buffer_ = 32;
}
