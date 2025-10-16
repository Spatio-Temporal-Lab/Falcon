#pragma once

#include <cstdint>

#include "BitDefine.cuh"

typedef struct {
  uint32_t *output;
  int64_t len;
  uint64_t buffer;
  int64_t cursor;
  int64_t bitcnt;
} BitWriter;

__device__ __forceinline__ void
initBitWriter(BitWriter *writer, uint32_t *output, uint64_t outlen) {
  writer->output = output;
  writer->len = outlen;
  writer->buffer = 0;
  writer->cursor = 0;
  writer->bitcnt = 0;
}

__device__ __forceinline__ void
write(BitWriter *writer, uint64_t data, uint64_t length) {
  // assert(length <= 32);
  data <<= (64 - length);
  writer->buffer |= data >> writer->bitcnt;
  writer->bitcnt += length;
  if (writer->bitcnt >= 32) {
    // assert(writer->cursor < writer->len);
    writer->output[writer->cursor++] = writer->buffer >> 32;
    writer->buffer <<= 32;
    writer->bitcnt -= 32;
  }
}

__device__ __forceinline__ void
writeLong(BitWriter *writer, uint64_t data, uint64_t length) {
  // assert(length <= 64);
  if (length == 0) return;
  if (length > 32) {
    write(writer, data >> (length - 32), 32);
    length -= 32;
  }
  write(writer, data, length);
}

__device__ __forceinline__ int
flush(BitWriter *writer) {
  if (writer->bitcnt) {
    // assert(writer->cursor < writer->len);
    writer->output[writer->cursor++] = writer->buffer >> 32;
    writer->buffer = 0;
    writer->bitcnt = 0;
  }
  return writer->cursor;
}