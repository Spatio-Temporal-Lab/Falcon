#ifndef CDF_DECOMPRESSOR_H
#define CDF_DECOMPRESSOR_H

#include <vector>
#include <iostream>
#include <thread> 
#include "input_bit_stream.h"

class Falcon_basic_decompressor {
public:
    void decompress(const std::vector<unsigned char>& input, std::vector<double>& output);

private:
    long zigzag_decode(unsigned long value);
    double decodeDoubleWithSignLast(uint64_t value);
    void decompressBlock(InputBitStream& bitStream, std::vector<long>& integers, int& totalBitsRead,size_t blockSize,int& maxDecimalPlaces,int& isOk);
};

#endif // CDF_DECOMPRESSOR_H