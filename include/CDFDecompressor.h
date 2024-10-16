#ifndef CDF_DECOMPRESSOR_H
#define CDF_DECOMPRESSOR_H

#include <vector>
#include <iostream>
#include "input_bit_stream.h"

class CDFDecompressor {
public:
    void decompress(const std::vector<unsigned char>& input, std::vector<double>& output);

private:
    long zigzag_decode(unsigned long value);
    void deltaDecode(const std::vector<long>& encoded, std::vector<long>& decoded);
    void decompressBlock(InputBitStream& bitStream, std::vector<long>& integers, int& totalBitsRead);
};

#endif // CDF_DECOMPRESSOR_H
