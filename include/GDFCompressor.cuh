//
// Created by lz on 24-9-26.
//
// GDFCompressor.cuh

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <cstring>

// 定义常量
#define BLOCK_SIZE_G 16
#define POW_NUM_G ((1L << 51) + (1L << 52))

class GDFCompressor {
public:
    GDFCompressor(int blockSize = BLOCK_SIZE_G) : blockSize(blockSize) {}

    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);

private:
    int blockSize;

    void setupDeviceMemory(const std::vector<double>& input, double*& d_input, unsigned char*& d_output, uint64_t*& d_bitSizes);

    void freeDeviceMemory(double* d_input, unsigned char* d_output, uint64_t* d_bitSizes);

};