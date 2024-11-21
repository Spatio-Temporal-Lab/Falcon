//
// Created by lz on 24-9-26.
//
// GDFCompressor.cuh

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <cstring>

// 定义常量
#define BLOCK_SIZE_G 1024
#define POW_NUM_G ((1L << 51) + (1L << 52))


__global__ static void compressBlockKernel(const double* input, int blockSize, int totalSize, unsigned char* output, int* bitSizes);

// 核函数
__device__ static unsigned long zigzag_encode_cuda(long value) {
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}

__device__ static int getDecimalPlaces(double value) {
    double trac = value + POW_NUM_G - POW_NUM_G;
    double temp = value;
    int digits = 0;
    int64_t int_temp, trac_temp;
    memcpy(&int_temp, &temp, sizeof(double));
    memcpy(&trac_temp, &trac, sizeof(double));
    while (std::abs(trac_temp - int_temp) > 1 && digits < 16) {
        digits++;
        double td = pow(10, digits);
        temp = value * td;
        memcpy(&int_temp, &temp, sizeof(double));
        trac = temp + POW_NUM_G - POW_NUM_G;
        memcpy(&trac_temp, &trac, sizeof(double));
    }
    return digits;
}

class GDFCompressor {
public:
    GDFCompressor(int blockSize = BLOCK_SIZE_G) : blockSize(blockSize) {}

    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);

private:
    int blockSize;

    void setupDeviceMemory(const std::vector<double>& input, double*& d_input, unsigned char*& d_output, int*& d_bitSizes);

    void freeDeviceMemory(double* d_input, unsigned char* d_output, int* d_bitSizes);
};