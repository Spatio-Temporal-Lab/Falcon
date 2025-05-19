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
#include <thrust/device_vector.h>
// 压缩常量
static const int cmp_tblock_size = 32; // 每个线程块包含32个常量 Fixed to 32, cannot be modified.
static const int dec_tblock_size = 32; // Fixed to 32, cannot be modified.
static const int cmp_chunk = 1024;     // 每个线程块，包含1024个数据，一个线程处理1024/32个数据
static const int dec_chunk = 1024;

class GDFCompressor {
public:
    GDFCompressor(){}

    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);
    static void GDFC_compress(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream);

private:

    void setupDeviceMemory(
    const std::vector<double>& input,
    double*& d_input,
    unsigned char*& d_output,
    uint64_t*& d_bitSizes
    ); 
    void freeDeviceMemory(
    double* d_input,
    unsigned char* d_output,
    uint64_t* d_bitSizes
    );

};

