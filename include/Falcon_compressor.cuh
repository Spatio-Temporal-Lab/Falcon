//
// Created by lz on 24-9-26.
//
// Falcon_compressor.cuh

#include <cuda_runtime.h>
#include <thread> 
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
static const int cmp_chunk = 1025;     // 每个线程块，包含1024个数据，一个线程处理1024/32个数据
static const int dec_chunk = 1025;

#define BLOCK_SIZE_G 32
#define POW_NUM_G ((1L << 51) + (1L << 52))
#define DATA_PER_THREAD 1025//
#define MAX_NUMS_PER_CHUNK 1025*1024*8
#define DATA_PER_ONE 32

#define MAX_BITCOUNT 64
#define MAX_BITSIZE_PER_BLOCK (64 + 64 + 8 + 8 +8 + 64 + (DATA_PER_THREAD) * MAX_BITCOUNT)
#define MAX_BYTES_PER_BLOCK ((MAX_BITSIZE_PER_BLOCK + 7) / 8)

class FalconCompressor {
public:
    FalconCompressor(){}

    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);
    static void Falcon_compress(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream);
    static void Falcon_compress_stream(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    // 分离位打包
    static void Falcon_compress_no_pack(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    // 暴力计算
    static void Falcon_compress_br(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    // spare
    static void Falcon_compress_spare(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
    // // string
    // static void Falcon_compress_string(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream);
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


