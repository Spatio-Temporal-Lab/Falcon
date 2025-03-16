#ifndef CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_F64_H
#define CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_F64_H
// 压缩常量
static const int cmp_tblock_size = 32; // 每个线程块包含32个常量 Fixed to 32, cannot be modified.
static const int dec_tblock_size = 32; // Fixed to 32, cannot be modified.
static const int cmp_chunk = 1024;     // 每个线程块，包含1024个数据，一个线程处理1024/32个数据
static const int dec_chunk = 1024;
#include <stdint.h>
__global__ void cuSZp_compress_kernel_outlier_f64(const double* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const double eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_outlier_f64(double* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const double eb, const size_t nbEle);
__global__ void cuSZp_compress_kernel_plain_f64(const double* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const double eb, const size_t nbEle);
__global__ void cuSZp_decompress_kernel_plain_f64(double* const __restrict__ decData, const unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const double eb, const size_t nbEle);
__global__ void GDFC_compress_kernel_plain_f64(const double* const __restrict__ oriData, unsigned char* const __restrict__ cmpData, volatile unsigned int* const __restrict__ cmpOffset, volatile unsigned int* const __restrict__ locOffset, volatile int* const __restrict__ flag, const size_t nbEle,uint64_t* bitSizes );

#endif // CUSZP_INCLUDE_CUSZP_CUSZP_KERNELS_F64_H