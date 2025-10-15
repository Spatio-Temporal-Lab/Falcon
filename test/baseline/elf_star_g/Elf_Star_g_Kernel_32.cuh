//
// 修改后的 Elf_Star_g_Kernel.cuh - 增加时间统计功能
//

#ifndef ELF_STAR_G_KERNEL_CUH
#define ELF_STAR_G_KERNEL_CUH
#include <cuda/std/cstdint>
#include <defs32.cuh>
#include <cstdint>
#include <cstdio>
#define CHUNK_SIZE 1024

#define LOG_2_10 3.32192809489
#define MAX_CHUNK_BYTES 8192

// 性能统计结构体
struct ElfStarTimingInfo_32 {
    float compress_h2d_time;      // 压缩H2D时间 (ms)
    float compress_kernel_time;   // 压缩核函数时间 (ms)
    float compress_d2h_time;      // 压缩D2H时间 (ms)
    float decompress_h2d_time;    // 解压H2D时间 (ms)
    float decompress_kernel_time; // 解压核函数时间 (ms)
    float decompress_d2h_time;    // 解压D2H时间 (ms)
    float total_compress_time;    // 总压缩时间 (ms)
    float total_decompress_time;  // 总解压时间 (ms)
    
    void print() const {
        printf("=== ELF Star 性能统计 ===\n");
        printf("压缩阶段:\n");
        printf("  H2D 时间: %.3f ms\n", compress_h2d_time);
        printf("  核函数时间: %.3f ms\n", compress_kernel_time);
        printf("  D2H 时间: %.3f ms\n", compress_d2h_time);
        printf("  总计: %.3f ms\n", total_compress_time);
        printf("解压阶段:\n");
        printf("  H2D 时间: %.3f ms\n", decompress_h2d_time);
        printf("  核函数时间: %.3f ms\n", decompress_kernel_time);
        printf("  D2H 时间: %.3f ms\n", decompress_d2h_time);
        printf("  总计: %.3f ms\n", total_decompress_time);
        printf("性能占比分析:\n");
        float total_time = total_compress_time + total_decompress_time;
        if (total_time > 0) {
            printf("  压缩核函数占比: %.1f%%\n", (compress_kernel_time / total_time) * 100.0);
            printf("  解压核函数占比: %.1f%%\n", (decompress_kernel_time / total_time) * 100.0);
            printf("  数据传输占比: %.1f%%\n", 
                   ((compress_h2d_time + compress_d2h_time + decompress_h2d_time + decompress_d2h_time) / total_time) * 100.0);
        }
        printf("========================\n");
    }
};

// ==== 设备端内核函数声明 ====

/**
 * @brief GPU解压缩内核
 */
__global__ void decompress_kernel_32(const uint8_t *d_in_data,
                                  const size_t *d_in_offsets,
                                  float *d_out_data,
                                  const size_t *d_out_offsets,
                                  int num_chunks);

/**
 * @brief GPU压缩内核
 */
__global__ void compress_kernel_32(const float *d_in_data,
                                const size_t *d_in_offsets,
                                uint8_t *d_out_data,
                                const size_t *d_out_offsets,
                                size_t *d_compressed_sizes_bytes,
                                uint8_t *d_temp_storage,
                                size_t max_chunk_len_elems,
                                int num_chunks);

// ==== 带时间统计的主机端接口函数声明 ====

/**
 * @brief 带时间统计的压缩接口
 */
ssize_t elf_star_encode_with_timing_32(float *in, ssize_t len, uint8_t **out, 
                                   int64_t **out_compressed_lengths,
                                   int64_t **out_compressed_offsets, 
                                   int64_t **out_decompressed_offsets, 
                                   int *out_num_blocks,
                                   ElfStarTimingInfo_32 *timing_info);

/**
 * @brief 简化的带时间统计的压缩接口
 */
ssize_t elf_star_encode_simple_with_timing_32(const float *in, ssize_t len, 
                                          uint8_t **out, ssize_t *out_len,
                                          ElfStarTimingInfo_32 *timing_info);

/**
 * @brief 带时间统计的解压缩接口
 */
ssize_t elf_star_decode_with_timing_32(const uint8_t *all_in_data,
                                   const size_t *in_offsets_bytes,
                                   const size_t *in_lengths_bytes,
                                   float *all_out_data,
                                   const size_t *out_offsets,
                                   int num_blocks,
                                   ElfStarTimingInfo_32 *timing_info);

/**
 * @brief 简化的带时间统计的解压缩接口
 */
ssize_t elf_star_decode_simple_with_timing_32(const uint8_t *compressed_data, ssize_t compressed_len, 
                                          float **out, ssize_t *out_len,
                                          ElfStarTimingInfo_32 *timing_info);

// ==== 原有接口保持不变 ====

/**
 * @brief 完整的主机端压缩接口
 */
ssize_t elf_star_encode_32(float *in, ssize_t len, uint8_t **out, 
                        int64_t **out_compressed_lengths,
                        int64_t **out_compressed_offsets, 
                        int64_t **out_decompressed_offsets, 
                        int *out_num_blocks);

/**
 * @brief 简化的主机端压缩接口
 */
ssize_t elf_star_encode_simple_32(const float *in, ssize_t len, uint8_t **out, ssize_t *out_len);

/**
 * @brief 完整的主机端解压缩接口
 */
ssize_t elf_star_decode_32(const uint8_t *all_in_data,
                        const size_t *in_offsets_bytes,
                        const size_t *in_lengths_bytes,
                        float *all_out_data,
                        const size_t *out_offsets,
                        int num_blocks);

/**
 * @brief 简化的主机端解压缩接口
 */
ssize_t elf_star_decode_simple_32(const uint8_t *compressed_data, ssize_t compressed_len, 
                              float **out, ssize_t *out_len);

// ==== 工具函数声明 ====

/**
 * @brief 从压缩数据中解析块信息
 */
int elf_star_parse_blocks_32(const uint8_t *compressed_data, ssize_t compressed_len,
                         int *out_num_blocks, size_t **out_block_sizes, 
                         size_t *out_total_elements);

/**
 * @brief 释放elf_star_encode分配的内存
 */
void elf_star_free_encode_result_32(uint8_t *compressed_data, 
                                int64_t *compressed_lengths,
                                int64_t *compressed_offsets,
                                int64_t *decompressed_offsets);

#endif //ELF_STAR_G_KERNEL_CUH