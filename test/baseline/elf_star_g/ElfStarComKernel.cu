#include <cstdint>
#include <cstdio>
#include "defs.cuh"
#include "BitStream/BitWriter.cuh"
#include "utils/post_office_solver.cuh"
#include "Elf_Star_g_Kernel.cuh"

class ElfStarXORCompressor_EdgeSafe {
private:
    int leading_representation[64];
    int trailing_representation[64];
    int leading_round[64];
    int trailing_round[64];
    int storedLeadingZeros = INT_MAX;
    int storedTrailingZeros = INT_MAX;
    uint64_t storedVal = 0;
    bool first = true;
    int leading_bits_per_value;
    int trailing_bits_per_value;
    BitWriter writer;
    
    int lead_distribution[64];
    int trail_distribution[64];

    __device__ __forceinline__ int initLeadingRoundAndRepresentation() {
        int lead_positions[32];
        int num_positions = initRoundAndRepresentation(
            lead_distribution, leading_representation, leading_round, lead_positions);
        
        if (num_positions <= 0 || num_positions > 32) {
            return 0;  // 失败情况
        }
        
        leading_bits_per_value = kPositionLength2Bits[num_positions];
        return write_positions_device(&writer, lead_positions, num_positions);
    }

    __device__ __forceinline__ int initTrailingRoundAndRepresentation() {
        int trail_positions[32];
        int num_positions = initRoundAndRepresentation(
            trail_distribution, trailing_representation, trailing_round, trail_positions);
        
        if (num_positions <= 0 || num_positions > 32) {
            return 0;  // 失败情况
        }
        
        trailing_bits_per_value = kPositionLength2Bits[num_positions];
        return write_positions_device(&writer, trail_positions, num_positions);
    }

    __device__ __forceinline__ int writeFirst(uint64_t value) {
        first = false;
        storedVal = value;
        
        int trailingZeros;
        if (value == 0) {
            trailingZeros = 64;
        } else {
            trailingZeros = __ffsll(value) - 1;
            if (trailingZeros < 0) trailingZeros = 0;
            if (trailingZeros > 63) trailingZeros = 63;
        }
        
        write(&writer, trailingZeros, 7);
        
        if (value != 0 && trailingZeros < 63) {
            int mantissaBits = 63 - trailingZeros;
            if (mantissaBits > 0 && mantissaBits <= 63) {
                writeLong(&writer, storedVal >> (trailingZeros + 1), mantissaBits);
                return 7 + mantissaBits;
            }
        }
        return 7;
    }

    __device__ __forceinline__ int compressValue(uint64_t value) {
        int thisSize = 0;
        uint64_t _xor = storedVal ^ value;
        
        if (_xor == 0) {
            write(&writer, 1, 2);
            thisSize += 2;
        } else {
            int leading_count = __clzll(_xor);
            int trailing_count = (_xor == 0) ? 64 : (__ffsll(_xor) - 1);
            
            // 边界检查
            if (leading_count < 0 || leading_count >= 64) leading_count = 0;
            if (trailing_count < 0 || trailing_count >= 64) trailing_count = 0;
            
            int leadingZeros = leading_round[leading_count];
            int trailingZeros = trailing_round[trailing_count];

            if (leadingZeros >= storedLeadingZeros && trailingZeros >= storedTrailingZeros &&
                (leadingZeros - storedLeadingZeros) + (trailingZeros - storedTrailingZeros) < 
                1 + leading_bits_per_value + trailing_bits_per_value) {
                
                // case 1: 重用
                int centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
                if (centerBits <= 0 || centerBits > 64) {
                    // 回退到新编码
                    storedLeadingZeros = leadingZeros;
                    storedTrailingZeros = trailingZeros;
                    centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
                    if (centerBits > 0 && centerBits <= 64) {
                        write(&writer, 0, 2);
                        write(&writer, (leading_representation[storedLeadingZeros] << trailing_bits_per_value) |
                              trailing_representation[storedTrailingZeros],
                              leading_bits_per_value + trailing_bits_per_value);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                        thisSize += 2 + leading_bits_per_value + trailing_bits_per_value + centerBits;
                    } else {
                        // 极端情况，写入原始值
                        write(&writer, 1, 2);
                        thisSize += 2;
                    }
                } else {
                    int len = 1 + centerBits;
                    if (len > 64) {
                        write(&writer, 1, 1);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                    } else {
                        writeLong(&writer, (1ULL << centerBits) | (_xor >> storedTrailingZeros), len);
                    }
                    thisSize += len;
                }
            } else {
                // case 00: 新的leading/trailing
                storedLeadingZeros = leadingZeros;
                storedTrailingZeros = trailingZeros;
                int centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
                
                if (centerBits <= 0 || centerBits > 64) {
                    // 极端情况处理
                    write(&writer, 1, 2);  // 当作相同值处理
                    thisSize += 2;
                } else {
                    int len = 2 + leading_bits_per_value + trailing_bits_per_value + centerBits;
                    
                    if (len > 64) {
                        write(&writer,
                              (leading_representation[storedLeadingZeros] << trailing_bits_per_value) |
                              trailing_representation[storedTrailingZeros],
                              2 + leading_bits_per_value + trailing_bits_per_value);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                    } else {
                        uint64_t combined = ((((uint64_t)leading_representation[storedLeadingZeros] 
                                             << trailing_bits_per_value) |
                                            trailing_representation[storedTrailingZeros])
                                           << centerBits) |
                                          (_xor >> storedTrailingZeros);
                        writeLong(&writer, combined, len);
                    }
                    thisSize += len;
                }
            }
            storedVal = value;
        }
        return thisSize;
    }

public:
    __device__ __forceinline__ void init(uint32_t *output, size_t out_len_byte) {
        if (out_len_byte > 4) {
            initBitWriter(&writer, output + 1, out_len_byte / 4 - 1);
        } else {
            // 对于极小的输出缓冲区，仍然尝试初始化
            initBitWriter(&writer, output + 1, 1);
        }
    }

    __device__ __forceinline__ int addValue(uint64_t value) {
        if (first) {
            int setup_size = initLeadingRoundAndRepresentation() + 
                            initTrailingRoundAndRepresentation();
            if (setup_size == 0) {
                // 设置失败，使用最小设置
                leading_bits_per_value = 5;  // 默认值
                trailing_bits_per_value = 5;
                setup_size = 10;  // 估算值
            }
            return setup_size + writeFirst(value);
        } else {
            return compressValue(value);
        }
    }

    __device__ __forceinline__ void close() {
        flush(&writer);
    }

    __device__ __forceinline__ void setDistribution(int *lead_dist, int *trail_dist) {
        for (int i = 0; i < 64; i++) {
            lead_distribution[i] = lead_dist[i];
            trail_distribution[i] = trail_dist[i];
        }
    }

    __device__ __forceinline__ BitWriter *getWriter() { return &writer; }
};

class ElfStarCompressor_EdgeSafe {
private:
    size_t size = 32;
    int lastBetaStar = INT_MAX;
    int numberOfValues = 0;
    ElfStarXORCompressor_EdgeSafe xorCompressor;
    int leadDistribution[64];
    int trailDistribution[64];

protected:
    __device__ __forceinline__ int writeInt(int n, int len) {
        write(xorCompressor.getWriter(), n, len);
        return len;
    }

    __device__ __forceinline__ int writeBit(bool bit) {
        write(xorCompressor.getWriter(), bit ? 1 : 0, 1);
        return 1;
    }

public:
    __device__ __forceinline__ void addValue(double v, int *betaStarList, uint64_t *vPrimeList) {
        DOUBLE data = {.d = v};
        
        if (v == 0.0 || isinf(v)) {
            vPrimeList[numberOfValues] = data.i;
            betaStarList[numberOfValues] = INT_MAX;
        } else if (isnan(v)) {
            vPrimeList[numberOfValues] = 0xfff8000000000000ULL & data.i;
            betaStarList[numberOfValues] = INT_MAX;
        } else {
            int alphaAndBetaStar[2];
            getAlphaAndBetaStar(v, lastBetaStar, alphaAndBetaStar);
            int e = ((int) (data.i >> 52)) & 0x7ff;
            int gAlpha = getFAlpha(alphaAndBetaStar[0]) + e - 1023;
            int eraseBits = 52 - gAlpha;
            
            if (eraseBits < 0) eraseBits = 0;
            if (eraseBits > 52) eraseBits = 52;
            
            uint64_t mask = (eraseBits >= 64) ? 0ULL : (0xffffffffffffffffULL << eraseBits);
            uint64_t delta = (~mask) & data.i;
            
            if (delta != 0 && eraseBits > 4) {
                if (alphaAndBetaStar[1] != lastBetaStar) {
                    lastBetaStar = alphaAndBetaStar[1];
                }
                betaStarList[numberOfValues] = lastBetaStar;
                vPrimeList[numberOfValues] = mask & data.i;
            } else {
                betaStarList[numberOfValues] = INT_MAX;
                vPrimeList[numberOfValues] = data.i;
            }
        }
        numberOfValues++;
    }

    __device__ __forceinline__ void calculateDistribution(int len, const uint64_t *vPrimeList) {
        for (int i = 0; i < 64; ++i) {
            leadDistribution[i] = 0;
            trailDistribution[i] = 0;
        }
        
        if (len <= 1) {
            // 单元素情况，设置默认分布
            leadDistribution[0] = 1;
            trailDistribution[0] = 1;
            return;
        }
        
        uint64_t lastValue = vPrimeList[0];
        bool hasValidXor = false;
        
        for (int i = 1; i < len; i++) {
            uint64_t xor_val = lastValue ^ vPrimeList[i];
            if (xor_val != 0) {
                int leading_zeros = __clzll(xor_val);
                int trailing_zeros = __ffsll(xor_val) - 1;
                
                if (leading_zeros >= 0 && leading_zeros < 64) {
                    leadDistribution[leading_zeros]++;
                    hasValidXor = true;
                }
                if (trailing_zeros >= 0 && trailing_zeros < 64) {
                    trailDistribution[trailing_zeros]++;
                }
                lastValue = vPrimeList[i];
            }
        }
        
        if (!hasValidXor) {
            // 如果没有有效的XOR，设置默认分布
            leadDistribution[0] = 1;
            trailDistribution[0] = 1;
        }
    }

    __device__ __forceinline__ void compress(int len, int *betaStarList, uint64_t *vPrimeList) {
        xorCompressor.setDistribution(leadDistribution, trailDistribution);
        
        int compressionLastBetaStar = INT_MAX;
        
        for (int i = 0; i < len; i++) {
            if (betaStarList[i] == INT_MAX) {
                size += writeInt(2, 2);  // '10'
            } else if (betaStarList[i] == compressionLastBetaStar) {
                size += writeBit(false);  // '0'
            } else {
                size += writeInt(betaStarList[i] | 0x30, 6);  // '11xxxx'
                compressionLastBetaStar = betaStarList[i];
            }
            size += xorCompressor.addValue(vPrimeList[i]);
        }
    }

    __device__ __forceinline__ void close(int len, uint32_t *out, int *betaStarList, uint64_t *vPrimeList) {
        calculateDistribution(len, vPrimeList);
        compress(len, betaStarList, vPrimeList);
        xorCompressor.close();
        out[0] = len;
    }

    __device__ __forceinline__ void init(uint32_t* d_out_chunk, size_t length) {
        xorCompressor.init(d_out_chunk, length);
    }

    __device__ __forceinline__ size_t get_size_bytes() {
        size_t bytes = (size + 31) / 32 * 4;
        return (bytes > 0) ? bytes : 4;  // 至少4字节
    }
};

// 边界安全的压缩函数
__device__ size_t compress_method_edge_safe(
    const double* d_in_chunk, ssize_t in_len,
    uint8_t* d_out_chunk, ssize_t out_len_bytes,
    int* temp_betaStarList, uint64_t* temp_vPrimeList) {
    
    if (in_len <= 0 || out_len_bytes <= 8) {
        return 0;
    }
    
    ElfStarCompressor_EdgeSafe compressor;
    compressor.init((uint32_t*)d_out_chunk, out_len_bytes);
    
    for (int i = 0; i < in_len; i++) {
        compressor.addValue(d_in_chunk[i], temp_betaStarList, temp_vPrimeList);
    }
    
    compressor.close(in_len, (uint32_t*)d_out_chunk, temp_betaStarList, temp_vPrimeList);
    
    size_t result_size = compressor.get_size_bytes();
    return (result_size <= out_len_bytes && result_size >= 4) ? result_size : 0;
}

__global__ void compress_kernel(const double* d_in_data,
                                const size_t* d_in_offsets,
                                uint8_t* d_out_data,
                                const size_t* d_out_offsets,
                                size_t* d_compressed_sizes_bytes,
                                uint8_t* d_temp_storage,
                                size_t max_chunk_len_elems,
                                int num_chunks) {
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_idx >= num_chunks) {
        return;
    }
    
    const size_t in_offset_start = d_in_offsets[chunk_idx];
    const size_t in_offset_end = d_in_offsets[chunk_idx + 1];
    const double* p_in_chunk = d_in_data + in_offset_start;
    const ssize_t in_chunk_len_elems = in_offset_end - in_offset_start;
    
    if (in_chunk_len_elems <= 0) {
        if (d_compressed_sizes_bytes) {
            d_compressed_sizes_bytes[chunk_idx] = 0;
        }
        return;
    }
    
    const size_t out_offset_start = d_out_offsets[chunk_idx];
    const size_t out_offset_end = d_out_offsets[chunk_idx + 1];
    uint8_t* p_out_chunk = d_out_data + out_offset_start;
    const ssize_t out_chunk_len_bytes = out_offset_end - out_offset_start;
    
    size_t temp_offset_bytes = chunk_idx * max_chunk_len_elems * (sizeof(int) + sizeof(uint64_t));
    int* p_temp_betaStar = (int*)(d_temp_storage + temp_offset_bytes);
    uint64_t* p_temp_vPrime = (uint64_t*)(d_temp_storage + temp_offset_bytes + max_chunk_len_elems * sizeof(int));
    
    size_t actual_compressed_size = compress_method_edge_safe(
        p_in_chunk, in_chunk_len_elems,
        p_out_chunk, out_chunk_len_bytes,
        p_temp_betaStar, p_temp_vPrime);
    
    if (d_compressed_sizes_bytes) {
        d_compressed_sizes_bytes[chunk_idx] = actual_compressed_size;
    }
    
    // if (chunk_idx < 3) {
    //     printf("块%d: %lld元素 -> %llu字节\n", 
    //            chunk_idx, (long long)in_chunk_len_elems, (unsigned long long)actual_compressed_size);
    // }
}