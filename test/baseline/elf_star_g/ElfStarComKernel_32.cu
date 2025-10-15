#include <cstdint>
#include <cstdio>
#include "defs32.cuh"
#include "BitStream/BitWriter.cuh"
#include "utils/post_office_solver32.cuh"
#include "Elf_Star_g_Kernel_32.cuh"

class ElfStarXORCompressor_EdgeSafe_32 {
private:
    int leading_representation[32];
    int trailing_representation[32];
    int leading_round[32];
    int trailing_round[32];
    int storedLeadingZeros = INT_MAX;
    int storedTrailingZeros = INT_MAX;
    uint32_t storedVal = 0;
    bool first = true;
    int leading_bits_per_value;
    int trailing_bits_per_value;
    BitWriter writer;
    
    int lead_distribution[32];
    int trail_distribution[32];

    __device__ __forceinline__ int initLeadingRoundAndRepresentation() {
        int lead_positions[16];
        int num_positions = initRoundAndRepresentation32(
            lead_distribution, leading_representation, leading_round, lead_positions);
        
        if (num_positions <= 0 || num_positions > 16) {
            return 0;
        }
        
        leading_bits_per_value = kPositionLength2Bits[num_positions];
        return write_positions_device32(&writer, lead_positions, num_positions);
    }

    __device__ __forceinline__ int initTrailingRoundAndRepresentation() {
        int trail_positions[16];
        int num_positions = initRoundAndRepresentation32(
            trail_distribution, trailing_representation, trailing_round, trail_positions);
        
        if (num_positions <= 0 || num_positions > 16) {
            return 0;
        }
        
        trailing_bits_per_value = kPositionLength2Bits[num_positions];
        return write_positions_device32(&writer, trail_positions, num_positions);
    }

    __device__ __forceinline__ int writeFirst(uint32_t value) {
        first = false;
        storedVal = value;
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[XOR-writeFirst] value=0x%08X\n", value);
        // }
        
        int trailingZeros;
        if (value == 0) {
            trailingZeros = 32;
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[XOR-writeFirst] value=0, trailingZeros=32\n");
            // }
        } else {
            trailingZeros = __ffs(value) - 1;
            if (trailingZeros < 0) trailingZeros = 0;
            if (trailingZeros > 31) trailingZeros = 31;
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[XOR-writeFirst] trailingZeros=%d\n", trailingZeros);
            // }
        }
        
        write(&writer, trailingZeros, 6);
        
        if (value != 0 && trailingZeros < 31) {
            int mantissaBits = 31 - trailingZeros;
            if (mantissaBits > 0 && mantissaBits <= 31) {
                uint32_t mantissa = storedVal >> (trailingZeros + 1);
                writeLong(&writer, mantissa, mantissaBits);
                
                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[XOR-writeFirst] mantissa=0x%08X, bits=%d\n", mantissa, mantissaBits);
                // }
                return 6 + mantissaBits;
            }
        }
        return 6;
    }
    __device__ __forceinline__ int compressValue(uint32_t value) {
        int thisSize = 0;
        uint32_t _xor = storedVal ^ value;
        
        if (_xor == 0) {
            write(&writer, 1, 2);
            thisSize += 2;
        } else {
            int leading_count = __clz(_xor);
            int trailing_count = (_xor == 0) ? 32 : (__ffs(_xor) - 1);
            
            if (leading_count < 0 || leading_count >= 32) leading_count = 0;
            if (trailing_count < 0 || trailing_count >= 32) trailing_count = 0;
            
            int leadingZeros = leading_round[leading_count];
            int trailingZeros = trailing_round[trailing_count];

            if (leadingZeros >= storedLeadingZeros && trailingZeros >= storedTrailingZeros &&
                (leadingZeros - storedLeadingZeros) + (trailingZeros - storedTrailingZeros) < 
                1 + leading_bits_per_value + trailing_bits_per_value) {
                
                int centerBits = 32 - storedLeadingZeros - storedTrailingZeros;
                if (centerBits <= 0 || centerBits > 32) {
                    storedLeadingZeros = leadingZeros;
                    storedTrailingZeros = trailingZeros;
                    centerBits = 32 - storedLeadingZeros - storedTrailingZeros;
                    if (centerBits > 0 && centerBits <= 32) {
                        write(&writer, 0, 2);
                        write(&writer, (leading_representation[storedLeadingZeros] << trailing_bits_per_value) |
                              trailing_representation[storedTrailingZeros],
                              leading_bits_per_value + trailing_bits_per_value);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                        thisSize += 2 + leading_bits_per_value + trailing_bits_per_value + centerBits;
                    } else {
                        write(&writer, 1, 2);
                        thisSize += 2;
                    }
                } else {
                    int len = 1 + centerBits;
                    if (len > 32) {
                        write(&writer, 1, 1);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                    } else {
                        writeLong(&writer, (1U << centerBits) | (_xor >> storedTrailingZeros), len);
                    }
                    thisSize += len;
                }
            } else {
                storedLeadingZeros = leadingZeros;
                storedTrailingZeros = trailingZeros;
                int centerBits = 32 - storedLeadingZeros - storedTrailingZeros;
                
                if (centerBits <= 0 || centerBits > 32) {
                    write(&writer, 1, 2);
                    thisSize += 2;
                } else {
                    int len = 2 + leading_bits_per_value + trailing_bits_per_value + centerBits;
                    
                    if (len > 32) {
                        write(&writer,
                              (leading_representation[storedLeadingZeros] << trailing_bits_per_value) |
                              trailing_representation[storedTrailingZeros],
                              2 + leading_bits_per_value + trailing_bits_per_value);
                        writeLong(&writer, _xor >> storedTrailingZeros, centerBits);
                    } else {
                        uint32_t combined = ((((uint32_t)leading_representation[storedLeadingZeros] 
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
            initBitWriter(&writer, output + 1, 1);
        }
    }

    __device__ __forceinline__ int addValue(uint32_t value) {
        if (first) {
            int setup_size = initLeadingRoundAndRepresentation() + 
                            initTrailingRoundAndRepresentation();
            if (setup_size == 0) {
                // 设置失败，使用最小设置
                leading_bits_per_value = 4;  // 默认值
                trailing_bits_per_value = 4;
                setup_size = 8;  // 估算值
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
        for (int i = 0; i < 32; i++) {
            lead_distribution[i] = lead_dist[i];
            trail_distribution[i] = trail_dist[i];
        }
    }

    __device__ __forceinline__ BitWriter *getWriter() { return &writer; }
};

class ElfStarCompressor_EdgeSafe_32 {
private:
    size_t size = 32;  // 🔥 关键修复: 32位版本也是32 bits初始值
    int lastBetaStar = INT_MAX;
    int numberOfValues = 0;
    ElfStarXORCompressor_EdgeSafe_32 xorCompressor;
    int leadDistribution[32];
    int trailDistribution[32];

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
    __device__ __forceinline__ void addValue(float v, int *betaStarList, uint32_t *vPrimeList) {
        FLOAT data = {.f = v};
        
        // 🔥 添加调试输出
        // if (numberOfValues == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[压缩-addValue] 第0个值: v=%.6f, raw=0x%08X\n", v, data.i);
        // }
        
        if (v == 0.0f || isinf(v)) {
            vPrimeList[numberOfValues] = data.i;
            betaStarList[numberOfValues] = INT_MAX;
            
            // if (numberOfValues == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[压缩-addValue] 分支: zero/inf, vPrime=0x%08X\n", data.i);
            // }
        } else if (isnan(v)) {
            vPrimeList[numberOfValues] = 0x7fc00000U & data.i;
            betaStarList[numberOfValues] = INT_MAX;
        } else {
            // 正常值
            int alphaAndBetaStar[2];
            getAlphaAndBetaStar_32(v, lastBetaStar, alphaAndBetaStar);
            
            int e = ((int) (data.i >> FLOAT_MANTISSA_BITS)) & 0xff;
            int gAlpha = getFAlpha_32(alphaAndBetaStar[0]) + e - FLOAT_EXPONENT_BIAS;
            int eraseBits = FLOAT_MANTISSA_BITS - gAlpha;
            
            if (eraseBits < 0) eraseBits = 0;
            if (eraseBits > FLOAT_MANTISSA_BITS) eraseBits = FLOAT_MANTISSA_BITS;
            
            // uint32_t mask = (eraseBits >= FLOAT_MANTISSA_BITS) ? 0U : (0xffffffffU << eraseBits);
            uint32_t mask = (eraseBits >= 32) ? 0U : (0xffffffffU << eraseBits);
            uint32_t delta = (~mask) & data.i;
            
            if (delta != 0 && eraseBits > 3) {
                if (alphaAndBetaStar[1] != lastBetaStar) {
                    lastBetaStar = alphaAndBetaStar[1];
                }
                betaStarList[numberOfValues] = lastBetaStar;
                vPrimeList[numberOfValues] = mask & data.i;
                
                // if (numberOfValues == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[压缩-addValue] 分支: 截断, betaStar=%d, vPrime=0x%08X, eraseBits=%d\n",
                //         lastBetaStar, vPrimeList[numberOfValues], eraseBits);
                // }
            } else {
                betaStarList[numberOfValues] = INT_MAX;
                vPrimeList[numberOfValues] = data.i;
                
                // if (numberOfValues == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[压缩-addValue] 分支: 保留原值, vPrime=0x%08X\n", data.i);
                // }
            }
        }
        numberOfValues++;
    }

    __device__ __forceinline__ void calculateDistribution(int len, const uint32_t *vPrimeList) {
        for (int i = 0; i < 32; ++i) {
            leadDistribution[i] = 0;
            trailDistribution[i] = 0;
        }
        
        if (len <= 1) {
            leadDistribution[0] = 1;
            trailDistribution[0] = 1;
            return;
        }
        
        uint32_t lastValue = vPrimeList[0];
        bool hasValidXor = false;
        
        for (int i = 1; i < len; i++) {
            uint32_t xor_val = lastValue ^ vPrimeList[i];
            if (xor_val != 0) {
                int leading_zeros = __clz(xor_val);
                int trailing_zeros = __ffs(xor_val) - 1;
                
                if (leading_zeros >= 0 && leading_zeros < 32) {
                    leadDistribution[leading_zeros]++;
                    hasValidXor = true;
                }
                if (trailing_zeros >= 0 && trailing_zeros < 32) {
                    trailDistribution[trailing_zeros]++;
                }
                lastValue = vPrimeList[i];
            }
        }
        
        if (!hasValidXor) {
            leadDistribution[0] = 1;
            trailDistribution[0] = 1;
        }
    }

    __device__ __forceinline__ void compress(int len, int *betaStarList, uint32_t *vPrimeList) {
        xorCompressor.setDistribution(leadDistribution, trailDistribution);
        
        int compressionLastBetaStar = INT_MAX;
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("\n[压缩-compress] 开始压缩%d个值\n", len);
        //     printf("[压缩-compress] 前3个vPrime: 0x%08X, 0x%08X, 0x%08X\n",
        //         vPrimeList[0], len > 1 ? vPrimeList[1] : 0, len > 2 ? vPrimeList[2] : 0);
        //     printf("[压缩-compress] 前3个betaStar: %d, %d, %d\n",
        //         betaStarList[0], len > 1 ? betaStarList[1] : 0, len > 2 ? betaStarList[2] : 0);
        // }
        
        for (int i = 0; i < len; i++) {
            if (betaStarList[i] == INT_MAX) {
                size += writeInt(2, 2);
            } else if (betaStarList[i] == compressionLastBetaStar) {
                size += writeBit(false);
            } else {
                size += writeInt(betaStarList[i] | 0x18, 5);
                compressionLastBetaStar = betaStarList[i];
            }
            
            // if (i == 0 && blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[压缩-compress] 即将addValue第0个: vPrime=0x%08X\n", vPrimeList[i]);
            // }
            
            size += xorCompressor.addValue(vPrimeList[i]);
        }
    }
    __device__ __forceinline__ void close(int len, uint32_t *out, int *betaStarList, uint32_t *vPrimeList) {
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
        return (bytes > 0) ? bytes : 4;
    }
};

__device__ size_t compress_method_edge_safe_32(
    const float* d_in_chunk, ssize_t in_len,
    uint8_t* d_out_chunk, ssize_t out_len_bytes,
    int* temp_betaStarList, uint32_t* temp_vPrimeList) {
    
    if (in_len <= 0 || out_len_bytes <= 8) {
        return 0;
    }
    
    ElfStarCompressor_EdgeSafe_32 compressor;
    compressor.init((uint32_t*)d_out_chunk, out_len_bytes);
    
    for (int i = 0; i < in_len; i++) {
        compressor.addValue(d_in_chunk[i], temp_betaStarList, temp_vPrimeList);
    }
    
    compressor.close(in_len, (uint32_t*)d_out_chunk, temp_betaStarList, temp_vPrimeList);
    
    size_t result_size = compressor.get_size_bytes();
    return result_size;
}

__global__ void compress_kernel_32(const float* d_in_data,
                                const size_t* d_in_offsets,
                                uint8_t* d_out_data,
                                const size_t* d_out_offsets,
                                size_t* d_compressed_sizes_bytes,
                                uint8_t* d_temp_storage,
                                size_t max_chunk_len_elems,
                                int num_chunks) {
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 🔥 添加总块数输出
    if (chunk_idx == 0) {
        printf("[压缩Kernel] 总块数=%d, max_chunk_len=%llu\n", 
               num_chunks, (unsigned long long)max_chunk_len_elems);
    }
    
    if (chunk_idx >= num_chunks) {
        return;
    }
    
    const size_t in_offset_start = d_in_offsets[chunk_idx];
    const size_t in_offset_end = d_in_offsets[chunk_idx + 1];
    const float* p_in_chunk = d_in_data + in_offset_start;
    const ssize_t in_chunk_len_elems = in_offset_end - in_offset_start;
    
    // 🔥 添加每个块的处理信息
    // if (chunk_idx < 3 || chunk_idx == num_chunks - 1) {
    //     printf("[压缩Kernel] 块%d: offset=%llu-%llu, 元素数=%lld\n",
    //            chunk_idx, (unsigned long long)in_offset_start, 
    //            (unsigned long long)in_offset_end, (long long)in_chunk_len_elems);
    // }
    
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
    
    size_t temp_offset_bytes = chunk_idx * max_chunk_len_elems * (sizeof(int) + sizeof(uint32_t));
    int* p_temp_betaStar = (int*)(d_temp_storage + temp_offset_bytes);
    uint32_t* p_temp_vPrime = (uint32_t*)(d_temp_storage + temp_offset_bytes + max_chunk_len_elems * sizeof(int));
    
    size_t actual_compressed_size = compress_method_edge_safe_32(
        p_in_chunk, in_chunk_len_elems,
        p_out_chunk, out_chunk_len_bytes,
        p_temp_betaStar, p_temp_vPrime);
    
    if (d_compressed_sizes_bytes) {
        d_compressed_sizes_bytes[chunk_idx] = actual_compressed_size;
    }
    
    // 🔥 输出压缩结果
    // if (chunk_idx < 3 || chunk_idx == num_chunks - 1) {
    //     printf("[压缩Kernel] 块%d压缩完成: %llu字节\n",
    //            chunk_idx, (unsigned long long)actual_compressed_size);
    // }
}