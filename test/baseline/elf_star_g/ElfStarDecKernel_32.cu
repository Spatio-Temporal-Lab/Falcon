// 🔍 完整调试版本 - 32位解压器

#include "Elf_Star_g_Kernel_32.cuh"
#include <BitReader.cuh>
#include <defs32.cuh>
#include <post_office_solver32.cuh>
#include <cuda/std/cstdint>

class ElfStarXORDecompressor_Debug_32 {
private:
    FLOAT storedVal = {.i = 0};
    int storedLeadingZeros = INT_MAX;
    int storedTrailingZeros = INT_MAX;
    bool first = true;
    bool endOfStream = false;
    BitReader reader;

    int leadingRepresentation[32];
    int trailingRepresentation[32];
    int leadingRepresentationSize;
    int trailingRepresentationSize;
    int leadingBitsPerValue;
    int trailingBitsPerValue;

    __device__ __forceinline__ int read_int(int length) { 
        return readInt(&reader, length); 
    }
    
    __device__ __forceinline__ int read_bit() { 
        return readInt(&reader, 1); 
    }
    
    __device__ __forceinline__ uint32_t read_long(int length) { 
        return readLong(&reader, length); 
    }

    __device__ __forceinline__ void initLeadingRepresentation() {
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[XOR-读取] 读取Leading前: buffer=0x%016llX, bitcnt=%lld\n",
        //            reader.buffer, reader.bitcnt);
        // }
        
        int num = read_int(4);
        if (num == 0) num = 16;
        leadingBitsPerValue = kPositionLength2Bits[num];
        leadingRepresentationSize = num;
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[XOR-读取] Leading位置数=%d, 每值bit数=%d\n", 
        //            num, leadingBitsPerValue);
        //     printf("[XOR-读取] 读取4位后: buffer=0x%016llX, bitcnt=%lld\n",
        //            reader.buffer, reader.bitcnt);
        // }
        
        for (int i = 0; i < num && i < 32; i++) {
            leadingRepresentation[i] = read_int(5);
        }
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[XOR-读取] 读取Leading完成: buffer=0x%016llX, bitcnt=%lld\n",
        //            reader.buffer, reader.bitcnt);
        // }
    }

    __device__ __forceinline__ void initTrailingRepresentation() {
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[XOR-读取] 读取Trailing前: buffer=0x%016llX, bitcnt=%lld\n",
        //            reader.buffer, reader.bitcnt);
        // }
        
        int num = read_int(4);
        if (num == 0) num = 16;
        trailingBitsPerValue = kPositionLength2Bits[num];
        trailingRepresentationSize = num;
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[XOR-读取] Trailing位置数=%d, 每值bit数=%d\n", 
        //            num, trailingBitsPerValue);
        //     printf("[XOR-读取] 读取4位后: buffer=0x%016llX, bitcnt=%lld\n",
        //            reader.buffer, reader.bitcnt);
        // }
        
        for (int i = 0; i < num && i < 32; i++) {
            trailingRepresentation[i] = read_int(5);
        }
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[XOR-读取] 读取Trailing完成: buffer=0x%016llX, bitcnt=%lld\n",
        //            reader.buffer, reader.bitcnt);
        // }
    }

    __device__ __forceinline__ void next() {
        if (first) {
            initLeadingRepresentation();
            initTrailingRepresentation();
            first = false;
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("\n[XOR-读取第一个值] 开始\n");
            //     printf("[XOR-读取] 初始buffer=0x%016llX, bitcnt=%lld, cursor=%lld\n",
            //            reader.buffer, reader.bitcnt, reader.cursor);
            // }
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[XOR-读取] 准备读取6位trailingZeros\n");
            //     printf("[XOR-读取] 读取前buffer=0x%016llX, bitcnt=%lld\n",
            //            reader.buffer, reader.bitcnt);
            // }
            
            int trailingZeros = read_int(6);
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[XOR-读取] trailingZeros=%d\n", trailingZeros);
            //     printf("[XOR-读取] 读取后buffer=0x%016llX, bitcnt=%lld\n",
            //            reader.buffer, reader.bitcnt);
            // }
            
            if (trailingZeros < 32) {
                int mantissaBits = 31 - trailingZeros;
                
                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[XOR-读取] mantissaBits=%d\n", mantissaBits);
                // }
                
                if (mantissaBits > 0) {
                    // if (blockIdx.x == 0 && threadIdx.x == 0) {
                    //     printf("[XOR-读取] 准备读取%d位mantissa\n", mantissaBits);
                    //     printf("[XOR-读取] 读取前buffer=0x%016llX, bitcnt=%lld\n",
                    //            reader.buffer, reader.bitcnt);
                    //     printf("[XOR-读取] 当前buffer高%d位: 0x%08X\n",
                    //            mantissaBits, (uint32_t)(reader.buffer >> (64 - mantissaBits)));
                    // }
                    
                    uint32_t mantissa = read_long(mantissaBits);
                    
                    // if (blockIdx.x == 0 && threadIdx.x == 0) {
                    //     printf("[XOR-读取] 读到mantissa=0x%08X\n", mantissa);
                    //     printf("[XOR-读取] 读取后buffer=0x%016llX, bitcnt=%lld\n",
                    //            reader.buffer, reader.bitcnt);
                    // }
                    
                    storedVal.i = ((mantissa << 1) + 1) << trailingZeros;
                    
                    // if (blockIdx.x == 0 && threadIdx.x == 0) {
                    //     printf("[XOR-读取] 计算: ((0x%08X << 1) + 1) << %d = 0x%08X\n",
                    //            mantissa, trailingZeros, storedVal.i);
                    //     printf("[XOR-读取] storedVal.f=%.6f\n", storedVal.f);
                    // }
                } else {
                    storedVal.i = 1 << trailingZeros;
                    
                    // if (blockIdx.x == 0 && threadIdx.x == 0) {
                    //     printf("[XOR-读取] 特殊情况: mantissaBits=0\n");
                    //     printf("[XOR-读取] storedVal.i=0x%08X\n", storedVal.i);
                    // }
                }
            } else {
                storedVal.i = 0;
                
                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[XOR-读取] trailingZeros>=32, 值为0\n");
                // }
            }
            
            if (isnan(storedVal.f)) {
                endOfStream = true;
                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[XOR-读取] 检测到NaN, 流结束\n");
                // }
            }
        } else {
            nextValue();
        }
    }

    __device__ __forceinline__ void nextValue() {
        FLOAT value;
        int centerBits;

        int first_bit = read_bit();
        
        if (first_bit == 1) {
            centerBits = 32 - storedLeadingZeros - storedTrailingZeros;
            if (centerBits > 0 && centerBits <= 32) {
                value.i = read_long(centerBits) << storedTrailingZeros;
                value.i = storedVal.i ^ value.i;
                if (isnan(value.f)) {
                    endOfStream = true;
                } else {
                    storedVal = value;
                }
            } else {
                endOfStream = true;
            }
        } else {
            int second_bit = read_bit();
            
            if (second_bit == 0) {
                int leadAndTrail = read_int(leadingBitsPerValue + trailingBitsPerValue);
                int lead = leadAndTrail >> trailingBitsPerValue;
                int trail = leadAndTrail & ((1 << trailingBitsPerValue) - 1);

                if (lead >= 0 && lead < leadingRepresentationSize &&
                    trail >= 0 && trail < trailingRepresentationSize) {
                    
                    storedLeadingZeros = leadingRepresentation[lead];
                    storedTrailingZeros = trailingRepresentation[trail];
                    centerBits = 32 - storedLeadingZeros - storedTrailingZeros;

                    if (centerBits > 0 && centerBits <= 32) {
                        value.i = read_long(centerBits) << storedTrailingZeros;
                        value.i = storedVal.i ^ value.i;
                        if (isnan(value.f)) {
                            endOfStream = true;
                        } else {
                            storedVal = value;
                        }
                    } else {
                        endOfStream = true;
                    }
                } else {
                    endOfStream = true;
                }
            }
        }
    }

public:
    size_t length = 0;

    __device__ __forceinline__ void init(uint32_t *in, size_t len) {
        if (len > 1) {
            initBitReader(&reader, in + 1, len - 1);
            length = in[0];
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[XOR-读取初始化] length=%llu, buffer长度=%llu\n", length, len);
            //     printf("[XOR-读取初始化] 前4个uint32:\n");
            //     for (int i = 0; i < 4 && i < len; i++) {
            //         printf("  in[%d]=0x%08X\n", i, in[i]);
            //     }
            //     printf("[XOR-读取初始化] 初始buffer=0x%016llX\n", reader.buffer);
            // }
        } else {
            length = 0;
            endOfStream = true;
        }
    }

    __device__ __forceinline__ float readValue() {
        if (endOfStream) {
            return -1.0f;
        }
        
        next();
        
        if (endOfStream) {
            return -1.0f;
        }
        
        return storedVal.f;
    }

    __device__ __forceinline__ BitReader *getReader() {
        return &reader;
    }
    
    __device__ __forceinline__ bool isEndOfStream() {
        return endOfStream;
    }
};

// ElfStar解压缩器 (使用调试版XOR)
class ElfStarDecompressor_Debug_32 {
private:
    ElfStarXORDecompressor_Debug_32 xorDecompressor;
    int lastBetaStar = INT_MAX;

    __device__ __forceinline__ float nextValue() {
        float v;
        
        int first_bit = read_int(1);
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[ElfStar-读取] 读取标记bit: first=%d\n", first_bit);
        // }
        
        if (first_bit == 0) {
            v = recoverVByBetaStar();
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[ElfStar-读取] case '0', 恢复值=%.6f\n", v);
            // }
        } else {
            int second_bit = read_int(1);
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[ElfStar-读取] second_bit=%d\n", second_bit);
            // }
            
            if (second_bit == 0) {
                v = xorDecompressor.readValue();
                
                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[ElfStar-读取] case '10', 直接值=%.6f\n", v);
                // }
            } else {
                lastBetaStar = read_int(3);
                v = recoverVByBetaStar();
                
                // if (blockIdx.x == 0 && threadIdx.x == 0) {
                //     printf("[ElfStar-读取] case '11', betaStar=%d, 恢复值=%.6f\n", 
                //            lastBetaStar, v);
                // }
            }
        }
        
        return v;
    }

    __device__ __forceinline__ float recoverVByBetaStar() {
        float vPrime = xorDecompressor.readValue();
        
        if (xorDecompressor.isEndOfStream()) {
            return -1.0f;
        }
        
        float v;
        int sp = getSP_32(fabsf(vPrime));
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[Recover] vPrime=%.6f, sp=%d, betaStar=%d\n", 
        //            vPrime, sp, lastBetaStar);
        // }
        
        if (lastBetaStar == 0) {
            v = get10iN_32(-sp - 1);
            if (vPrime < 0.0f) {
                v = -v;
            }
        } else {
            int alpha = lastBetaStar - sp - 1;
            v = roundUp_32(vPrime, alpha);
            
            // if (blockIdx.x == 0 && threadIdx.x == 0) {
            //     printf("[Recover] alpha=%d, 最终值=%.6f\n", alpha, v);
            // }
        }
        
        return v;
    }

protected:
    __device__ __forceinline__ int read_int(int len) {
        return readInt(xorDecompressor.getReader(), len);
    }
    
    __device__ __forceinline__ int getLength() {
        return xorDecompressor.length;
    }

public:
    __device__ __forceinline__ void init(uint32_t *in, size_t len) { 
        xorDecompressor.init(in, len);
        lastBetaStar = INT_MAX;
    }

    __device__ __forceinline__ int decompress(float *output) {
        int len = getLength();
        
        if (len <= 0 || output == nullptr) {
            return 0;
        }
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[ElfStar解压] 开始, 期望%d个元素\n", len);
        // }
        
        for (int i = 0; i < len; i++) {
            float value = nextValue();
            if (xorDecompressor.isEndOfStream()) {
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    printf("[ElfStar解压] 提前结束于第%d个元素\n", i);
                }
                return i;
            }
            output[i] = value;
            
            // if (blockIdx.x == 0 && threadIdx.x == 0 && i == 0) {
            //     printf("[ElfStar解压] 第0个元素=%.6f\n", value);
            // }
        }
        
        // if (blockIdx.x == 0 && threadIdx.x == 0) {
        //     printf("[ElfStar解压] 完成, 共%d个元素\n", len);
        // }
        
        return len;
    }
};

// 调试版解压函数
__device__ int decompress_method_debug_32(
    uint8_t *d_in, ssize_t len, float *d_out_chunks, int thread_id) {
    
    if (len <= 4 || !d_in || !d_out_chunks) {
        return 0;
    }
    
    uint32_t num_elements;
    if (((uintptr_t)d_in) % 4 == 0) {
        num_elements = ((uint32_t*)d_in)[0];
    } else {
        uint8_t bytes[4];
        for (int i = 0; i < 4; i++) {
            bytes[i] = d_in[i];
        }
        num_elements = *((uint32_t*)bytes);
    }
    
    // if (thread_id == 0) {
    //     printf("\n=== 块%d 解压开始 ===\n", thread_id);
    //     printf("输入长度: %lld字节\n", (long long)len);
    //     printf("元素数量: %u\n", num_elements);
    // }
    
    if (num_elements == 0 || num_elements > 100000) {
        if (thread_id == 0) {
            printf("元素数量异常!\n");
        }
        return 0;
    }
    
    ElfStarDecompressor_Debug_32 decompressor;
    decompressor.init((uint32_t*)d_in, len / 4);
    
    int result = decompressor.decompress(d_out_chunks);
    
    // if (thread_id == 0) {
    //     printf("=== 块%d 解压完成, 返回%d个元素 ===\n\n", thread_id, result);
    // }
    
    return result;
}

__global__ void decompress_kernel_32(const uint8_t* d_in_data,
                                const size_t* d_in_offsets,
                                float* d_out_data,
                                const size_t* d_out_offsets,
                                int num_chunks) {
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 🔥 添加总块数输出
    if (chunk_idx == 0) {
        printf("[解压Kernel] 总块数=%d\n", num_chunks);
    }

    if (chunk_idx >= num_chunks) {
        return;
    }

    const size_t in_offset_start = d_in_offsets[chunk_idx];
    const size_t in_offset_end = d_in_offsets[chunk_idx + 1];
    uint8_t* p_in_chunk = const_cast<uint8_t*>(d_in_data + in_offset_start);
    const ssize_t in_chunk_len_bytes = in_offset_end - in_offset_start;

    const size_t out_offset_start = d_out_offsets[chunk_idx];
    float* p_out_chunk = d_out_data + out_offset_start;

    // 🔥 添加每个块的处理信息
    // if (chunk_idx < 3 || chunk_idx == num_chunks - 1) {
    //     printf("[解压Kernel] 块%d: 输入=%llu-%llu (%lld字节), 输出offset=%llu\\n",
    //            chunk_idx, 
    //            (unsigned long long)in_offset_start, 
    //            (unsigned long long)in_offset_end,
    //            (long long)in_chunk_len_bytes,
    //            (unsigned long long)out_offset_start);
    // }

    if (in_chunk_len_bytes <= 4) {
        return;
    }

    int decompressed_count = decompress_method_debug_32(
        p_in_chunk, in_chunk_len_bytes, p_out_chunk, chunk_idx);
    
    // 🔥 输出解压结果
    // if (chunk_idx < 3 || chunk_idx == num_chunks - 1) {
    //     printf("[解压Kernel] 块%d解压完成: %d个元素\n",
    //            chunk_idx, decompressed_count);
    // }
}