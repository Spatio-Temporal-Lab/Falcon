#include "Elf_Star_g_Kernel_32.cuh"
#include <BitReader.cuh>
#include <defs32.cuh>
#include <post_office_solver32.cuh>
#include <cuda/std/cstdint>

class ElfStarXORDecompressor_GPU {
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
        int num = read_int(4);
        if (num == 0) {
            num = 16;
        }
        leadingBitsPerValue = kPositionLength2Bits[num];
        leadingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 16; i++) {
            leadingRepresentation[i] = read_int(5);
        }
    }

    __device__ __forceinline__ void initTrailingRepresentation() {
        int num = read_int(4);
        if (num == 0) {
            num = 16;
        }
        trailingBitsPerValue = kPositionLength2Bits[num];
        trailingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 16; i++) {
            trailingRepresentation[i] = read_int(5);
        }
    }

    __device__ __forceinline__ void next() {
        if (first) {
            initLeadingRepresentation();
            initTrailingRepresentation();
            first = false;
            
            int trailingZeros = read_int(6);
            if (trailingZeros < 32) {
                uint32_t mantissa = read_long(31 - trailingZeros);
                storedVal.i = ((mantissa << 1) + 1) << trailingZeros;
            } else {
                storedVal.i = 0;
            }
            
            if (isnan(storedVal.f)) {
                endOfStream = true;
            }
        } else {
            nextValue();
        }
    }

    __device__ __forceinline__ void nextValue() {
        FLOAT value;
        
        // 修复：正确处理所有编码情况
        int first_bit = read_bit();
        
        if (first_bit == 1) {
            // 可能是 '1' (重用) 或者 '10' (值相同) 或者 '11'
            int second_bit = read_bit();
            
            if (second_bit == 0) {
                // case '10' - 这应该不会发生在XOR层，但为了安全处理
                endOfStream = true;
                return;
            } else {
                // case '11' - 这也不应该发生在XOR层
                endOfStream = true; 
                return;
            }
        } else {
            // first_bit == 0
            int second_bit = read_bit();
            
            if (second_bit == 0) {
                // case '00' - 新的leading/trailing
                int leadAndTrail = read_int(leadingBitsPerValue + trailingBitsPerValue);
                int lead = leadAndTrail >> trailingBitsPerValue;
                int trail = leadAndTrail & ~((0xffff << trailingBitsPerValue));

                if (lead >= 0 && lead < leadingRepresentationSize &&
                    trail >= 0 && trail < trailingRepresentationSize) {
                    
                    storedLeadingZeros = leadingRepresentation[lead];
                    storedTrailingZeros = trailingRepresentation[trail];
                    int centerBits = 32 - storedLeadingZeros - storedTrailingZeros;

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
            } else {
                // case '01' - 值相同，不需要读取任何数据
                // storedVal保持不变
            }
        }
    }

public:
    size_t length = 0;

    __device__ __forceinline__ void init(uint32_t *in, size_t len) {
        if (len > 1) {
            initBitReader(&reader, in + 1, len - 1);
            length = in[0];
        } else {
            length = 0;
            endOfStream = true;
        }
    }

    __device__ __forceinline__ float readValue() {
        if (endOfStream) {
            return -1;
        }
        
        next();
        
        if (endOfStream) {
            return -1;
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

// 让我修正XOR解压器以完全匹配CPU版本
class ElfStarXORDecompressor_CPU_Compatible {
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
        int num = read_int(4);
        if (num == 0) {
            num = 16;
        }
        leadingBitsPerValue = kPositionLength2Bits[num];
        leadingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 32; i++) {
            leadingRepresentation[i] = read_int(5);
        }
    }

    __device__ __forceinline__ void initTrailingRepresentation() {
        int num = read_int(4);
        if (num == 0) {
            num = 16;
        }
        trailingBitsPerValue = kPositionLength2Bits[num];
        trailingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 32; i++) {
            trailingRepresentation[i] = read_int(5);
        }
    }

    __device__ __forceinline__ void next() {
        if (first) {
            initLeadingRepresentation();
            initTrailingRepresentation();
            first = false;
            
            int trailingZeros = read_int(6);
            if (trailingZeros < 32) {
                uint32_t mantissa = read_long(31 - trailingZeros);
                storedVal.i = ((mantissa << 1) + 1) << trailingZeros;
            } else {
                storedVal.i = 0;
            }
            
            if (isnan(storedVal.f)) {
                endOfStream = true;
            }
        } else {
            nextValue();
        }
    }

    // 完全按照CPU版本的nextValue逻辑
    __device__ __forceinline__ void nextValue() {
        FLOAT value;
        int centerBits;

        if (read_bit() == 1) {
            // case 1: 重用leading/trailing
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
        } else if (read_bit() == 0) {
            // case 00: 新的leading/trailing
            int leadAndTrail = read_int(leadingBitsPerValue + trailingBitsPerValue);
            int lead = leadAndTrail >> trailingBitsPerValue;
            int trail = ~(0xffff << trailingBitsPerValue) & leadAndTrail;

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
        // CPU版本没有处理'01'的情况，这可能导致问题
        // 但为了完全兼容，我们保持相同的行为
    }

public:
    size_t length = 0;

    __device__ __forceinline__ void init(uint32_t *in, size_t len) {
        if (len > 1) {
            initBitReader(&reader, in + 1, len - 1);
            length = in[0];
        } else {
            length = 0;
            endOfStream = true;
        }
    }

    __device__ __forceinline__ float readValue() {
        if (endOfStream) {
            return -1;
        }
        
        next();
        
        if (endOfStream) {
            return -1;
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

class ElfStarDecompressor_GPU {
private:
    ElfStarXORDecompressor_CPU_Compatible xorDecompressor;
    int lastBetaStar = INT_MAX;

    __device__ __forceinline__ float nextValue() {
        float v;
        
        if (read_int(1) == 0) {
            // case 0
            v = recoverVByBetaStar();
        } else if (read_int(1) == 0) {
            // case 10
            v = xorDecompressor.readValue();
        } else {
            // case 11
            lastBetaStar = read_int(4);
            v = recoverVByBetaStar();
        }
        return v;
    }

    __device__ __forceinline__ float recoverVByBetaStar() {
        float vPrime = xorDecompressor.readValue();
        
        if (xorDecompressor.isEndOfStream()) {
            return -1;
        }
        
        float v;
        int sp = getSP_32(fabs(vPrime));
        
        if (lastBetaStar == 0) {
            v = get10iN_32(-sp - 1);
            if (vPrime < 0) {
                v = -v;
            }
        } else {
            int alpha = lastBetaStar - sp - 1;
            v = roundUp_32(vPrime, alpha);
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
        
        for (int i = 0; i < len; i++) {
            float value = nextValue();
            if (xorDecompressor.isEndOfStream()) {
                return i;
            }
            output[i] = value;
        }
        
        return len;
    }
};

__device__ int decompress_method_final_fix(
    uint8_t *d_in, ssize_t len, float *d_out_chunks, int thread_id) {
    
    if (len <= 4 || !d_in || !d_out_chunks) {
        return 0;
    }
    
    // 确保内存对齐
    uint32_t num_elements;
    if (((uintptr_t)d_in) % 4 == 0) {
        num_elements = ((uint32_t*)d_in)[0];
    } else {
        // 手动读取4字节
        uint8_t bytes[4];
        for (int i = 0; i < 4; i++) {
            bytes[i] = d_in[i];
        }
        num_elements = *((uint32_t*)bytes);
    }
    
    if (num_elements == 0 || num_elements > 100000) {
        return 0;
    }
    
    ElfStarDecompressor_GPU decompressor;
    decompressor.init((uint32_t*)d_in, len / 4);
    
    int result = decompressor.decompress(d_out_chunks);
    
    return result;
}

__device__ void decompress_method(uint8_t *d_in, ssize_t len, float *d_out_chunks) {
    decompress_method_final_fix(d_in, len, d_out_chunks, 0);
}

__global__ void decompress_kernel(const uint8_t* d_in_data,
                                const size_t* d_in_offsets,
                                float* d_out_data,
                                const size_t* d_out_offsets,
                                int num_chunks) {
    int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (chunk_idx >= num_chunks) {
        return;
    }

    const size_t in_offset_start = d_in_offsets[chunk_idx];
    const size_t in_offset_end = d_in_offsets[chunk_idx + 1];
    uint8_t* p_in_chunk = const_cast<uint8_t*>(d_in_data + in_offset_start);
    const ssize_t in_chunk_len_bytes = in_offset_end - in_offset_start;

    const size_t out_offset_start = d_out_offsets[chunk_idx];
    float* p_out_chunk = d_out_data + out_offset_start;

    if (in_chunk_len_bytes <= 4) {
        return;
    }

    int decompressed_count = decompress_method_final_fix(
        p_in_chunk, in_chunk_len_bytes, p_out_chunk, chunk_idx);
    
    // if (chunk_idx < 3) {
    //     printf("块%d解压: %lld字节 -> %d元素\n", 
    //            chunk_idx, (long long)in_chunk_len_bytes, decompressed_count);
    // }
}