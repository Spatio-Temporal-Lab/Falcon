#include "Elf_Star_g_Kernel.cuh"
#include <BitReader.cuh>
#include <defs.cuh>
#include <post_office_solver.cuh>
#include <cuda/std/cstdint>

class ElfStarXORDecompressor_GPU {
private:
    DOUBLE storedVal = {.i = 0};
    int storedLeadingZeros = INT_MAX;
    int storedTrailingZeros = INT_MAX;
    bool first = true;
    bool endOfStream = false;
    BitReader reader;

    int leadingRepresentation[64];
    int trailingRepresentation[64];
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
    
    __device__ __forceinline__ uint64_t read_long(int length) { 
        return readLong(&reader, length); 
    }

    __device__ __forceinline__ void initLeadingRepresentation() {
        int num = read_int(5);
        if (num == 0) {
            num = 32;
        }
        leadingBitsPerValue = kPositionLength2Bits[num];
        leadingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 64; i++) {
            leadingRepresentation[i] = read_int(6);
        }
    }

    __device__ __forceinline__ void initTrailingRepresentation() {
        int num = read_int(5);
        if (num == 0) {
            num = 32;
        }
        trailingBitsPerValue = kPositionLength2Bits[num];
        trailingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 64; i++) {
            trailingRepresentation[i] = read_int(6);
        }
    }

    __device__ __forceinline__ void next() {
        if (first) {
            initLeadingRepresentation();
            initTrailingRepresentation();
            first = false;
            
            int trailingZeros = read_int(7);
            if (trailingZeros < 64) {
                uint64_t mantissa = read_long(63 - trailingZeros);
                storedVal.i = ((mantissa << 1) + 1) << trailingZeros;
            } else {
                storedVal.i = 0;
            }
            
            if (isnan(storedVal.d)) {
                endOfStream = true;
            }
        } else {
            nextValue();
        }
    }

    __device__ __forceinline__ void nextValue() {
        DOUBLE value;
        
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
                int trail = leadAndTrail & ((1 << trailingBitsPerValue) - 1);

                if (lead >= 0 && lead < leadingRepresentationSize &&
                    trail >= 0 && trail < trailingRepresentationSize) {
                    
                    storedLeadingZeros = leadingRepresentation[lead];
                    storedTrailingZeros = trailingRepresentation[trail];
                    int centerBits = 64 - storedLeadingZeros - storedTrailingZeros;

                    if (centerBits > 0 && centerBits <= 64) {
                        value.i = read_long(centerBits) << storedTrailingZeros;
                        value.i = storedVal.i ^ value.i;
                        if (isnan(value.d)) {
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

    __device__ __forceinline__ double readValue() {
        if (endOfStream) {
            return -1;
        }
        
        next();
        
        if (endOfStream) {
            return -1;
        }
        
        return storedVal.d;
    }

    __device__ __forceinline__ BitReader *getReader() {
        return &reader;
    }
    
    __device__ __forceinline__ bool isEndOfStream() {
        return endOfStream;
    }
};

// 等一下，我重新分析CPU版本的逻辑...
// 让我修正XOR解压器以完全匹配CPU版本
class ElfStarXORDecompressor_CPU_Compatible {
private:
    DOUBLE storedVal = {.i = 0};
    int storedLeadingZeros = INT_MAX;
    int storedTrailingZeros = INT_MAX;
    bool first = true;
    bool endOfStream = false;
    BitReader reader;

    int leadingRepresentation[64];
    int trailingRepresentation[64];
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
    
    __device__ __forceinline__ uint64_t read_long(int length) { 
        return readLong(&reader, length); 
    }

    __device__ __forceinline__ void initLeadingRepresentation() {
        int num = read_int(5);
        if (num == 0) {
            num = 32;
        }
        leadingBitsPerValue = kPositionLength2Bits[num];
        leadingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 64; i++) {
            leadingRepresentation[i] = read_int(6);
        }
    }

    __device__ __forceinline__ void initTrailingRepresentation() {
        int num = read_int(5);
        if (num == 0) {
            num = 32;
        }
        trailingBitsPerValue = kPositionLength2Bits[num];
        trailingRepresentationSize = num;
        
        for (int i = 0; i < num && i < 64; i++) {
            trailingRepresentation[i] = read_int(6);
        }
    }

    __device__ __forceinline__ void next() {
        if (first) {
            initLeadingRepresentation();
            initTrailingRepresentation();
            first = false;
            
            int trailingZeros = read_int(7);
            if (trailingZeros < 64) {
                uint64_t mantissa = read_long(63 - trailingZeros);
                storedVal.i = ((mantissa << 1) + 1) << trailingZeros;
            } else {
                storedVal.i = 0;
            }
            
            if (isnan(storedVal.d)) {
                endOfStream = true;
            }
        } else {
            nextValue();
        }
    }

    // 完全按照CPU版本的nextValue逻辑
    __device__ __forceinline__ void nextValue() {
        DOUBLE value;
        int centerBits;

        if (read_bit() == 1) {
            // case 1: 重用leading/trailing
            centerBits = 64 - storedLeadingZeros - storedTrailingZeros;
            if (centerBits > 0 && centerBits <= 64) {
                value.i = read_long(centerBits) << storedTrailingZeros;
                value.i = storedVal.i ^ value.i;
                if (isnan(value.d)) {
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
            int trail = ~(0xffffffff << trailingBitsPerValue) & leadAndTrail;

            if (lead >= 0 && lead < leadingRepresentationSize &&
                trail >= 0 && trail < trailingRepresentationSize) {
                
                storedLeadingZeros = leadingRepresentation[lead];
                storedTrailingZeros = trailingRepresentation[trail];
                centerBits = 64 - storedLeadingZeros - storedTrailingZeros;

                if (centerBits > 0 && centerBits <= 64) {
                    value.i = read_long(centerBits) << storedTrailingZeros;
                    value.i = storedVal.i ^ value.i;
                    if (isnan(value.d)) {
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

    __device__ __forceinline__ double readValue() {
        if (endOfStream) {
            return -1;
        }
        
        next();
        
        if (endOfStream) {
            return -1;
        }
        
        return storedVal.d;
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

    __device__ __forceinline__ double nextValue() {
        double v;
        
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

    __device__ __forceinline__ double recoverVByBetaStar() {
        double vPrime = xorDecompressor.readValue();
        
        if (xorDecompressor.isEndOfStream()) {
            return -1;
        }
        
        double v;
        int sp = getSP(fabs(vPrime));
        
        if (lastBetaStar == 0) {
            v = get10iN(-sp - 1);
            if (vPrime < 0) {
                v = -v;
            }
        } else {
            int alpha = lastBetaStar - sp - 1;
            v = roundUp(vPrime, alpha);
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

    __device__ __forceinline__ int decompress(double *output) {
        int len = getLength();
        
        if (len <= 0 || output == nullptr) {
            return 0;
        }
        
        for (int i = 0; i < len; i++) {
            double value = nextValue();
            if (xorDecompressor.isEndOfStream()) {
                return i;
            }
            output[i] = value;
        }
        
        return len;
    }
};

__device__ int decompress_method_final_fix(
    uint8_t *d_in, ssize_t len, double *d_out_chunks, int thread_id) {
    
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

__device__ void decompress_method(uint8_t *d_in, ssize_t len, double *d_out_chunks) {
    decompress_method_final_fix(d_in, len, d_out_chunks, 0);
}

__global__ void decompress_kernel(const uint8_t* d_in_data,
                                const size_t* d_in_offsets,
                                double* d_out_data,
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
    double* p_out_chunk = d_out_data + out_offset_start;

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