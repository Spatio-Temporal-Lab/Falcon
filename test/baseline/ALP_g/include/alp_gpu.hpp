// #pragma once
// #include <cstdint>
// #include <cstddef>
// #include <cassert>
// #include <cmath>
// #include <algorithm>
// #include <numeric>
// #include <stdexcept>
// #include <iostream>
// #include <climits>
// #include <cstdlib>
// #include <vector>
// #include <cuda_runtime.h> 
// #include "alp/encoder.hpp"
// #include "alp/rd.hpp"
// #include "alp/sampler.hpp"
// #include "alp/config.hpp"

// #define CUDA_CHECK(call) \
//     do { \
//         cudaError_t error = call; \
//         if (error != cudaSuccess) { \
//             std::cerr << "[CUDA Error] " << __FILE__ << ":" << __LINE__ \
//                       << " " << cudaGetErrorString(error) << std::endl; \
//         } \
//     } while(0)

// namespace alp_gpu {

// // ALP记录系统 - 添加GPU版本的记录功能
// inline std::vector<std::pair<uint8_t, uint8_t>>& get_gpu_vector_ef_records() {
//     static std::vector<std::pair<uint8_t, uint8_t>> records;
//     return records;
// }

// inline std::vector<std::vector<std::pair<std::pair<int, int>, int>>>& get_gpu_rowgroup_k_combinations() {
//     static std::vector<std::vector<std::pair<std::pair<int, int>, int>>> records;
//     return records;
// }

// inline bool& get_gpu_recording_enabled() {
//     static bool enabled = false;
//     return enabled;
// }

// // GPU记录函数
// inline void record_gpu_vector_ef(uint8_t e, uint8_t f) {
//     if (get_gpu_recording_enabled()) {
//         get_gpu_vector_ef_records().emplace_back(e, f);
//     }
// }

// inline void record_gpu_rowgroup_combinations(const std::vector<std::pair<std::pair<int, int>, int>>& combinations) {
//     if (get_gpu_recording_enabled()) {
//         get_gpu_rowgroup_k_combinations().push_back(combinations);
//     }
// }

// // 控制函数
// inline void enable_gpu_alp_recording(bool enable = true) {
//     get_gpu_recording_enabled() = enable;
// }

// inline void clear_gpu_alp_records() {
//     get_gpu_vector_ef_records().clear();
//     get_gpu_rowgroup_k_combinations().clear();
// }

// // 获取记录数据的访问函数
// inline const std::vector<std::pair<uint8_t, uint8_t>>& get_gpu_vector_ef_records_const() {
//     return get_gpu_vector_ef_records();
// }

// inline const std::vector<std::vector<std::pair<std::pair<int, int>, int>>>& get_gpu_rowgroup_k_combinations_const() {
//     return get_gpu_rowgroup_k_combinations();
// }

// // 使用官方的state结构体和配置
// template<typename T>
// using AlpState = alp::state<T>;

// // 统一使用CPU版本的配置，避免重复定义
// namespace config {
//     // 直接使用CPU版本的配置
//     static constexpr int VECTOR_SIZE = alp::config::VECTOR_SIZE;//1024
//     static constexpr int ROWGROUP_SIZE = alp::config::ROWGROUP_SIZE;//102400
//     static constexpr int ROWGROUP_VECTOR_SAMPLES = alp::config::ROWGROUP_VECTOR_SAMPLES;//8
//     static constexpr int SAMPLES_PER_VECTOR = alp::config::SAMPLES_PER_VECTOR;//32
//     static constexpr int MAX_K_COMBINATIONS = alp::config::MAX_K_COMBINATIONS;//5
//     static constexpr int ROWGROUP_SAMPLES_JUMP = alp::config::ROWGROUP_SAMPLES_JUMP;//(ROWGROUP_SIZE / ROWGROUP_VECTOR_SAMPLES) / VECTOR_SIZE;
//     static constexpr int SAMPLING_EARLY_EXIT_THRESHOLD = alp::SAMPLING_EARLY_EXIT_THRESHOLD;//2
//     static inline constexpr double  MAGIC_NUMBER {0x0018000000000000};
//     static constexpr double  ENCODING_UPPER_LIMIT = 9223372036854774784;
//     static constexpr double  ENCODING_LOWER_LIMIT = -9223372036854774784;
// }

// // 使用CPU版本的常量定义
// template<typename T> struct Constants {
//     static constexpr int EXCEPTION_SIZE = alp::Constants<T>::EXCEPTION_SIZE;
//     static constexpr int EXCEPTION_POSITION_SIZE = alp::EXCEPTION_POSITION_SIZE;
//     static constexpr int MAX_EXPONENT = alp::Constants<T>::MAX_EXPONENT;        
//     static constexpr size_t RD_SIZE_THRESHOLD_LIMIT = alp::Constants<T>::RD_SIZE_THRESHOLD_LIMIT;
// };

// // (e,f)组合及其统计信息
// struct EFCombination {
//     uint8_t e, f;
//     int count;
//     double score;
// };

// // 压缩模式（与CPU版一致）
// enum class CompressionMode : std::uint8_t {
//     ALP  = 0,
//     ALPrd = 1
// };

// // 运行参数
// struct Params {
//     int  vectorSize      = config::VECTOR_SIZE;
//     int  blockSize       = config::ROWGROUP_SIZE;
//     int  threadsPerBlock = 256;
//     bool use_alprd_cutting = true;
//     bool prefer_alprd      = false;
//     bool debug             = false;
//     bool enable_recording  = false;  // 新增：启用记录功能
// };

// // 压缩结果
// struct Compressed {
//     std::vector<std::uint8_t>  data;
//     std::vector<std::uint64_t> offsets;
//     std::vector<std::uint64_t> bit_sizes;
//     std::vector<std::uint32_t> elem_counts;
//     int                        vectorSize = config::VECTOR_SIZE;

//     inline bool   empty() const { return data.empty(); }
//     inline size_t bytes() const { return data.size(); }
//     inline size_t blocks() const { return offsets.size(); }
//     inline size_t total_elems() const {
//         size_t s = 0; for (auto v : elem_counts) s += v; return s;
//     }
// };

// // 设备端压缩结果
// struct CompressedDevice {
//     uint8_t* d_data;
//     size_t data_size;
//     std::vector<uint64_t> offsets;
//     std::vector<uint64_t> bit_sizes;
//     std::vector<uint32_t> elem_counts;
//     int vectorSize;
    
//     ~CompressedDevice() {
//         if(d_data) cudaFree(d_data);
//     }
    
//     CompressedDevice(CompressedDevice&& other) noexcept 
//         : d_data(other.d_data), data_size(other.data_size),
//           offsets(std::move(other.offsets)), bit_sizes(std::move(other.bit_sizes)),
//           elem_counts(std::move(other.elem_counts)), vectorSize(other.vectorSize) {
//         other.d_data = nullptr;
//     }
    
//     CompressedDevice() : d_data(nullptr), data_size(0), vectorSize(config::VECTOR_SIZE) {}
// };

// // API声明
// Compressed compress_double(const double* data, size_t n, const Params& p);
// Compressed compress_float (const float*  data, size_t n, const Params& p);
// void decompress_double(const Compressed& c, double* out, size_t n, const Params& p);
// void decompress_float (const Compressed& c, float*  out, size_t n, const Params& p);

// CompressedDevice compress_double_device(const double* d_data, size_t n, const Params& p, cudaStream_t stream = 0);
// CompressedDevice compress_float_device(const float* d_data, size_t n, const Params& p, cudaStream_t stream = 0);
// void decompress_double_device(const CompressedDevice& c, double* d_out, size_t n, const Params& p, cudaStream_t stream = 0);
// void decompress_float_device(const CompressedDevice& c, float* d_out, size_t n, const Params& p, cudaStream_t stream = 0);

// Compressed device_to_host(const CompressedDevice& cd);
// CompressedDevice host_to_device(const Compressed& c);

// // 修正的常量数组（与CPU版本完全一致）
// __device__ __constant__ double D_EXP_ARR[24] = {
// 	    1.0,
// 	    10.0,
// 	    100.0,
// 	    1000.0,
// 	    10000.0,
// 	    100000.0,
// 	    1000000.0,
// 	    10000000.0,
// 	    100000000.0,
// 	    1000000000.0,
// 	    10000000000.0,
// 	    100000000000.0,
// 	    1000000000000.0,
// 	    10000000000000.0,
// 	    100000000000000.0,
// 	    1000000000000000.0,
// 	    10000000000000000.0,
// 	    100000000000000000.0,
// 	    1000000000000000000.0,
// 	    10000000000000000000.0,
// 	    100000000000000000000.0,
// 	    1000000000000000000000.0,
// 	    10000000000000000000000.0,
// 	    100000000000000000000000.0,
// };

// __device__ __constant__ int64_t D_FACT_ARR[19] = {
//                                         1,
//                                         10,
//                                         100,
//                                         1000,
//                                         10000,
//                                         100000,
//                                         1000000,
//                                         10000000,
//                                         100000000,
//                                         1000000000,
//                                         10000000000,
//                                         100000000000,
//                                         1000000000000,
//                                         10000000000000,
//                                         100000000000000,
//                                         1000000000000000,
//                                         10000000000000000,
//                                         100000000000000000,
//                                         1000000000000000000};

// __device__ __constant__ double D_FRAC_ARR[21] = {
// 	    1.0,
// 	    0.1,
// 	    0.01,
// 	    0.001,
// 	    0.0001,
// 	    0.00001,
// 	    0.000001,
// 	    0.0000001,
// 	    0.00000001,
// 	    0.000000001,
// 	    0.0000000001,
// 	    0.00000000001,
// 	    0.000000000001,
// 	    0.0000000000001,
// 	    0.00000000000001,
// 	    0.000000000000001,
// 	    0.0000000000000001,
// 	    0.00000000000000001,
// 	    0.000000000000000001,
// 	    0.0000000000000000001,
// 	    0.00000000000000000001,
// };

// static constexpr int   DICT_BW  = 3;
// static constexpr int   DICT_SZ  = 1 << DICT_BW;
// static constexpr int   CUT_LIM  = 16;
// static constexpr int   MAX_VEC  = 4096;

// // 安全的设备端位流 Writer/Reader
// struct SafeBitWriter {
//     uint8_t* buf;
//     uint64_t bitpos;
//     uint64_t max_bits;
    
//     __device__ SafeBitWriter(uint8_t* buffer, uint64_t start_bit, uint64_t buffer_size_bits)
//         : buf(buffer), bitpos(start_bit), max_bits(start_bit + buffer_size_bits) {}
    
//     __device__ bool put1(int b){
//         if (bitpos >= max_bits) return false;
//         if (!b) { ++bitpos; return true; }
        
//         uint64_t byte = bitpos >> 3;
//         int off = 7 - int(bitpos & 7ULL);
//         buf[byte] |= (uint8_t(1u) << off);
//         ++bitpos;
//         return true;
//     }
    
//     __device__ bool putN(uint64_t v, int bits){
//         if (bits <= 0 || bits > 64) return false;
//         if (bitpos + bits > max_bits) return false;
//         for(int i=bits-1;i>=0;--i) {
//             if (!put1( (v>>i) & 1ULL )) return false;
//         }
//         return true;
//     }
    
//     __device__ uint64_t get_pos() const { return bitpos; }
//     __device__ uint64_t remaining_bits() const { return max_bits - bitpos; }
// };

// struct BitReader {
//     const uint8_t* buf;
//     uint64_t bitpos;
//     __device__ int get1(){
//         uint64_t byte = bitpos >> 3;
//         int off = 7 - int(bitpos & 7ULL);
//         int b = (buf[byte] >> off) & 1;
//         ++bitpos; return b;
//     }
//     __device__ uint64_t getN(int bits){
//         uint64_t v=0;
//         for(int i=0;i<bits;++i){ v = (v<<1) | get1(); }
//         return v;
//     }
//     __device__ uint64_t get_pos() const { return bitpos; }
// };

// // 工具函数
// __device__ __forceinline__ int width_needed_unsigned(unsigned long long range){
//     if (range==0ULL) return 1;
//     int c=0; while(range){ ++c; range>>=1ULL; } return c;
// }

// __device__ inline long long fast_round_double(double x){
//     if(!std::isfinite(x) || std::isnan(x) || x > config::ENCODING_UPPER_LIMIT || x < config::ENCODING_LOWER_LIMIT ||
// 		       (x == 0.0 && std::signbit(x)))
//     {
//         return config::ENCODING_UPPER_LIMIT;
//     }
//     const double SWEET = config::MAGIC_NUMBER;
//     return (long long)(x + SWEET) - (long long)SWEET;
// }

// __device__ inline uint32_t mask_lo(int bits){
//     return (bits >= 32) ? 0xFFFFFFFFu : ((1u<<bits) - 1u);
// }

// // 前向声明
// template<typename T> __device__ inline bool alp_exact_equal(T v, uint8_t e, uint8_t f);
// template<typename T> __device__ inline void alp_vector_analyze(const T* v, int n, uint8_t e, uint8_t f, short& bitw, long long& FOR, int& exc_cnt);
// template<typename T> __device__ inline void alp_vector_choose_best_bits(const T* v, int n, uint8_t& best_e, uint8_t& best_f, short& bitw, long long& FOR, int& exc);
// template<typename T> __device__ inline uint64_t alp_vector_size_bits_safe(int n, uint8_t e, uint8_t f, short bitw, int exc_cnt);
// template<typename T> __device__ inline bool alp_vector_write_safe(SafeBitWriter& bw, const T* v, int n, uint8_t e, uint8_t f, short bitw, long long FOR);

// // 修正后的ALPrd字典结构，添加actual_dictionary_size字段
// template<typename T> struct ALPrdDict {
//     uint8_t rightBW;
//     uint8_t leftBW;
//     uint32_t dict[DICT_SZ];
//     uint8_t actual_dictionary_size;  // 新增：实际字典大小
// };

// template<typename T> __device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D);
// template<typename T> __device__ inline uint64_t alprd_vector_size_bits_safe(int n, const ALPrdDict<T>& D, int exc_cnt);
// template<typename T> __device__ inline bool alprd_vector_write_safe(SafeBitWriter& bw, const uint64_t* in, int n, const ALPrdDict<T>& D);

// // Kernel声明
// template<typename T>
// __global__ void kernel_rowgroup_sampling(
//     const T* data,
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes,
//     int numBlocks,
//     int vectorSize,
//     uint8_t* out_modes,
//     uint8_t* out_k_actual,
//     uint8_t* out_k_combinations,
//     uint8_t* out_alprd_right_bw,
//     uint8_t* out_alprd_left_bw,
//     uint32_t* out_alprd_dicts
// );

// template<typename T>
// __global__ void kernel_vector_parameter_selection(
//     const T* data,
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes,
//     const uint8_t* modes,
//     const uint8_t* k_actual,
//     const uint8_t* k_combinations,
//     const uint8_t* alprd_right_bw,
//     const uint8_t* alprd_left_bw,
//     const uint32_t* alprd_dicts,
//     int numBlocks,
//     int vectorSize,
//     uint64_t total_vectors,
//     uint8_t* vec_e,
//     uint8_t* vec_f,
//     uint16_t* vec_bitw,
//     int64_t* vec_FOR,
//     uint16_t* vec_exc_cnt,
//     uint64_t* vec_bit_sizes
// );

// template<typename T>
// __global__ void kernel_compress_and_write(
//     const T* data,
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes,
//     const uint8_t* modes,
//     const uint8_t* alprd_right_bw,
//     const uint8_t* alprd_left_bw,
//     const uint32_t* alprd_dicts,
//     const uint64_t* vec_bit_offsets,
//     const uint8_t* vec_e,
//     const uint8_t* vec_f,
//     const uint16_t* vec_bitw,
//     const int64_t* vec_FOR,
//     int numBlocks,
//     int vectorSize,
//     uint8_t* out_bytes
// );

// __global__ void kernel_write_rowgroup_headers(
//     const uint64_t* blk_sizes,
//     const uint64_t* blk_offsets,
//     int numBlocks,
//     int vectorSize,
//     uint8_t* out_bytes
// );

// template<typename T>
// __global__ void kernel_decompress_debug(const uint8_t* bytes,
//                                         const uint64_t* blk_starts_bits,
//                                         const uint64_t* blk_bits,
//                                         const uint64_t* out_starts,
//                                         const int vectorSize,
//                                         T* out_data, 
//                                         int numBlocks);

// } // namespace alp_gpu
#pragma once
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <climits>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h> 
#include "alp/encoder.hpp"
#include "alp/rd.hpp"
#include "alp/sampler.hpp"
#include "alp/config.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "[CUDA Error] " << __FILE__ << ":" << __LINE__ \
                      << " " << cudaGetErrorString(error) << std::endl; \
        } \
    } while(0)

namespace alp_gpu {

// ALP记录系统 - 添加GPU版本的记录功能
inline std::vector<std::pair<uint8_t, uint8_t>>& get_gpu_vector_ef_records() {
    static std::vector<std::pair<uint8_t, uint8_t>> records;
    return records;
}

inline std::vector<std::vector<std::pair<std::pair<int, int>, int>>>& get_gpu_rowgroup_k_combinations() {
    static std::vector<std::vector<std::pair<std::pair<int, int>, int>>> records;
    return records;
}

inline bool& get_gpu_recording_enabled() {
    static bool enabled = false;
    return enabled;
}

// GPU记录函数
inline void record_gpu_vector_ef(uint8_t e, uint8_t f) {
    if (get_gpu_recording_enabled()) {
        get_gpu_vector_ef_records().emplace_back(e, f);
    }
}

inline void record_gpu_rowgroup_combinations(const std::vector<std::pair<std::pair<int, int>, int>>& combinations) {
    if (get_gpu_recording_enabled()) {
        get_gpu_rowgroup_k_combinations().push_back(combinations);
    }
}

// 控制函数
inline void enable_gpu_alp_recording(bool enable = true) {
    get_gpu_recording_enabled() = enable;
}

inline void clear_gpu_alp_records() {
    get_gpu_vector_ef_records().clear();
    get_gpu_rowgroup_k_combinations().clear();
}

// 获取记录数据的访问函数
inline const std::vector<std::pair<uint8_t, uint8_t>>& get_gpu_vector_ef_records_const() {
    return get_gpu_vector_ef_records();
}

inline const std::vector<std::vector<std::pair<std::pair<int, int>, int>>>& get_gpu_rowgroup_k_combinations_const() {
    return get_gpu_rowgroup_k_combinations();
}

// 使用官方的state结构体和配置
template<typename T>
using AlpState = alp::state<T>;

// 统一使用CPU版本的配置，避免重复定义
namespace config {
    // 直接使用CPU版本的配置
    static constexpr int VECTOR_SIZE = alp::config::VECTOR_SIZE;//1024
    static constexpr int ROWGROUP_SIZE = alp::config::ROWGROUP_SIZE;//102400
    static constexpr int ROWGROUP_VECTOR_SAMPLES = alp::config::ROWGROUP_VECTOR_SAMPLES;//8
    static constexpr int SAMPLES_PER_VECTOR = alp::config::SAMPLES_PER_VECTOR;//32
    static constexpr int MAX_K_COMBINATIONS = alp::config::MAX_K_COMBINATIONS;//5
    static constexpr int ROWGROUP_SAMPLES_JUMP = alp::config::ROWGROUP_SAMPLES_JUMP;//(ROWGROUP_SIZE / ROWGROUP_VECTOR_SAMPLES) / VECTOR_SIZE;
    static constexpr int SAMPLING_EARLY_EXIT_THRESHOLD = alp::SAMPLING_EARLY_EXIT_THRESHOLD;//2
    static inline constexpr double  MAGIC_NUMBER_DOUBLE {0x0018000000000000};
    static inline constexpr float   MAGIC_NUMBER_FLOAT = 12582912.0f;
    static constexpr double  ENCODING_UPPER_LIMIT = 9223372036854774784;
    static constexpr double  ENCODING_LOWER_LIMIT = -9223372036854774784;
}

template <class T>
struct Constants {};

// 使用CPU版本的常量定义
template<> struct Constants<float> {
    static constexpr int EXCEPTION_SIZE = alp::Constants<float>::EXCEPTION_SIZE;
    static constexpr int EXCEPTION_POSITION_SIZE = alp::EXCEPTION_POSITION_SIZE;
    static constexpr int MAX_EXPONENT = alp::Constants<float>::MAX_EXPONENT;        
    static constexpr size_t RD_SIZE_THRESHOLD_LIMIT = alp::Constants<float>::RD_SIZE_THRESHOLD_LIMIT;
};
template<> struct Constants<double> {
    static constexpr int EXCEPTION_SIZE = alp::Constants<double>::EXCEPTION_SIZE;
    static constexpr int EXCEPTION_POSITION_SIZE = alp::EXCEPTION_POSITION_SIZE;
    static constexpr int MAX_EXPONENT = alp::Constants<double>::MAX_EXPONENT;        
    static constexpr size_t RD_SIZE_THRESHOLD_LIMIT = alp::Constants<double>::RD_SIZE_THRESHOLD_LIMIT;
};
// (e,f)组合及其统计信息
struct EFCombination {
    uint8_t e, f;
    int count;
    double score;
};

// 压缩模式（与CPU版一致）
enum class CompressionMode : std::uint8_t {
    ALP  = 0,
    ALPrd = 1
};

// 运行参数
struct Params {
    int  vectorSize      = config::VECTOR_SIZE;
    int  blockSize       = config::ROWGROUP_SIZE;
    int  threadsPerBlock = 256;
    bool use_alprd_cutting = true;
    bool prefer_alprd      = false;
    bool debug             = false;
    bool enable_recording  = false;  // 新增：启用记录功能
};

// 压缩结果
struct Compressed {
    std::vector<std::uint8_t>  data;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint64_t> bit_sizes;
    std::vector<std::uint32_t> elem_counts;
    int                        vectorSize = config::VECTOR_SIZE;

    inline bool   empty() const { return data.empty(); }
    inline size_t bytes() const { return data.size(); }
    inline size_t blocks() const { return offsets.size(); }
    inline size_t total_elems() const {
        size_t s = 0; for (auto v : elem_counts) s += v; return s;
    }
};

// 设备端压缩结果
struct CompressedDevice {
    uint8_t* d_data;
    size_t data_size;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> bit_sizes;
    std::vector<uint32_t> elem_counts;
    int vectorSize;
    
    ~CompressedDevice() {
        if(d_data) cudaFree(d_data);
    }
    
    CompressedDevice(CompressedDevice&& other) noexcept 
        : d_data(other.d_data), data_size(other.data_size),
          offsets(std::move(other.offsets)), bit_sizes(std::move(other.bit_sizes)),
          elem_counts(std::move(other.elem_counts)), vectorSize(other.vectorSize) {
        other.d_data = nullptr;
    }
    
    CompressedDevice() : d_data(nullptr), data_size(0), vectorSize(config::VECTOR_SIZE) {}
};

// API声明
Compressed compress_double(const double* data, size_t n, const Params& p);
Compressed compress_float (const float*  data, size_t n, const Params& p);
void decompress_double(const Compressed& c, double* out, size_t n, const Params& p);
void decompress_float (const Compressed& c, float*  out, size_t n, const Params& p);

CompressedDevice compress_double_device(const double* d_data, size_t n, const Params& p, cudaStream_t stream = 0);
CompressedDevice compress_float_device(const float* d_data, size_t n, const Params& p, cudaStream_t stream = 0);
void decompress_double_device(const CompressedDevice& c, double* d_out, size_t n, const Params& p, cudaStream_t stream = 0);
void decompress_float_device(const CompressedDevice& c, float* d_out, size_t n, const Params& p, cudaStream_t stream = 0);

Compressed device_to_host(const CompressedDevice& cd);
CompressedDevice host_to_device(const Compressed& c);

// 修正的常量数组（与CPU版本完全一致）- Double版本
__device__ __constant__ double D_EXP_ARR[24] = {
	    1.0,
	    10.0,
	    100.0,
	    1000.0,
	    10000.0,
	    100000.0,
	    1000000.0,
	    10000000.0,
	    100000000.0,
	    1000000000.0,
	    10000000000.0,
	    100000000000.0,
	    1000000000000.0,
	    10000000000000.0,
	    100000000000000.0,
	    1000000000000000.0,
	    10000000000000000.0,
	    100000000000000000.0,
	    1000000000000000000.0,
	    10000000000000000000.0,
	    100000000000000000000.0,
	    1000000000000000000000.0,
	    10000000000000000000000.0,
	    100000000000000000000000.0,
};

__device__ __constant__ int64_t D_FACT_ARR[19] = {
                                        1,
                                        10,
                                        100,
                                        1000,
                                        10000,
                                        100000,
                                        1000000,
                                        10000000,
                                        100000000,
                                        1000000000,
                                        10000000000,
                                        100000000000,
                                        1000000000000,
                                        10000000000000,
                                        100000000000000,
                                        1000000000000000,
                                        10000000000000000,
                                        100000000000000000,
                                        1000000000000000000};

__device__ __constant__ double D_FRAC_ARR[21] = {
	    1.0,
	    0.1,
	    0.01,
	    0.001,
	    0.0001,
	    0.00001,
	    0.000001,
	    0.0000001,
	    0.00000001,
	    0.000000001,
	    0.0000000001,
	    0.00000000001,
	    0.000000000001,
	    0.0000000000001,
	    0.00000000000001,
	    0.000000000000001,
	    0.0000000000000001,
	    0.00000000000000001,
	    0.000000000000000001,
	    0.0000000000000000001,
	    0.00000000000000000001,
};

// 新增：Float版本的常量数组
__device__ __constant__ float F_EXP_ARR[11] = {
    1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f, 100000.0f, 
    1000000.0f, 10000000.0f, 100000000.0f, 1000000000.0f, 10000000000.0f
};

__device__ __constant__ float F_FRAC_ARR[11] = {
    1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 
    0.000001f, 0.0000001f, 0.00000001f, 0.000000001f, 0.0000000001f
};

__device__ __constant__ int32_t F_FACT_ARR[10] = {
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
};

static constexpr int   DICT_BW  = 3;
static constexpr int   DICT_SZ  = 1 << DICT_BW;
static constexpr int   CUT_LIM  = 16;
static constexpr int   MAX_VEC  = 4096;

// 安全的设备端位流 Writer/Reader
struct SafeBitWriter {
    uint8_t* buf;
    uint64_t bitpos;
    uint64_t max_bits;
    
    __device__ SafeBitWriter(uint8_t* buffer, uint64_t start_bit, uint64_t buffer_size_bits)
        : buf(buffer), bitpos(start_bit), max_bits(start_bit + buffer_size_bits) {}
    
    __device__ bool put1(int b){
        if (bitpos >= max_bits) return false;
        if (!b) { ++bitpos; return true; }
        
        uint64_t byte = bitpos >> 3;
        int off = 7 - int(bitpos & 7ULL);
        buf[byte] |= (uint8_t(1u) << off);
        ++bitpos;
        return true;
    }
    
    __device__ bool putN(uint64_t v, int bits){
        if (bits <= 0 || bits > 64) return false;
        if (bitpos + bits > max_bits) return false;
        for(int i=bits-1;i>=0;--i) {
            if (!put1( (v>>i) & 1ULL )) return false;
        }
        return true;
    }
    
    __device__ uint64_t get_pos() const { return bitpos; }
    __device__ uint64_t remaining_bits() const { return max_bits - bitpos; }
};

struct BitReader {
    const uint8_t* buf;
    uint64_t bitpos;
    __device__ int get1(){
        uint64_t byte = bitpos >> 3;
        int off = 7 - int(bitpos & 7ULL);
        int b = (buf[byte] >> off) & 1;
        ++bitpos; return b;
    }
    __device__ uint64_t getN(int bits){
        uint64_t v=0;
        for(int i=0;i<bits;++i){ v = (v<<1) | get1(); }
        return v;
    }
    __device__ uint64_t get_pos() const { return bitpos; }
};

// 工具函数
__device__ __forceinline__ int width_needed_unsigned(unsigned long long range){
    if (range==0ULL) return 1;
    int c=0; while(range){ ++c; range>>=1ULL; } return c;
}

__device__ inline long long fast_round_double(double x){
    if(!std::isfinite(x) || std::isnan(x) || x > config::ENCODING_UPPER_LIMIT || x < config::ENCODING_LOWER_LIMIT ||
		       (x == 0.0 && std::signbit(x)))
    {
        return config::ENCODING_UPPER_LIMIT;
    }
    const double SWEET = config::MAGIC_NUMBER_DOUBLE;
    return (long long)(x + SWEET) - (long long)SWEET;
}

__device__ inline int fast_round_float(float x){
    if(!std::isfinite(x) || std::isnan(x) || x > (float)config::ENCODING_UPPER_LIMIT || x < (float)config::ENCODING_LOWER_LIMIT ||
		       (x == 0.0f && std::signbit(x)))
    {
        return 2147483647; // INT_MAX instead of casting large double
    }
    const float SWEET = config::MAGIC_NUMBER_FLOAT;
    return (int)(x + SWEET) - (int)SWEET;
}

__device__ inline uint32_t mask_lo(int bits){
    return (bits >= 32) ? 0xFFFFFFFFu : ((1u<<bits) - 1u);
}

// 前向声明
template<typename T> __device__ inline bool alp_exact_equal(T v, uint8_t e, uint8_t f);
template<typename T> __device__ inline void alp_vector_analyze(const T* v, int n, uint8_t e, uint8_t f, short& bitw, long long& FOR, int& exc_cnt);
template<typename T> __device__ inline void alp_vector_choose_best_bits(const T* v, int n, uint8_t& best_e, uint8_t& best_f, short& bitw, long long& FOR, int& exc);
template<typename T> __device__ inline uint64_t alp_vector_size_bits_safe(int n, uint8_t e, uint8_t f, short bitw, int exc_cnt);
template<typename T> __device__ inline bool alp_vector_write_safe(SafeBitWriter& bw, const T* v, int n, uint8_t e, uint8_t f, short bitw, long long FOR);

// 修正后的ALPrd字典结构，添加actual_dictionary_size字段
template<typename T> struct ALPrdDict {
    uint8_t rightBW;
    uint8_t leftBW;
    uint32_t dict[DICT_SZ];
    uint8_t actual_dictionary_size;  // 新增：实际字典大小
};

template<typename T> __device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D);
template<typename T> __device__ inline uint64_t alprd_vector_size_bits_safe(int n, const ALPrdDict<T>& D, int exc_cnt);
template<typename T> __device__ inline bool alprd_vector_write_safe(SafeBitWriter& bw, const uint64_t* in, int n, const ALPrdDict<T>& D);

// Kernel声明
template<typename T>
__global__ void kernel_rowgroup_sampling(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    int numBlocks,
    int vectorSize,
    uint8_t* out_modes,
    uint8_t* out_k_actual,
    uint8_t* out_k_combinations
);

template<typename T>
__global__ void kernel_vector_parameter_selection(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint8_t* modes,
    const uint8_t* k_actual,
    const uint8_t* k_combinations,
    int numBlocks,
    int vectorSize,
    uint64_t total_vectors,
    uint8_t* vec_e,
    uint8_t* vec_f,
    uint16_t* vec_bitw,
    int64_t* vec_FOR,
    uint16_t* vec_exc_cnt,
    uint64_t* vec_bit_sizes
);

template<typename T>
__global__ void kernel_compress_and_write(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint8_t* modes,
    const uint64_t* vec_bit_offsets,
    const uint8_t* vec_e,
    const uint8_t* vec_f,
    const uint16_t* vec_bitw,
    const int64_t* vec_FOR,
    int numBlocks,
    int vectorSize,
    uint8_t* out_bytes
);

__global__ void kernel_write_rowgroup_headers(
    const uint64_t* blk_sizes,
    const uint64_t* blk_offsets,
    int numBlocks,
    int vectorSize,
    uint8_t* out_bytes
);

template<typename T>
__global__ void kernel_decompress_debug(const uint8_t* bytes,
                                        const uint64_t* blk_starts_bits,
                                        const uint64_t* blk_bits,
                                        const uint64_t* out_starts,
                                        const int vectorSize,
                                        T* out_data, 
                                        int numBlocks);

} // namespace alp_gpu