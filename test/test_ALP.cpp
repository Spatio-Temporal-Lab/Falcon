
// #include "alp.hpp"
// #include "gtest/gtest.h"
// #include "data/dataset_utils.hpp"
// #include <filesystem>
// #include <fstream>
// #include <iostream>
// #include <string>
// #include <vector>
// #include <memory>
// #include <chrono>
// #include <sstream>
// #include <iomanip>
// #include <map>
// #include <algorithm>

// #include "CDFCompressor.h"
// #include "CDFDecompressor.h"
// namespace fs = std::filesystem;

// namespace test {
// template <typename T>
// void ALP_ASSERT(T original_val, T decoded_val) {
//     if (original_val == 0.0 && std::signbit(original_val)) {
//         ASSERT_EQ(decoded_val, 0.0);
//         ASSERT_TRUE(std::signbit(decoded_val));
//     } else if (std::isnan(original_val)) {
//         ASSERT_TRUE(std::isnan(decoded_val));
//     } else {
//         ASSERT_EQ(original_val, decoded_val);
//     }
// }
// } // namespace test

// // 文件压缩统计结构
// struct FileCompressionStats {
//     std::string file_path;
//     double total_original_size = 0;
//     double total_compressed_size = 0;
//     std::chrono::duration<double> total_encode_time{0.0};
//     std::chrono::duration<double> total_decode_time{0.0};
//     size_t total_data_points = 0;
//     int batches_processed = 0;
//     std::map<alp::Scheme, int> scheme_usage;
    
//     double get_compression_ratio() const {
//         return total_original_size > 0 ? total_compressed_size / total_original_size : 0.0;
//     }
    
//     double get_encode_throughput_mbps() const {
//         double time_seconds = total_encode_time.count();
//         if (time_seconds > 0) {
//             double mb_per_second = (total_original_size / (1024.0 * 1024.0)) / time_seconds;
//             return mb_per_second;
//         }
//         return 0.0;
//     }
    
//     double get_decode_throughput_mbps() const {
//         double time_seconds = total_decode_time.count();
//         if (time_seconds > 0) {
//             double mb_per_second = (total_original_size / (1024.0 * 1024.0)) / time_seconds;
//             return mb_per_second;
//         }
//         return 0.0;
//     }
// };

// class alp_test : public ::testing::Test {
// public:
//     std::unique_ptr<double[]> input_buf;
//     std::unique_ptr<double[]> sample_buf;
//     std::unique_ptr<double[]> exception_buf;
//     std::unique_ptr<double[]> decoded_buf;
//     std::unique_ptr<double[]> glue_buf;
//     std::unique_ptr<uint16_t[]> rd_exc_arr;
//     std::unique_ptr<uint16_t[]> pos_arr;
//     std::unique_ptr<uint16_t[]> exc_c_arr;
//     std::unique_ptr<int64_t[]> ffor_buf;
//     std::unique_ptr<int64_t[]> unffor_arr;
//     std::unique_ptr<int64_t[]> base_buf;
//     std::unique_ptr<int64_t[]> encoded_buf;
//     std::unique_ptr<uint64_t[]> ffor_right_buf;
//     std::unique_ptr<uint16_t[]> ffor_left_arr;
//     std::unique_ptr<uint64_t[]> right_buf;
//     std::unique_ptr<uint16_t[]> left_arr;
//     std::unique_ptr<uint64_t[]> unffor_right_buf;
//     std::unique_ptr<uint16_t[]> unffor_left_arr;
//     alp::bw_t bit_width {};

//     void SetUp() override {
//         try {
//             input_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
//             sample_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
//             exception_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
//             decoded_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
//             glue_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
//             right_buf = std::make_unique<uint64_t[]>(alp::config::VECTOR_SIZE);
//             ffor_right_buf = std::make_unique<uint64_t[]>(alp::config::VECTOR_SIZE);
//             unffor_right_buf = std::make_unique<uint64_t[]>(alp::config::VECTOR_SIZE);
//             encoded_buf = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
//             base_buf = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
//             ffor_buf = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
//             rd_exc_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
//             pos_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
//             exc_c_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
//             unffor_arr = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
//             left_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
//             ffor_left_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
//             unffor_left_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
//         } catch (const std::bad_alloc& e) {
//             std::cerr << "Memory allocation failed: " << e.what() << std::endl;
//             throw;
//         }
//     }

//     template <typename PT>
//     size_t calculate_compressed_size_alp_rd(const alp::state<PT>& stt, size_t batch_size, uint16_t exceptions_count) {
//         size_t right_compressed_size = (batch_size * stt.right_bit_width + 7) / 8;
//         size_t left_compressed_size = (batch_size * stt.left_bit_width + 7) / 8;
//         size_t exceptions_size = exceptions_count * sizeof(PT);
//         size_t positions_size = exceptions_count * sizeof(uint16_t);
//         size_t metadata_size = sizeof(stt.right_for_base) + sizeof(stt.left_for_base) + 
//                               sizeof(stt.right_bit_width) + sizeof(stt.left_bit_width) +
//                               sizeof(uint16_t); // 异常计数
        
//         return right_compressed_size + left_compressed_size + exceptions_size + positions_size + metadata_size;
//     }

//     template <typename PT>
//     size_t calculate_compressed_size_alp(const alp::state<PT>& stt, size_t batch_size, 
//                                         alp::bw_t bit_width, uint16_t exceptions_count) {
//         size_t encoded_data_size = (batch_size * bit_width + 7) / 8;
//         size_t exceptions_size = exceptions_count * sizeof(PT);
//         size_t positions_size = exceptions_count * sizeof(uint16_t);
//         size_t metadata_size = sizeof(int64_t) + sizeof(bit_width) + 
//                               sizeof(stt.fac) + sizeof(stt.exp) + 
//                               sizeof(uint16_t); // 异常计数
        
//         // printf("bit_width = %d \n", bit_width);
        
//         // printf("encoded_data_size = %d, exceptions_size = %d, positions_size = %d, metadata_size = %d \n", encoded_data_size, exceptions_size,positions_size,metadata_size);
//         return encoded_data_size + exceptions_size + positions_size + metadata_size;
//     }

//     template <typename PT>
//     void compress_data_batch(const std::vector<double>& all_data, size_t start_idx, size_t batch_size, 
//                            FileCompressionStats& file_stats) {
//         using UT = typename alp::inner_t<PT>::ut;
//         using ST = typename alp::inner_t<PT>::st;

//         auto* input_arr = reinterpret_cast<PT*>(input_buf.get());
//         auto* sample_arr = reinterpret_cast<PT*>(sample_buf.get());
//         auto* right_arr = reinterpret_cast<UT*>(right_buf.get());
//         auto* ffor_right_arr = reinterpret_cast<UT*>(ffor_right_buf.get());
//         auto* unffor_right_arr = reinterpret_cast<UT*>(unffor_right_buf.get());
//         auto* glue_arr = reinterpret_cast<PT*>(glue_buf.get());
//         auto* exc_arr = reinterpret_cast<PT*>(exception_buf.get());
//         auto* dec_dbl_arr = reinterpret_cast<PT*>(decoded_buf.get());
//         auto* encoded_arr = reinterpret_cast<ST*>(encoded_buf.get());
//         auto* base_arr = reinterpret_cast<ST*>(base_buf.get());
//         auto* ffor_arr = reinterpret_cast<ST*>(ffor_buf.get());

//         // 复制数据到输入缓冲区
//         for (size_t i = 0; i < batch_size; ++i) {
//             input_arr[i] = static_cast<PT>(all_data[start_idx + i]);
//             // printf("%f, ",all_data[start_idx + i]);
//         }

//         alp::state<PT> stt;
//         size_t rowgroup_offset = 0;

//         auto init_start = std::chrono::high_resolution_clock::now();
//         alp::encoder<PT>::init(input_arr, rowgroup_offset, batch_size, sample_arr, stt);
//         auto init_end = std::chrono::high_resolution_clock::now();

//         // 统计方案使用情况
//         file_stats.scheme_usage[stt.scheme]++;

//         switch (stt.scheme) {
//         case alp::Scheme::ALP_RD: {
//             auto encode_start = std::chrono::high_resolution_clock::now();
            
//             alp::rd_encoder<PT>::init(input_arr, rowgroup_offset, batch_size, sample_arr, stt);
//             alp::rd_encoder<PT>::encode(input_arr, rd_exc_arr.get(), pos_arr.get(), exc_c_arr.get(), right_arr, left_arr.get(), stt);
//             ffor::ffor(right_arr, ffor_right_arr, stt.right_bit_width, &stt.right_for_base);
//             ffor::ffor(left_arr.get(), ffor_left_arr.get(), stt.left_bit_width, &stt.left_for_base);
            
//             auto encode_end = std::chrono::high_resolution_clock::now();

//             auto decode_start = std::chrono::high_resolution_clock::now();
            
//             unffor::unffor(ffor_right_arr, unffor_right_arr, stt.right_bit_width, &stt.right_for_base);
//             unffor::unffor(ffor_left_arr.get(), unffor_left_arr.get(), stt.left_bit_width, &stt.left_for_base);
//             alp::rd_encoder<PT>::decode(glue_arr, unffor_right_arr, unffor_left_arr.get(), rd_exc_arr.get(), pos_arr.get(), exc_c_arr.get(), stt);

//             auto decode_end = std::chrono::high_resolution_clock::now();

//             // 验证结果
//             for (size_t i = 0; i < batch_size; ++i) { 
//                 test::ALP_ASSERT(input_arr[i], glue_arr[i]);
//             }

//             // 计算压缩大小
//             uint16_t exceptions_count = exc_c_arr.get()[0]; // 假设异常计数存储在第一个位置
//             size_t compressed_size = calculate_compressed_size_alp_rd(stt, batch_size, exceptions_count);

//             file_stats.total_encode_time += (encode_end - encode_start);
//             file_stats.total_decode_time += (decode_end - decode_start);
//             file_stats.total_compressed_size += static_cast<double>(compressed_size);

//             break;
//         }
//         case alp::Scheme::ALP: {
//             auto encode_start = std::chrono::high_resolution_clock::now();

//             alp::encoder<PT>::encode(input_arr, exc_arr, pos_arr.get(), exc_c_arr.get(), encoded_arr, stt);
//             alp::encoder<PT>::analyze_ffor(encoded_arr, bit_width, base_arr);
//             ffor::ffor(encoded_arr, ffor_arr, bit_width, base_arr);

//             auto encode_end = std::chrono::high_resolution_clock::now();

//             auto decode_start = std::chrono::high_resolution_clock::now();

//             generated::falp::fallback::scalar::falp(ffor_arr, dec_dbl_arr, bit_width, base_arr, stt.fac, stt.exp);
//             alp::decoder<PT>::patch_exceptions(dec_dbl_arr, exc_arr, pos_arr.get(), exc_c_arr.get());

//             auto decode_end = std::chrono::high_resolution_clock::now();

//             // 验证结果
//             for (size_t i = 0; i < batch_size; ++i) {
//                 test::ALP_ASSERT(input_arr[i], dec_dbl_arr[i]);
//             }

//             // 计算压缩大小
//             uint16_t exceptions_count = exc_c_arr.get()[0]; // 假设异常计数存储在第一个位置
//             size_t compressed_size = calculate_compressed_size_alp(stt, batch_size, bit_width, exceptions_count);
//             size_t osize = batch_size*64;
//             file_stats.total_encode_time += (encode_end - encode_start);
//             file_stats.total_decode_time += (decode_end - decode_start);
//             file_stats.total_compressed_size += static_cast<double>(compressed_size);
//             if(start_idx<3000){

//             size_t total_compressed_size = file_stats.total_compressed_size;
//             // printf("total_compressed_size = %d \n", total_compressed_size);
//             // printf("compressed_size = %d \n", compressed_size);
//             // printf("osize = %d \n", osize);
//             }

//             break;
//         }
//         default:
//             std::cerr << "未知的压缩方案" << std::endl;
//             break;
//         }

//         file_stats.total_original_size += static_cast<double>(batch_size * sizeof(PT));
//         file_stats.total_data_points += batch_size;
//         file_stats.batches_processed++;
//     }

//     template <typename PT>
//     void test_file_data(const std::string& file_path, FileCompressionStats& file_stats) {
//         // 一次性读取整个文件的所有数据
//         std::vector<double> all_data = read_data(file_path, false);
        
//         if (all_data.empty()) {
//             std::cerr << "警告: 文件 " << file_path << " 中没有有效数据" << std::endl;
//             return;
//         }

//         // std::cout << "文件总数据点: " << all_data.size() << std::endl;

//         // 按批次处理数据
//         size_t data_index = 0;
//         size_t all_batch = 0;
//         // for(int i=0;(i<10);i++)
//         // {
//         //     printf("%f, ",all_data[data_index+i]);
//         // }
//         while (data_index < all_data.size()) {

//             size_t batch_size = std::min(static_cast<size_t>(alp::config::VECTOR_SIZE), 
//                                        all_data.size() - data_index);

//             try {
//                 compress_data_batch<PT>(all_data, data_index, batch_size, file_stats);
                
//             } catch (const std::exception& e) {
//                 std::cerr << "压缩批次 " << data_index << " 失败: " << e.what() << std::endl;
//                 throw;
//             }

//             data_index += batch_size;
//             all_batch++;
//         }
//         // printf("all_batch : %d \n", all_batch);

//         // printf("data_index : %d \n", data_index);
//     }
// };

// std::string dir_path = "../test/data/tsbs_csv";

// TEST_F(alp_test, ratio) {
//     // 获取目录中的所有CSV文件
//     std::vector<std::string> csv_files;
    
//     try {
//         for (const auto& entry : fs::directory_iterator(dir_path)) {
//             if (entry.is_regular_file() && entry.path().extension() == ".csv") {
//                 csv_files.push_back(entry.path().string());
//             }
//         }
//     } catch (const std::exception& e) {
//         std::cerr << "读取目录失败: " << e.what() << std::endl;
//         FAIL() << "无法读取数据目录: " << dir_path;
//     }

//     // ASSERT_FALSE(csv_files.empty()) << "在目录 " << dir_path << " 中没有找到CSV文件!";
    
//     // std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;
    
//     std::vector<FileCompressionStats> all_file_stats;
    
//     // 按文件处理
//     for (const auto& file_path : csv_files) {
//         std::cout << "\nProcessing file: " << file_path << std::endl;

        
//         FileCompressionStats file_stats;
//         file_stats.file_path = file_path;
        
//         try {
//             // 进行3次测试取平均值
//             FileCompressionStats total_stats;
//             total_stats.file_path = file_path;
            
//             for (int run = 0; run < 3; run++) {
//                 FileCompressionStats run_stats;
//                 run_stats.file_path = file_path;

                
//                 // std::cout << "第 " << (run + 1) << " 次运行..." << std::endl;
//                 test_file_data<double>(file_path, run_stats);
                
//                 size_t total_compressed_size = run_stats.total_compressed_size;
//                 printf("file_stats : %d \n", total_compressed_size);
//                 // 累加统计信息
//                 total_stats.total_original_size += run_stats.total_original_size;
//                 total_stats.total_compressed_size += run_stats.total_compressed_size;
//                 total_stats.total_encode_time += run_stats.total_encode_time;
//                 total_stats.total_decode_time += run_stats.total_decode_time;
//                 total_stats.total_data_points += run_stats.total_data_points;
//                 total_stats.batches_processed += run_stats.batches_processed;
                
//                 // 合并方案使用统计
//                 for (const auto& [scheme, count] : run_stats.scheme_usage) {
//                     total_stats.scheme_usage[scheme] += count;
//                 }
//             }
            
//             // 计算平均值
//             file_stats.total_original_size = total_stats.total_original_size / 3.0;
//             file_stats.total_compressed_size = total_stats.total_compressed_size / 3.0;
//             file_stats.total_encode_time = total_stats.total_encode_time / 3.0;
//             file_stats.total_decode_time = total_stats.total_decode_time / 3.0;
//             file_stats.total_data_points = total_stats.total_data_points / 3;
//             file_stats.batches_processed = total_stats.batches_processed / 3;
            
//             // 方案使用统计取平均
//             for (const auto& [scheme, count] : total_stats.scheme_usage) {
//                 file_stats.scheme_usage[scheme] = count / 3;
//             }
            
//             // 输出统计结果
//             CompressionInfo compression_info{
//                 file_stats.total_original_size / (1024.0 * 1024.0),
//                 file_stats.total_compressed_size / (1024.0 * 1024.0),
//                 file_stats.get_compression_ratio(),
//                 0,
//                 std::chrono::duration<double, std::milli>(file_stats.total_encode_time).count(),
//                 file_stats.get_encode_throughput_mbps() / 1024.0,
//                 0,
//                 std::chrono::duration<double, std::milli>(file_stats.total_decode_time).count(),
//                 file_stats.get_decode_throughput_mbps() / 1024.0
//             };
            
//             compression_info.print();
            
//             // 输出方案使用统计
//             // std::cout << "压缩方案使用统计:" << std::endl;
//             // for (const auto& [scheme, count] : file_stats.scheme_usage) {
//             //     std::cout << "  方案 " << static_cast<int>(scheme) << ": " << count << " 次" << std::endl;
//             // }
            
//             // std::cout << "处理批次数: " << file_stats.batches_processed << std::endl;
//             // std::cout << "=============================================" << std::endl;
            
//             all_file_stats.push_back(file_stats);
//         std::cout << "========================================" << std::endl;
//         } catch (const std::exception& e) {
//             std::cerr << "处理文件 " << file_path << " 时出错: " << e.what() << std::endl;
//         }
//     }
    
// }

// int main(int argc, char *argv[]) {
//     if (argc >= 3 && std::string(argv[1]) == "--dir") {
//         dir_path = argv[2];
//     }
    
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }



#include "alp.hpp"
#include "gtest/gtest.h"
#include "data/dataset_utils.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cstring>

#include "CDFCompressor.h"
#include "CDFDecompressor.h"
namespace fs = std::filesystem;

namespace test {
template <typename T>
void ALP_ASSERT(T original_val, T decoded_val) {
    if (original_val == 0.0 && std::signbit(original_val)) {
        ASSERT_EQ(decoded_val, 0.0);
        ASSERT_TRUE(std::signbit(decoded_val));
    } else if (std::isnan(original_val)) {
        ASSERT_TRUE(std::isnan(decoded_val));
    } else {
        ASSERT_EQ(original_val, decoded_val);
    }
}
} // namespace test

// 文件压缩统计结构
struct FileCompressionStats {
    std::string file_path;
    double total_original_size = 0;
    double total_compressed_size = 0;
    std::chrono::duration<double> total_encode_time{0.0};
    std::chrono::duration<double> total_decode_time{0.0};
    size_t total_data_points = 0;
    int batches_processed = 0;
    std::map<alp::Scheme, int> scheme_usage;
    
    double get_compression_ratio() const {
        return total_original_size > 0 ? total_compressed_size / total_original_size : 0.0;
    }
    
    double get_encode_throughput_mbps() const {
        double time_seconds = total_encode_time.count();
        if (time_seconds > 0) {
            double mb_per_second = (total_original_size / (1024.0 * 1024.0)) / time_seconds;
            return mb_per_second;
        }
        return 0.0;
    }
    
    double get_decode_throughput_mbps() const {
        double time_seconds = total_decode_time.count();
        if (time_seconds > 0) {
            double mb_per_second = (total_original_size / (1024.0 * 1024.0)) / time_seconds;
            return mb_per_second;
        }
        return 0.0;
    }
};

class alp_test : public ::testing::Test {
public:
    void SetUp() override {
        // 初始化代码，如果需要
    }

    template <typename PT>
    size_t calculate_compressed_size_alp_rd(const alp::state<PT>& stt, size_t vector_size, uint16_t exceptions_count) {
        size_t right_compressed_size = (vector_size * stt.right_bit_width + 7) / 8;
        size_t left_compressed_size = (vector_size * stt.left_bit_width + 7) / 8;
        size_t exceptions_size = exceptions_count * sizeof(uint16_t);
        size_t positions_size = exceptions_count * sizeof(uint16_t);
        size_t metadata_size = sizeof(stt.right_for_base) + sizeof(stt.left_for_base) + 
                              sizeof(stt.right_bit_width) + sizeof(stt.left_bit_width) +
                              sizeof(uint16_t) + stt.actual_dictionary_size_bytes;
        
        return right_compressed_size + left_compressed_size + exceptions_size + positions_size + metadata_size;
    }

    template <typename PT>
    size_t calculate_compressed_size_alp(const alp::state<PT>& stt, size_t vector_size, 
                                        alp::bw_t bit_width, uint16_t exceptions_count) {
        size_t encoded_data_size = (vector_size * bit_width + 7) / 8;
        size_t exceptions_size = exceptions_count * sizeof(PT);
        size_t positions_size = exceptions_count * sizeof(uint16_t);
        size_t metadata_size = sizeof(int64_t) + sizeof(bit_width) + 
                              sizeof(stt.fac) + sizeof(stt.exp) + 
                              sizeof(uint16_t);
        
        return encoded_data_size + exceptions_size + positions_size + metadata_size;
    }

    template <typename PT>
    void process_vector_alp_rd(const PT* vector_data, size_t vector_size, 
                              alp::state<PT>& stt, FileCompressionStats& file_stats) {
        using UT = typename alp::inner_t<PT>::ut;
        
        // 对于不完整的vector，需要填充到VECTOR_SIZE避免越界
        bool is_incomplete_vector = (vector_size < alp::config::VECTOR_SIZE);
        size_t padded_size = alp::config::VECTOR_SIZE;
        
        auto encode_start = std::chrono::high_resolution_clock::now();
        
        // 分配固定大小的缓冲区
        auto padded_input = std::make_unique<PT[]>(padded_size);
        auto right_arr = std::make_unique<UT[]>(padded_size);
        auto left_arr = std::make_unique<uint16_t[]>(padded_size);
        auto rd_exc_arr = std::make_unique<uint16_t[]>(padded_size);
        auto pos_arr = std::make_unique<uint16_t[]>(padded_size);
        auto exc_c_arr = std::make_unique<uint16_t[]>(1);
        auto ffor_right_arr = std::make_unique<UT[]>(padded_size);
        auto ffor_left_arr = std::make_unique<uint16_t[]>(padded_size);
        auto decoded_arr = std::make_unique<PT[]>(padded_size);
        
        // 复制原始数据，对不完整vector进行填充
        std::memcpy(padded_input.get(), vector_data, vector_size * sizeof(PT));
        if (is_incomplete_vector) {
            // 用最后一个有效值填充剩余位置
            PT fill_value = (vector_size > 0) ? vector_data[vector_size - 1] : PT(0.0);
            for (size_t i = vector_size; i < padded_size; ++i) {
                padded_input[i] = fill_value;
            }
            // std::cout << "      填充不完整vector: " << vector_size << " -> " << padded_size << std::endl;
        }
        
        // ALP_RD编码 - 使用rowgroup级别确定的参数
        alp::rd_encoder<PT>::encode(padded_input.get(), rd_exc_arr.get(), pos_arr.get(), 
                                  exc_c_arr.get(), right_arr.get(), left_arr.get(), stt);
        
        // FFOR压缩
        ffor::ffor(right_arr.get(), ffor_right_arr.get(), stt.right_bit_width, &stt.right_for_base);
        ffor::ffor(left_arr.get(), ffor_left_arr.get(), stt.left_bit_width, &stt.left_for_base);
        
        auto encode_end = std::chrono::high_resolution_clock::now();

        // 解码验证
        auto decode_start = std::chrono::high_resolution_clock::now();
        auto unffor_right_arr = std::make_unique<UT[]>(padded_size);
        auto unffor_left_arr = std::make_unique<uint16_t[]>(padded_size);
        
        unffor::unffor(ffor_right_arr.get(), unffor_right_arr.get(), stt.right_bit_width, &stt.right_for_base);
        unffor::unffor(ffor_left_arr.get(), unffor_left_arr.get(), stt.left_bit_width, &stt.left_for_base);
        alp::rd_encoder<PT>::decode(decoded_arr.get(), unffor_right_arr.get(), unffor_left_arr.get(), 
                                  rd_exc_arr.get(), pos_arr.get(), exc_c_arr.get(), stt);
        auto decode_end = std::chrono::high_resolution_clock::now();

        // 只验证原始数据部分
        for (size_t i = 0; i < vector_size; ++i) {
            test::ALP_ASSERT(vector_data[i], decoded_arr[i]);
        }

        // 统计信息只计算实际数据的压缩效果
        file_stats.total_encode_time += (encode_end - encode_start);
        file_stats.total_decode_time += (decode_end - decode_start);
        
        uint16_t exceptions_count = exc_c_arr.get()[0];
        // 对于填充的vector，按原始大小计算压缩率更准确
        size_t compressed_size = calculate_compressed_size_alp_rd(stt, vector_size, exceptions_count);
        file_stats.total_compressed_size += static_cast<double>(compressed_size);
    }

    template <typename PT>
    void process_vector_alp(const PT* vector_data, size_t vector_size, 
                           alp::state<PT>& stt, FileCompressionStats& file_stats) {
        using ST = typename alp::inner_t<PT>::st;
        
        // 对于不完整的vector，需要填充到VECTOR_SIZE避免越界
        bool is_incomplete_vector = (vector_size < alp::config::VECTOR_SIZE);
        size_t padded_size = alp::config::VECTOR_SIZE;
        
        auto encode_start = std::chrono::high_resolution_clock::now();

        // 分配固定大小的缓冲区
        auto padded_input = std::make_unique<PT[]>(padded_size);
        auto exc_arr = std::make_unique<PT[]>(padded_size);
        auto pos_arr = std::make_unique<uint16_t[]>(padded_size);
        auto exc_c_arr = std::make_unique<uint16_t[]>(1);
        auto encoded_arr = std::make_unique<ST[]>(padded_size);
        auto base_arr = std::make_unique<ST[]>(1);
        auto ffor_arr = std::make_unique<ST[]>(padded_size);
        auto decoded_arr = std::make_unique<PT[]>(padded_size);
        
        // 复制原始数据，对不完整vector进行填充
        std::memcpy(padded_input.get(), vector_data, vector_size * sizeof(PT));
        if (is_incomplete_vector) {
            // 用最后一个有效值填充剩余位置，避免引入噪声
            PT fill_value = (vector_size > 0) ? vector_data[vector_size - 1] : PT(0.0);
            for (size_t i = vector_size; i < padded_size; ++i) {
                padded_input[i] = fill_value;
            }
            // std::cout << "      填充不完整vector: " << vector_size << " -> " << padded_size << std::endl;
        }
        
        // 设置向量大小并编码（使用填充后的大小）
        stt.vector_size = padded_size;
        alp::encoder<PT>::encode(padded_input.get(), exc_arr.get(), pos_arr.get(), 
                               exc_c_arr.get(), encoded_arr.get(), stt);
        
        // FFOR压缩
        alp::bw_t bit_width;
        alp::encoder<PT>::analyze_ffor(encoded_arr.get(), bit_width, base_arr.get());
        ffor::ffor(encoded_arr.get(), ffor_arr.get(), bit_width, base_arr.get());

        auto encode_end = std::chrono::high_resolution_clock::now();

        // 解码验证
        auto decode_start = std::chrono::high_resolution_clock::now();
        
        generated::falp::fallback::scalar::falp(ffor_arr.get(), decoded_arr.get(), 
                                               bit_width, base_arr.get(), stt.fac, stt.exp);
        alp::decoder<PT>::patch_exceptions(decoded_arr.get(), exc_arr.get(), pos_arr.get(), exc_c_arr.get());
        auto decode_end = std::chrono::high_resolution_clock::now();

        // 只验证原始数据部分
        for (size_t i = 0; i < vector_size; ++i) {
            test::ALP_ASSERT(vector_data[i], decoded_arr[i]);
        }

        // 统计信息只计算实际数据的压缩效果
        file_stats.total_encode_time += (encode_end - encode_start);
        file_stats.total_decode_time += (decode_end - decode_start);
        
        uint16_t exceptions_count = exc_c_arr.get()[0];
        // 对于填充的vector，按原始大小计算压缩率更准确
        size_t compressed_size = calculate_compressed_size_alp(stt, vector_size, bit_width, exceptions_count);
        file_stats.total_compressed_size += static_cast<double>(compressed_size);
    }

    template <typename PT>
    void compress_rowgroup(const std::vector<double>& all_data, size_t start_idx, 
                          size_t rowgroup_size, FileCompressionStats& file_stats) {
        
        // std::cout << "处理Rowgroup: 起始索引=" << start_idx 
        //           << ", 大小=" << rowgroup_size << " 个数据点" << std::endl;
        
        // 分配rowgroup级别的数据缓冲区
        auto rowgroup_data = std::make_unique<PT[]>(rowgroup_size);
        for (size_t i = 0; i < rowgroup_size; ++i) {
            rowgroup_data[i] = static_cast<PT>(all_data[start_idx + i]);
        }

        // 第一阶段：在整个rowgroup上进行采样和参数选择
        alp::state<PT> stt;
        auto sample_buf = std::make_unique<PT[]>(rowgroup_size); // 用于采样的临时缓冲区
        
        auto init_start = std::chrono::high_resolution_clock::now();
        
        // 在整个rowgroup上进行初始化和采样
        alp::encoder<PT>::init(rowgroup_data.get(), 0, rowgroup_size, sample_buf.get(), stt);
        
        // 对于ALP_RD，需要在rowgroup级别构建字典
        if (stt.scheme == alp::Scheme::ALP_RD) {
            alp::rd_encoder<PT>::init(rowgroup_data.get(), 0, rowgroup_size, sample_buf.get(), stt);
        }
        
        auto init_end = std::chrono::high_resolution_clock::now();
        
        // std::cout << "  选择压缩方案: " << 
        //     (stt.scheme == alp::Scheme::ALP ? "ALP" : "ALP_RD") << std::endl;
        // std::cout << "  采样和初始化时间: " << 
        //     std::chrono::duration<double, std::milli>(init_end - init_start).count() << " ms" << std::endl;

        file_stats.scheme_usage[stt.scheme]++;

        // 第二阶段：将rowgroup分解为多个vector进行编码
        size_t processed = 0;
        size_t vector_count = 0;
        
        while (processed < rowgroup_size) {
            size_t current_vector_size = std::min(static_cast<size_t>(alp::config::VECTOR_SIZE),
                                                rowgroup_size - processed);
            
            PT* current_vector = rowgroup_data.get() + processed;
            
            // std::cout << "    处理Vector " << vector_count << ": " 
            //           << current_vector_size << " 个数据点" << std::endl;
            
            // 根据scheme处理当前vector
            switch (stt.scheme) {
            case alp::Scheme::ALP_RD:
                process_vector_alp_rd<PT>(current_vector, current_vector_size, stt, file_stats);
                break;
            case alp::Scheme::ALP:
                process_vector_alp<PT>(current_vector, current_vector_size, stt, file_stats);
                break;
            default:
                throw std::runtime_error("未知的压缩方案");
            }
            
            file_stats.total_original_size += static_cast<double>(current_vector_size * sizeof(PT));
            file_stats.total_data_points += current_vector_size;
            
            processed += current_vector_size;
            vector_count++;
        }
        
        // std::cout << "  完成处理 " << vector_count << " 个vectors" << std::endl;
        file_stats.batches_processed++;
    }

    template <typename PT>
    void test_file_data(const std::string& file_path, FileCompressionStats& file_stats) {
        std::vector<double> all_data = read_data(file_path, false);
        
        if (all_data.empty()) {
            std::cerr << "警告: 文件 " << file_path << " 中没有有效数据" << std::endl;
            return;
        }

        std::cout << "文件总数据量: " << all_data.size() << " 个数据点" << std::endl;
        std::cout << "预计需要处理 " << 
            (all_data.size() + alp::config::ROWGROUP_SIZE - 1) / alp::config::ROWGROUP_SIZE 
            << " 个rowgroups" << std::endl;

        // 按完整的rowgroup进行处理
        size_t data_index = 0;
        size_t rowgroup_count = 0;
        
        while (data_index < all_data.size()) {
            size_t rowgroup_size = std::min(static_cast<size_t>(alp::config::ROWGROUP_SIZE),
                                          all_data.size() - data_index);
            
            // std::cout << "\n=== 处理第 " << rowgroup_count + 1 << " 个Rowgroup ===" << std::endl;
            
            try {
                compress_rowgroup<PT>(all_data, data_index, rowgroup_size, file_stats);
            } catch (const std::exception& e) {
                std::cerr << "处理rowgroup " << rowgroup_count << " 失败: " << e.what() << std::endl;
                throw;
            }

            data_index += rowgroup_size;
            rowgroup_count++;
        }
        
        std::cout << "\n文件处理完成，共处理了 " << rowgroup_count << " 个rowgroups" << std::endl;
    }
};

std::string dir_path = "../test/data/tsbs_csv";

TEST_F(alp_test, ratio) {
    std::vector<std::string> csv_files;
    
    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".csv") {
                csv_files.push_back(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "读取目录失败: " << e.what() << std::endl;
        FAIL() << "无法读取数据目录: " << dir_path;
    }

    std::cout << "找到 " << csv_files.size() << " 个CSV文件" << std::endl;
    
    std::vector<FileCompressionStats> all_file_stats;
    
    for (const auto& file_path : csv_files) {
        std::cout << "正在处理文件: " << file_path << std::endl;
        
        FileCompressionStats file_stats;
        file_stats.file_path = file_path;
        
        try {
            // 只运行一次，避免重复输出干扰分析
            test_file_data<double>(file_path, file_stats);
            
            CompressionInfo compression_info{
                file_stats.total_original_size / (1024.0 * 1024.0),
                file_stats.total_compressed_size / (1024.0 * 1024.0),
                file_stats.get_compression_ratio(),
                0,
                std::chrono::duration<double, std::milli>(file_stats.total_encode_time).count(),
                file_stats.get_encode_throughput_mbps() / 1024.0,
                0,
                std::chrono::duration<double, std::milli>(file_stats.total_decode_time).count(),
                file_stats.get_decode_throughput_mbps() / 1024.0
            };
            
            compression_info.print();
            
            std::cout << "\n压缩方案使用统计:" << std::endl;
            for (const auto& [scheme, count] : file_stats.scheme_usage) {
                std::string scheme_name = (scheme == alp::Scheme::ALP) ? "ALP" : "ALP_RD";
                std::cout << "  " << scheme_name << ": " << count << " 个rowgroups" << std::endl;
            }
            
            std::cout << "总处理的rowgroups: " << file_stats.batches_processed << std::endl;
            std::cout << "平均每个rowgroup的数据点: " << 
                (file_stats.batches_processed > 0 ? file_stats.total_data_points / file_stats.batches_processed : 0) 
                << std::endl;
            
            all_file_stats.push_back(file_stats);
            
        } catch (const std::exception& e) {
            std::cerr << "处理文件 " << file_path << " 时出错: " << e.what() << std::endl;
            continue; // 继续处理下一个文件
        }
    }
    
    // 输出总体统计信息
    if (!all_file_stats.empty()) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "总体统计信息" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        double total_original = 0, total_compressed = 0;
        std::chrono::duration<double> total_encode_time{0.0}, total_decode_time{0.0};
        std::map<alp::Scheme, int> total_scheme_usage;
        size_t total_rowgroups = 0;
        
        for (const auto& stats : all_file_stats) {
            total_original += stats.total_original_size;
            total_compressed += stats.total_compressed_size;
            total_encode_time += stats.total_encode_time;
            total_decode_time += stats.total_decode_time;
            total_rowgroups += stats.batches_processed;
            
            for (const auto& [scheme, count] : stats.scheme_usage) {
                total_scheme_usage[scheme] += count;
            }
        }
        
        std::cout << "总原始大小: " << std::fixed << std::setprecision(2) 
                  << total_original / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "总压缩大小: " << std::fixed << std::setprecision(2) 
                  << total_compressed / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "总压缩比: " << std::fixed << std::setprecision(4) 
                  << (total_original > 0 ? total_compressed / total_original : 0.0) << std::endl;
        std::cout << "总编码时间: " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double, std::milli>(total_encode_time).count() << " ms" << std::endl;
        std::cout << "总解码时间: " << std::fixed << std::setprecision(2)
                  << std::chrono::duration<double, std::milli>(total_decode_time).count() << " ms" << std::endl;
        
        std::cout << "\n总体方案使用统计:" << std::endl;
        for (const auto& [scheme, count] : total_scheme_usage) {
            std::string scheme_name = (scheme == alp::Scheme::ALP) ? "ALP" : "ALP_RD";
            std::cout << "  " << scheme_name << ": " << count << " 个rowgroups" << std::endl;
        }
        
        std::cout << "总处理的rowgroups: " << total_rowgroups << std::endl;
        std::cout << "处理的文件数: " << all_file_stats.size() << std::endl;
    }
}

int main(int argc, char *argv[]) {
    if (argc >= 3 && std::string(argv[1]) == "--dir") {
        dir_path = argv[2];
    }
    
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}