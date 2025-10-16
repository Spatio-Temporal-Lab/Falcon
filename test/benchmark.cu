
#include "alp.hpp"
#include "data/dataset_utils.hpp"  // 包含 dataset_utils.hpp 来获取 get_dynamic_dataset
#include "CDFDecompressor.h"
#include "CDFCompressor.h"         // 包含 CDFCompressor 和 CDFDecompressor 的头文件
#include "GDFCompressor.cuh"
#include "GDFDecompressor.cuh"
#include <benchmark/benchmark.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <sstream>
#include <chrono>

namespace fs = std::filesystem;

// 静态定义 CDF 压缩函数
static void compress_data_cdf(const std::vector<double>& oriData, std::vector<unsigned char>& cmpData) {
    CDFCompressor CDFC;
    CDFC.compress(oriData, cmpData);
}
static void compress_data_gdf(const std::vector<double>& oriData, std::vector<unsigned char>& cmpData) {
    GDFCompressor GDFC;
    GDFC.compress(oriData, cmpData);
}


// 静态定义 CDF 解压缩函数
static void decompress_data_cdf(const std::vector<unsigned char>& cmpData, std::vector<double>& decompressedData) {
    CDFDecompressor CDFD;
    CDFD.decompress(cmpData, decompressedData);
}

// 静态定义 CDF 解压缩函数
static void decompress_data_gdf(const std::vector<unsigned char>& cmpData, std::vector<double>& decompressedData, size_t nbEle) {
    GDFDecompressor GDFD;
    GDFD.decompress(cmpData, decompressedData, nbEle);
}

// 定义 CDF 基准测试函数
void BM_Compression_CDF(benchmark::State& state, const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path, false);
    std::vector<unsigned char> cmpData;
    std::vector<double> decompressedData;

    for (auto _ : state) {
        state.PauseTiming();  // 暂停计时
        cmpData.clear();      // 清除压缩数据
        decompressedData.clear();
        state.ResumeTiming(); // 开始计时

        // 压缩数据
        compress_data_cdf(oriData, cmpData);

        state.PauseTiming(); // 暂停计时

        // 解压缩数据
        decompress_data_cdf(cmpData, decompressedData);
        
        size_t original_size = oriData.size() * sizeof(double);
        size_t compressed_size = cmpData.size() * sizeof(unsigned char);
        double compression_ratio = compressed_size/ static_cast<double>(original_size);

        // 将压缩率添加到基准测试报告中
        state.counters["Compression Ratio"] = benchmark::Counter(compression_ratio);


        state.ResumeTiming(); // 恢复计时
    }
}


// 定义 GDF 基准测试函数
void BM_Compression_GDF(benchmark::State& state, const std::string& file_path) {
    // 读取数据
    std::vector<double> oriData = read_data(file_path, false);
    std::vector<unsigned char> cmpData;
    std::vector<double> decompressedData;
    size_t nbEle = oriData.size();
    for (auto _ : state) {
        state.PauseTiming();  // 暂停计时
        cmpData.clear();      // 清除压缩数据
        decompressedData.clear();
        state.ResumeTiming(); // 开始计时

        // 压缩数据
        compress_data_gdf(oriData, cmpData);

        state.PauseTiming(); // 暂停计时

        // 解压缩数据
        decompress_data_gdf(cmpData, decompressedData,nbEle);
        
        size_t original_size = oriData.size() * sizeof(double);
        size_t compressed_size = cmpData.size() * sizeof(unsigned char);
        double compression_ratio = compressed_size/ static_cast<double>(original_size);

        // 将压缩率添加到基准测试报告中
        state.counters["Compression Ratio"] = benchmark::Counter(compression_ratio);


        state.ResumeTiming(); // 恢复计时
    }
}

// 定义 ALP 编码的基准测试函数
template <typename PT>
void BM_ColumnCompression(benchmark::State& state, const Column& column) {
    using UT = typename alp::inner_t<PT>::ut;
    using ST = typename alp::inner_t<PT>::st;

    // 预先分配缓冲区
    std::unique_ptr<PT[]> input_buf = std::make_unique<PT[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<PT[]> sample_buf = std::make_unique<PT[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<UT[]> right_buf = std::make_unique<UT[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<UT[]> ffor_right_buf = std::make_unique<UT[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<uint16_t[]> unffor_left_buf = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<PT[]> glue_buf = std::make_unique<PT[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<PT[]> exc_buf = std::make_unique<PT[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<PT[]> decoded_buf = std::make_unique<PT[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<ST[]> encoded_buf = std::make_unique<ST[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<ST[]> base_buf = std::make_unique<ST[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<ST[]> ffor_buf = std::make_unique<ST[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<uint16_t[]> pos_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
    std::unique_ptr<uint16_t[]> exc_c_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
    alp::bw_t bit_width {};

    // 打开文件并读取数据
    std::ifstream file(column.csv_file_path, std::ios::in);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + column.csv_file_path);
    }


    // 计时
    for (auto _ : state) {
        state.PauseTiming(); // 暂停计时以读取数据
        double original_size=0,compressed_size=0;
        size_t row_idx = 0;
        std::string val_str;
        double value_to_encode;

        while (row_idx < alp::config::VECTOR_SIZE && file >> val_str) {
            value_to_encode = std::stod(val_str);
            input_buf[row_idx++] = static_cast<PT>(value_to_encode);
        }

        if (row_idx == 0) {
            state.SkipWithError("No valid data read from file.");
            break;
        }

        size_t tuples_count = row_idx;
        alp::state<PT> stt;
        alp::encoder<PT>::init(input_buf.get(), 0, tuples_count, sample_buf.get(), stt);

        state.ResumeTiming(); // 恢复计时以进行压缩和解压

        if (stt.scheme == alp::Scheme::ALP_RD) {
            // 编码过程
            alp::rd_encoder<PT>::init(input_buf.get(), 0, tuples_count, sample_buf.get(), stt);
            alp::rd_encoder<PT>::encode(input_buf.get(), exc_c_arr.get(), pos_arr.get(), exc_c_arr.get(), right_buf.get(), pos_arr.get(), stt);
            ffor::ffor(right_buf.get(), ffor_right_buf.get(), stt.right_bit_width, &stt.right_for_base);

            // 解码过程
            unffor::unffor(ffor_right_buf.get(), ffor_right_buf.get(), stt.right_bit_width, &stt.right_for_base);
            alp::rd_encoder<PT>::decode(glue_buf.get(), ffor_right_buf.get(), unffor_left_buf.get(), exc_c_arr.get(), pos_arr.get(), exc_c_arr.get(), stt);
            original_size += static_cast<double>(tuples_count * sizeof(PT));
            compressed_size += static_cast<double>(tuples_count) * (static_cast<double>(stt.right_bit_width) / 8.0 + static_cast<double>(stt.left_bit_width) / 8.0);

        } else if (stt.scheme == alp::Scheme::ALP) {
            // 编码过程
            alp::encoder<PT>::encode(input_buf.get(), exc_buf.get(), pos_arr.get(), exc_c_arr.get(), encoded_buf.get(), stt);
            alp::encoder<PT>::analyze_ffor(encoded_buf.get(), bit_width, base_buf.get());
            ffor::ffor(encoded_buf.get(), ffor_buf.get(), bit_width, base_buf.get());

            // 解码过程
            generated::falp::fallback::scalar::falp(ffor_buf.get(), decoded_buf.get(), bit_width, base_buf.get(), stt.fac, stt.exp);
            alp::decoder<PT>::patch_exceptions(decoded_buf.get(), exc_buf.get(), pos_arr.get(), exc_c_arr.get());
            original_size += static_cast<double>(tuples_count * sizeof(PT));
            compressed_size += static_cast<double>((tuples_count * bit_width) / 8);
        }

        state.PauseTiming(); // 暂停计时以进行文件读取或其他操作
            
        // Calculate sizes
        
        double compression_ratio = compressed_size / static_cast<double>(original_size);

        // Add compression ratio to benchmark report
        state.counters["Compression Ratio"] = benchmark::Counter(compression_ratio);

        file.clear();        // 清除文件状态标志，准备下一次读取
        file.seekg(0, std::ios::beg);
    }

    file.close();
}

// 注册基准测试
void RegisterCompressionBenchmarks() {
    std::string dir_path = "../../test/data/float";
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();

            benchmark::RegisterBenchmark(
                ("BM_Compression_CDF_" + entry.path().filename().string()).c_str(),
                [file_path](benchmark::State& state) {
                    BM_Compression_CDF(state, file_path);
                })->Unit(benchmark::kMillisecond);

            benchmark::RegisterBenchmark(
                ("BM_Compression_GDF_" + entry.path().filename().string()).c_str(),
                [file_path](benchmark::State& state) {
                    BM_Compression_GDF(state, file_path);
                })->Unit(benchmark::kMillisecond);

            // benchmark::RegisterBenchmark(
            //     ("BM_ColumnCompression_" + entry.path().filename().string()).c_str(),
            //     [file_path](benchmark::State& state) {
            //         Column column;
            //         column.csv_file_path = file_path;
            //         BM_ColumnCompression<double>(state, column);
            //     })->Unit(benchmark::kMillisecond);
        }
    }
}

int main(int argc, char** argv) {
    RegisterCompressionBenchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
