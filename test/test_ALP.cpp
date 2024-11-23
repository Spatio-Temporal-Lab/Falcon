#include "alp.hpp"
#include "gtest/gtest.h"
#include "data/dataset_utils.hpp"  // 包含 dataset_utils.hpp 来获取 get_dynamic_dataset
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>  // 用于计时

#include "CDFCompressor.h"
#include "CDFDecompressor.h"
namespace fs = std::filesystem;

double ALP_compression_ratio=0;
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

// 测试类定义
class alp_test : public ::testing::Test {
public:
    std::unique_ptr<double[]> input_buf;
    std::unique_ptr<double[]> sample_buf;
    std::unique_ptr<double[]> exception_buf;
    std::unique_ptr<double[]> decoded_buf;
    std::unique_ptr<double[]> glue_buf;
    std::unique_ptr<uint16_t[]> rd_exc_arr;
    std::unique_ptr<uint16_t[]> pos_arr;
    std::unique_ptr<uint16_t[]> exc_c_arr;
    std::unique_ptr<int64_t[]> ffor_buf;
    std::unique_ptr<int64_t[]> unffor_arr;
    std::unique_ptr<int64_t[]> base_buf;
    std::unique_ptr<int64_t[]> encoded_buf;
    std::unique_ptr<uint64_t[]> ffor_right_buf;
    std::unique_ptr<uint16_t[]> ffor_left_arr;
    std::unique_ptr<uint64_t[]> right_buf;
    std::unique_ptr<uint16_t[]> left_arr;
    std::unique_ptr<uint64_t[]> unffor_right_buf;
    std::unique_ptr<uint16_t[]> unffor_left_arr;
    alp::bw_t bit_width {};

    void SetUp() override {
        try {
            input_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
            sample_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
            exception_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
            decoded_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
            glue_buf = std::make_unique<double[]>(alp::config::VECTOR_SIZE);
            right_buf = std::make_unique<uint64_t[]>(alp::config::VECTOR_SIZE);
            ffor_right_buf = std::make_unique<uint64_t[]>(alp::config::VECTOR_SIZE);
            unffor_right_buf = std::make_unique<uint64_t[]>(alp::config::VECTOR_SIZE);
            encoded_buf = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
            base_buf = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
            ffor_buf = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
            rd_exc_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
            pos_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
            exc_c_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
            unffor_arr = std::make_unique<int64_t[]>(alp::config::VECTOR_SIZE);
            left_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
            ffor_left_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
            unffor_left_arr = std::make_unique<uint16_t[]>(alp::config::VECTOR_SIZE);
        } catch (const std::bad_alloc& e) {
            std::cerr << "Memory allocation failed: " << e.what() << std::endl;
            throw;
        }
    }

    template <typename PT>
    void test_column(const Column& column) {
        using UT = typename alp::inner_t<PT>::ut;
        using ST = typename alp::inner_t<PT>::st;

        auto* input_arr = reinterpret_cast<PT*>(input_buf.get());
        auto* sample_arr = reinterpret_cast<PT*>(sample_buf.get());
        auto* right_arr = reinterpret_cast<UT*>(right_buf.get());
        auto* ffor_right_arr = reinterpret_cast<UT*>(ffor_right_buf.get());
        auto* unffor_right_arr = reinterpret_cast<UT*>(unffor_right_buf.get());
        auto* glue_arr = reinterpret_cast<PT*>(glue_buf.get());
        auto* exc_arr = reinterpret_cast<PT*>(exception_buf.get());
        auto* dec_dbl_arr = reinterpret_cast<PT*>(decoded_buf.get());
        auto* encoded_arr = reinterpret_cast<ST*>(encoded_buf.get());
        auto* base_arr = reinterpret_cast<ST*>(base_buf.get());
        auto* ffor_arr = reinterpret_cast<ST*>(ffor_buf.get());

        std::ifstream file(column.csv_file_path, std::ios::in);
        if (!file) {
            throw std::runtime_error(column.csv_file_path + " : " + strerror(errno));
        }

        alp::state<PT> stt;
        size_t rowgroup_offset {0};
        double value_to_encode;
        std::string val_str;

        size_t total_processed {0};

        std::chrono::duration<double> encode_t{0.0}, decode_t{0.0};

        double original_size=0,compressed_size=0;

        while (file) {
            size_t row_idx = 0;

            // 读取一个批次的数据
            while (row_idx < alp::config::VECTOR_SIZE && file >> val_str) {
                if constexpr (std::is_same_v<PT, double>) {
                    value_to_encode = std::stod(val_str);
                } else {
                    value_to_encode = std::stof(val_str);
                }
                input_arr[row_idx++] = value_to_encode;
            }

            if (row_idx == 0) {
                break; // 没有更多的数据可读取
            }

            size_t tuples_count = row_idx;

            // Init
            alp::encoder<PT>::init(input_arr, rowgroup_offset, tuples_count, sample_arr, stt);

            switch (stt.scheme) {
            case alp::Scheme::ALP_RD: {
                alp::rd_encoder<PT>::init(input_arr, rowgroup_offset, tuples_count, sample_arr, stt);

           
                auto encode_start = std::chrono::high_resolution_clock::now();//压缩开始
                
                alp::rd_encoder<PT>::encode(input_arr, rd_exc_arr.get(), pos_arr.get(), exc_c_arr.get(), right_arr, left_arr.get(), stt);
                ffor::ffor(right_arr, ffor_right_arr, stt.right_bit_width, &stt.right_for_base);
                ffor::ffor(left_arr.get(), ffor_left_arr.get(), stt.left_bit_width, &stt.left_for_base);
                
                auto encode_end = std::chrono::high_resolution_clock::now();//压缩结束

                // Decode
                auto decode_start = std::chrono::high_resolution_clock::now();//解压缩开始
                unffor::unffor(ffor_right_arr, unffor_right_arr, stt.right_bit_width, &stt.right_for_base);
                unffor::unffor(ffor_left_arr.get(), unffor_left_arr.get(), stt.left_bit_width, &stt.left_for_base);
                alp::rd_encoder<PT>::decode(
                    glue_arr, unffor_right_arr, unffor_left_arr.get(), rd_exc_arr.get(), pos_arr.get(), exc_c_arr.get(), stt);

                auto decode_end = std::chrono::high_resolution_clock::now();//解压缩结束

                for (size_t i = 0; i < tuples_count; ++i) {
                    auto l = input_arr[i];
                    auto r = glue_arr[i];
                    if (l != r) {
                        std::cout << i << " | " << total_processed + i << " r : " << r << " l : " << l << '\n';
                    }
                    test::ALP_ASSERT(r, l);
                }

                encode_t += encode_end - encode_start;
                decode_t += decode_end - decode_start;
                
                //std::cout << "Encoding Time: " << encode_duration.count() << " seconds." << std::endl;
                //std::cout << "Decoding Time: " << decode_duration.count() << " seconds." << std::endl;


                // Calculating compression ratio
                original_size += static_cast<double>(tuples_count * sizeof(PT));
                compressed_size += static_cast<double>(tuples_count) * (static_cast<double>(stt.right_bit_width) / 8.0 + static_cast<double>(stt.left_bit_width) / 8.0);

                if (compressed_size == 0) {
                    std::cerr << "Warning: Compressed size is zero, possibly due to zero bit width." << std::endl;
                    //compressed_size = 1; // 防止压缩率计算时除以零
                }

                break;
            }
            case alp::Scheme::ALP: {
                // Encode               
                auto encode_start = std::chrono::high_resolution_clock::now();//压缩开始

                alp::encoder<PT>::encode(input_arr, exc_arr, pos_arr.get(), exc_c_arr.get(), encoded_arr, stt);
                alp::encoder<PT>::analyze_ffor(encoded_arr, bit_width, base_arr);
                ffor::ffor(encoded_arr, ffor_arr, bit_width, base_arr);
 
                auto encode_end = std::chrono::high_resolution_clock::now();//压缩结束


                // Decode

                auto decode_start = std::chrono::high_resolution_clock::now();//解压缩开始

                generated::falp::fallback::scalar::falp(ffor_arr, dec_dbl_arr, bit_width, base_arr, stt.fac, stt.exp);
                alp::decoder<PT>::patch_exceptions(dec_dbl_arr, exc_arr, pos_arr.get(), exc_c_arr.get());

                auto decode_end = std::chrono::high_resolution_clock::now();//解压缩结束
                // validation
                for (size_t i = 0; i < tuples_count; ++i) {
                    test::ALP_ASSERT(input_arr[i], dec_dbl_arr[i]);
                }


                encode_t += encode_end - encode_start;
                decode_t += decode_end - decode_start;

                // Calculating compression ratio
                original_size += static_cast<double>(tuples_count * sizeof(PT));
                compressed_size += static_cast<double>((tuples_count * bit_width) / 8);
                if (compressed_size == 0) {
                    std::cerr << "Warning: Compressed size is zero, possibly due to zero bit width." << std::endl;
                    //compressed_size = 1; // 防止压缩率计算时除以零
                }

                break;
            }
            default:
                break;
            }

            total_processed += tuples_count;
        }
        //std::cout << "Encoding Time: " << encode_t.count() << " seconds." << std::endl;
        //std::cout << "Decoding Time: " << decode_t.count() << " seconds." << std::endl;
        ALP_compression_ratio =  compressed_size / original_size;
        std::cout << "ALP Compression Ratio: " << ALP_compression_ratio << std::endl;

        //std::cout << "\033[32m-- " << column.name << '\n';
        file.close();
    }
};

// 测试函数
TEST_F(alp_test, test_alp_double) {
    auto dataset = get_dynamic_dataset("/mnt/e/start/gpu/CUDA/ALP_learn/tests/data/float");
    ASSERT_FALSE(dataset.empty()) << "Dataset is empty, check data directory!";

    for (const auto& col : dataset) {
        std::cout << "Testing file: " << col.csv_file_path << std::endl;
        ASSERT_NO_THROW(test_column<double>(col));
    }
}

double compression_ratio=0;
// 测试压缩和解压缩
void test_compression(const std::string& file_path) {
    std::vector<double> oriData = read_data(file_path);
    std::vector<unsigned char> cmpData;
    size_t nbEle = oriData.size();
    //进行压缩
    CDFCompressor CDFC;
    CDFC.compress(oriData,cmpData);
    // 解压缩相关变量
    std::vector<double> decompressedData;

    //进行解压
    CDFDecompressor CDFD;
    CDFD.decompress(cmpData,decompressedData);

    // 计算压缩率
    size_t original_size = oriData.size() * sizeof(double);
    size_t compressed_size = cmpData.size() * sizeof(unsigned char);
    compression_ratio = compressed_size/ static_cast<double>(original_size);

    // 打印压缩率
    std::cout << "压缩率: " << compression_ratio << std::endl;
    ASSERT_EQ(decompressedData.size() , oriData.size()) << "解压失败，数据不一致。";
    for(int i=0;i<oriData[i];i++)
    {
        // 验证解压结果是否与原始数据一致
        // std::cout << std::fixed << std::setprecision(10)<<decompressedData[i]<<" "<<oriData[i]<<std::endl;
        ASSERT_EQ(decompressedData[i] , oriData[i]) <<i<< "解压失败，数据不一致。";

    }

}
TEST_F(alp_test,ratio){
    std::string dir_path = "/mnt/e/start/gpu/CUDA/cuCompressor/test/data/float";//有毛病还没有数据集
    auto dataset = get_dynamic_dataset(dir_path);
    ASSERT_FALSE(dataset.empty()) << "Dataset is empty, check data directory!";
    int i=0;
    int ans=0;
    for (const auto& entry : fs::directory_iterator(dir_path)) {
        
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::cout << "正在处理文件: " << std::endl;//file_path << std::endl;
            test_compression(file_path);
            ASSERT_NO_THROW(test_column<double>(dataset[i]));
            std::cout << "---------------------------------------------" << std::endl;
            if(ALP_compression_ratio<compression_ratio)
            {
                ans++;
            }
        }
        i++;
    }
    std::cout<<"\n\n "<<i<<" : "<< ans<<" \n\n";


}





