// #include <gtest/gtest.h>
// #include <fstream>
// #include <sstream>
// #include <filesystem>
// #include <iostream>
// #include <chrono>
// #include <vector>
// #include <stdexcept>
// #include <cassert>
// #include <cstring>
// #include <memory>
// #include <iomanip>

// #include <cstdlib>
// #include <optional>

// #include <boost/program_options.hpp>
// #include <ndzip/offload.hh>
// #include <boost/filesystem.hpp>
// #include "io.hh"

// namespace fs = std::filesystem;  // 加入这一行

// namespace opts = boost::program_options;

// namespace ndzip::detail {

// enum class data_type { t_float, t_double };

// template<typename T>
// void compress_stream(const std::string &in, const std::string &out, const ndzip::extent &size,
//         ndzip::offloader<T> &offloader, const ndzip::detail::io_factory &io) {
//     using compressed_type = ndzip::compressed_type<T>;

//     const auto array_chunk_length = static_cast<size_t>(num_elements(size));
//     const auto array_chunk_size = array_chunk_length * sizeof(T);
//     const auto max_compressed_chunk_length = ndzip::compressed_length_bound<T>(size);
//     const auto max_compressed_chunk_size = max_compressed_chunk_length * sizeof(compressed_type);

//     size_t compressed_length = 0;
//     size_t n_chunks = 0;
//     kernel_duration total_duration{};
//     {
//         auto in_stream = io.create_input_stream(in, array_chunk_size);
//         auto out_stream = io.create_output_stream(out, max_compressed_chunk_size);

//         while (auto *chunk = in_stream->read_exact()) {
//             const auto input_buffer = static_cast<const T *>(chunk);
//             const auto write_buffer = static_cast<compressed_type *>(out_stream->get_write_buffer());
//             kernel_duration chunk_duration;
//             const auto compressed_chunk_length = offloader.compress(input_buffer, size, write_buffer, &chunk_duration);
//             const auto compressed_chunk_size = compressed_chunk_length * sizeof(compressed_type);
//             assert(compressed_chunk_length <= max_compressed_chunk_length);
//             out_stream->commit_chunk(compressed_chunk_size);
//             compressed_length += compressed_chunk_length;
//             total_duration += chunk_duration;
//             ++n_chunks;
//         }
//     }

//     const auto in_file_size = n_chunks * array_chunk_size;
//     const auto compressed_size = compressed_length * sizeof(compressed_type);
//     std::cerr << "raw = " << n_chunks * in_file_size << " bytes";
//     if (n_chunks > 1) { std::cerr << " (" << n_chunks << " chunks à " << array_chunk_size << " bytes)"; }
//     std::cerr << ", compressed = " << compressed_size << " bytes";
//     std::cerr << ", ratio = " << std::fixed << std::setprecision(4)
//               << (static_cast<double>(compressed_size) / in_file_size);
//     std::cerr << ", time = " << std::setprecision(3) << std::fixed
//               << std::chrono::duration_cast<std::chrono::duration<double>>(total_duration).count() << "s\n";
// }

// template<typename T>
// void decompress_stream(const std::string &in, const std::string &out, const ndzip::extent &size,
//         ndzip::offloader<T> &offloader, const ndzip::detail::io_factory &io) {
//     using compressed_type = ndzip::compressed_type<T>;

//     const auto array_chunk_length = static_cast<size_t>(num_elements(size));
//     const auto array_chunk_size = array_chunk_length * sizeof(T);
//     const auto max_compressed_chunk_length = ndzip::compressed_length_bound<T>(size);
//     const auto max_compressed_chunk_size = max_compressed_chunk_length * sizeof(compressed_type);

//     const auto in_stream = io.create_input_stream(in, max_compressed_chunk_size);
//     const auto out_stream = io.create_output_stream(out, array_chunk_size);

//     size_t compressed_bytes_left = 0;
//     for (;;) {
//         const auto [chunk, bytes_in_chunk] = in_stream->read_some(compressed_bytes_left);
//         if (bytes_in_chunk == 0) { break; }

//         const auto chunk_buffer = static_cast<const compressed_type *>(chunk);
//         const auto chunk_buffer_length = bytes_in_chunk / sizeof(compressed_type);  // floor division!
//         const auto output_buffer = static_cast<T *>(out_stream->get_write_buffer());
//         const auto compressed_length = offloader.decompress(chunk_buffer, chunk_buffer_length, output_buffer, size);
//         const auto compressed_size = compressed_length * sizeof(compressed_type);
//         assert(compressed_length <= chunk_buffer_length);
//         out_stream->commit_chunk(array_chunk_size);
//         compressed_bytes_left = bytes_in_chunk - compressed_size;
//     }
// }

// template<typename T>
// void process_stream(bool decompress, const std::string &in, const std::string &out, const ndzip::extent &size,
//         ndzip::offloader<T> &offloader, const ndzip::detail::io_factory &io) {
//     if (decompress) {
//         decompress_stream(in, out, size, offloader, io);
//     } else {
//         compress_stream(in, out, size, offloader, io);
//     }
// }

// template<typename T>
// void process_stream(bool decompress, const ndzip::extent &size, ndzip::target target,
//         std::optional<size_t> num_cpu_threads, const std::string &in, const std::string &out,
//         const ndzip::detail::io_factory &io) {
//     std::unique_ptr<ndzip::offloader<T>> offloader;
//     if (target == ndzip::target::cpu && num_cpu_threads.has_value()) {
//         offloader = ndzip::make_cpu_offloader<T>(size.dimensions(), *num_cpu_threads);
//     } else {
//         offloader = ndzip::make_offloader<T>(target, size.dimensions(), true /* enable_profiling */);
//     }
//     process_stream(decompress, in, out, size, *offloader, io);
// }

// void process_stream(bool decompress, const ndzip::extent &size, ndzip::target target,
//         std::optional<size_t> num_cpu_threads, const data_type &data_type, const std::string &in,
//         const std::string &out, const ndzip::detail::io_factory &io) {
//     switch (data_type) {
//         case detail::data_type::t_float:
//             return process_stream<float>(decompress, size, target, num_cpu_threads, in, out, io);
//         case detail::data_type::t_double:
//             return process_stream<double>(decompress, size, target, num_cpu_threads, in, out, io);
//         default: std::terminate();
//     }
// }

// }  // namespace ndzip::detail
// // 根据平台选择具体的 io_factory 实现类

// #if NDZIP_SUPPORT_MMAP
//     using io_factory = ndzip::detail::mmap_io_factory;
// #else
//     using io_factory = ndzip::detail::stdio_io_factory;
// #endif

// TEST(CompressionTest, CompressAndDecompressCSVFiles) {
//     // 测试文件夹路径
//     std::string input_dir = "/mnt/e/start/gpu/CUDA/cuCompressor/test/data/float";  // 你的文件夹路径
//     std::string output_dir = "/mnt/e/start/gpu/CUDA/cuCompressor/test/output";  // 输出文件夹路径

//     // 确保输出目录存在
//     fs::create_directory(output_dir);

//     // 创建io_factory对象
//     std::unique_ptr<ndzip::detail::io_factory> io_factory;

//     // 选择具体的 io_factory 子类
//     #if NDZIP_SUPPORT_MMAP
//         io_factory = std::make_unique<ndzip::detail::mmap_io_factory>();
//     #else
//         io_factory = std::make_unique<ndzip::detail::stdio_io_factory>();
//     #endif

//     for (const auto &entry : fs::directory_iterator(input_dir)) {
//         if (entry.is_regular_file() && entry.path().extension() == ".csv") {
//             std::string input_file = entry.path().string();
//             std::string output_file = output_dir + "/" + entry.path().stem().string() + ".ndzip";  // 压缩后的输出文件
//             std::string decompressed_file = output_dir + "/" + entry.path().stem().string() + "_decompressed.csv";  // 解压后的文件

//             std::cout << "Processing file: " << input_file << std::endl;

//             // 设置压缩所需的尺寸，这里假设你已经知道文件的尺寸（例如通过读取文件头）
//             ndzip::extent size = {/* 填写你的尺寸信息 */};  // 根据实际情况设置

//             // 创建压缩时使用的 offloader
//             std::unique_ptr<ndzip::offloader<float>> offloader = ndzip::make_offloader<float>(ndzip::target::cpu, size.dimensions(), true);

//             // 压缩文件
//             ndzip::detail::process_stream(false, size, ndzip::target::cpu, std::nullopt, ndzip::detail::data_type::t_float, input_file, output_file, *io_factory);

//             // 解压文件
//             ndzip::detail::process_stream(true, size, ndzip::target::cpu, std::nullopt, ndzip::detail::data_type::t_float, output_file, decompressed_file, *io_factory);

//             // 验证解压后的文件与原始文件内容是否相同
//             std::ifstream original(input_file, std::ios::binary);
//             std::ifstream decompressed(decompressed_file, std::ios::binary);

//             ASSERT_TRUE(original.is_open());
//             ASSERT_TRUE(decompressed.is_open());

//             // 读取文件内容并进行比较
//             std::stringstream original_stream;
//             original_stream << original.rdbuf();
//             std::stringstream decompressed_stream;
//             decompressed_stream << decompressed.rdbuf();

//             ASSERT_EQ(original_stream.str(), decompressed_stream.str()) << "The decompressed file does not match the original.";

//             std::cout << "Successfully processed: " << input_file << std::endl;
//         }
//     }
// }






// int main(int argc, char **argv) {
//     ::testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
