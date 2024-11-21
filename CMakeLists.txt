cmake_minimum_required(VERSION 3.10)

project(cuCompressor-all)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)


# CUDA 设置
# 设置 CUDA 编译器使用 C++17 标准
set(CMAKE_CUDA_STANDARD 17)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

# 包含目录
include_directories(include)
include_directories(src/utils)

# 源文件
set(SRC_CPU
    src/cpu/CDFCompressor.cpp
    src/cpu/CDFDecompressor.cpp
    src/utils/output_bit_stream.cc
    src/utils/input_bit_stream.cc
)

set(SRC_GPU
    src/gpu/GDFCompressor.cu
    src/gpu/GDFDecompressor.cu
    src/utils/output_bit_stream.cc
    src/utils/input_bit_stream.cc
)
# 添加测试模块
add_subdirectory(test)

# 查找 GTest 和 Google Benchmark
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)







