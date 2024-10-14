cmake_minimum_required(VERSION 3.20)

project(cuCompressor-all)

set(CMAKE_CXX_STANDARD 17)

# CUDA 设置
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

# 包含目录
include_directories(include)
include_directories(Serf/src/utils)

# 源文件
set(SRC_CPU
        src/cpu/CDFCompressor.cpp
        src/cpu/CDFDecompressor.cpp
        src/utils/output_bit_stream.cc
)

set(SRC_GPU
        src/gpu/GDFCompressor.cu
        src/gpu/GDFDecompressor.cu
)

# 添加可执行目标
add_executable(test_cpu test/CDF_test.cpp ${SRC_CPU})
add_executable(test_gpu test/GDF_test.cpp ${SRC_GPU})

# 链接 CUDA 库
target_link_libraries(test_gpu ${CUDA_LIBRARIES})