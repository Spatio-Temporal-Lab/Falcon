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
set(CMAKE_CUDA_ARCHITECTURES 75 89 86 90)
#enable_language(CXX)
#enable_language(CUDA)
# 包含目录
include_directories(include)
include_directories(src/utils)

set(SRC_GPU
    
    ${CMAKE_SOURCE_DIR}/src/gpu/FalconCompressor_1024.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/FalconDecompressor_1024.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_pipeline.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_pipeline.cu
    src/utils/output_bit_stream.cc
    src/utils/input_bit_stream.cc
)

# 添加测试模块
add_subdirectory(test)


# 查找 GTest 和 Google Benchmark
find_package(GTest REQUIRED)
# find_package(benchmark REQUIRED)


# 添加库不存在的下载方法
if (NOT GTest_FOUND)
    message(STATUS "GTest not found. Downloading and building...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.11.0  # 目前我的版本
    )
    FetchContent_MakeAvailable(googletest)
endif()


# if (NOT benchmark_FOUND)
#     message(STATUS "Google Benchmark library not found. Downloading and building it locally...")
    
#     # 设置下载目录
#     include(FetchContent)
#     FetchContent_Declare(
#         benchmark
#         GIT_REPOSITORY https://github.com/google/benchmark.git
#         GIT_TAG v1.6.1 # 目前我的版本
#     )
#     FetchContent_MakeAvailable(benchmark)
# endif()







