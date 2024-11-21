cmake_minimum_required(VERSION 3.18)

# 项目名称
project(ndzip LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_70,code=sm_70 --expt-relaxed-constexpr")
# 查找 CUDA
find_package(CUDA REQUIRED)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 70 80)  # 支持 Volta 和 Ampere

# 设置源文件目录
set(NDZIP_SRC_DIR ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/src/ndzip)  
set(NDZIP_INC_DIR ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include)  

# 头文件包含路径
include_directories(${NDZIP_INC_DIR})

# 定义库
add_library(ndzip SHARED
    ${NDZIP_INC_DIR}/ndzip/ndzip.hh
    ${NDZIP_INC_DIR}/ndzip/offload.hh
    ${NDZIP_SRC_DIR}/common.hh
    ${NDZIP_SRC_DIR}/common.cc
    ${NDZIP_SRC_DIR}/cpu_codec.inl
    ${NDZIP_SRC_DIR}/cpu_factory.cc
)

add_library(ndzip-cuda SHARED
    ${NDZIP_INC_DIR}/ndzip/cuda.hh
    ${NDZIP_SRC_DIR}/cuda_bits.cuh
    ${NDZIP_SRC_DIR}/cuda_codec.inl
    ${NDZIP_SRC_DIR}/cuda_factory.cu
)

# 添加头文件包含目录
target_include_directories(ndzip PRIVATE ${NDZIP_INC_DIR})
target_include_directories(ndzip-cuda PRIVATE ${NDZIP_INC_DIR})
# 确保 io 路径也被包含
target_include_directories(ndzip PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/src/io)

# 如果需要，添加库之间的依赖
add_dependencies(ndzip-cuda ndzip)

