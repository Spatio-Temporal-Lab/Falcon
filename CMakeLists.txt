cmake_minimum_required(VERSION 3.20)

project(cuCompressor-all)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)

# CUDA 设置
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

# 包含目录
include_directories(include)
include_directories(src/utils)
#serf文件现在还是空的
# include_directories(Serf/src/utils)

# 源文件
set(SRC_CPU
        src/cpu/CDFCompressor.cpp
        src/cpu/CDFDecompressor.cpp
        src/utils/output_bit_stream.cc
        src/utils/input_bit_stream.cc
        )

#先不测试
set(SRC_GPU
        #src/gpu/GDFCompressor.cu
        #src/gpu/GDFDecompressor.cu
)

# 添加可执行目标
#add_executable(test_cpu test/CDF_test.cpp ${SRC_CPU})
#先不测试
# add_executable(test_gpu test/GDF_test.cpp ${SRC_GPU})

# 链接 CUDA 库
# target_link_libraries(test_gpu ${CUDA_LIBRARIES})

# 链接googletest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(test_cpu test/CDF_test.cpp ${SRC_CPU})

target_link_libraries(test_cpu ${GTEST_LIBRARIES} gtest_main pthread)

# cmake_minimum_required(VERSION 3.20)

# project(cuCompressor-all)

# set(CMAKE_CXX_STANDARD 17)

# # CUDA 设置
# find_package(CUDA REQUIRED)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3)

# # 包含目录
# include_directories(include)
# include_directories(src/utils)

# # 源文件
# set(SRC_CPU
#         src/cpu/CDFCompressor.cpp
#         src/utils/output_bit_stream.cc
# )

# # 添加可执行目标
# add_executable(test_cpu test/CDF_test.cpp ${SRC_CPU})

# # 链接 googletest
# find_package(GTest REQUIRED)
# include_directories(${GTEST_INCLUDE_DIRS})
# target_link_libraries(test_cpu ${GTEST_LIBRARIES} gtest_main pthread)

# # 如果使用 CUDA，可以链接 GPU 源文件
# set(SRC_GPU
#         #src/gpu/GDFCompressor.cu
#         #src/gpu/GDFDecompressor.cu
# )
