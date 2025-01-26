cmake_minimum_required(VERSION 3.11)

project(CompressionTests)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 添加 ALP 库
add_subdirectory(baseline/ALP)
# 添加 ndzip 项目路径
add_subdirectory(baseline/ndzip)


# 添加 ndzip 库的路径
# add_subdirectory(baseline/ndzip)

# 查找 GTest
find_package(GTest REQUIRED)

# 查找 Google Benchmark 库
find_package(benchmark REQUIRED)


find_package(Boost REQUIRED COMPONENTS program_options)

# 查找 CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Include Directories: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 70)

# 包含头文件路径
include_directories(${CMAKE_SOURCE_DIR}/test/baseline/ALP/include)
include_directories(${CMAKE_SOURCE_DIR}/test/data)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/utils)
include_directories(${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include)  # 添加该路径

# 源文件
set(SRC_CPU
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFCompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFDecompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/output_bit_stream.cc
    ${CMAKE_SOURCE_DIR}/src/utils/input_bit_stream.cc
)

set(SRC_GPU
    ${CMAKE_SOURCE_DIR}/src/gpu/GDFCompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/GDFDecompressor.cu
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFCompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFDecompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/output_bit_stream.cc
    ${CMAKE_SOURCE_DIR}/src/utils/input_bit_stream.cc
)

set(TEST_SOURCES_CPU
    CDF_test.cpp
    data/dataset_utils.cpp
)

set(TEST_SOURCES_GPU
    GDF_test.cpp
    data/dataset_utils.cpp
)

set(BENCHMARK_SOURCES
    benchmark.cpp
    data/dataset_utils.cpp
)

# 添加可执行文件
add_executable(test_cpu ${SRC_CPU} ${TEST_SOURCES_CPU})

# 使用 CUDA 编译目标
cuda_add_executable(test_gpu ${SRC_GPU} ${TEST_SOURCES_GPU})

cuda_add_executable(benchmark_tests ${SRC_CPU} ${SRC_GPU} ${BENCHMARK_SOURCES})
add_executable(test_ALP test_ALP.cpp data/dataset_utils.cpp ${SRC_CPU})
cuda_add_executable(test_ndzip test_ndzip.cpp data/dataset_utils.cpp)

# 链接库时使用 PRIVATE，确保符号在不同目标之间不可见
target_link_libraries(test_cpu PRIVATE GTest::GTest GTest::Main pthread ALP)
target_link_libraries(test_gpu ${CUDA_LIBRARIES} GTest::GTest GTest::Main pthread ALP)
target_link_libraries(benchmark_tests ${CUDA_LIBRARIES} benchmark::benchmark GTest::GTest GTest::Main pthread ALP)
target_link_libraries(test_ALP PRIVATE GTest::GTest GTest::Main pthread ALP)
target_link_libraries(test_ndzip ${CUDA_LIBRARIES} GTest::GTest ndzip-cuda io Boost::program_options GTest::Main ndzip pthread )  # 链接到 ndzip 库

# 设置 include 目录
target_include_directories(test_cpu PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include ${CMAKE_SOURCE_DIR}/test/data ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/utils)
target_include_directories(test_gpu PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include ${CMAKE_SOURCE_DIR}/test/data ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/utils)
target_include_directories(benchmark_tests PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include ${CMAKE_SOURCE_DIR}/test/data ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/utils)
target_include_directories(test_ALP PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include ${CMAKE_SOURCE_DIR}/test/data ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/utils)
# 在 CMake 配置文件中为 test_ndzip 目标增加 io 目录
target_include_directories(test_ndzip PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/src/io  # 这里添加 io 的路径
    ${CMAKE_SOURCE_DIR}/test/data
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
)
# 设置编译标准
set_target_properties(test_cpu test_gpu benchmark_tests test_ALP PROPERTIES CXX_STANDARD 17)
target_compile_options(test_ndzip PRIVATE ${NDZIP_CUDA_FLAGS})