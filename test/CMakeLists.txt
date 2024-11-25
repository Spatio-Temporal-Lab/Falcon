cmake_minimum_required(VERSION 3.11)


project(CompressionTests)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 GTest
find_package(GTest REQUIRED)

# 查找 Google Benchmark 库
find_package(benchmark REQUIRED)

# 包含头文件路径
include_directories(${CMAKE_SOURCE_DIR}/test/baseline/ALP/include)
include_directories(${CMAKE_SOURCE_DIR}/test/data)
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/utils)

# 源文件
set(SRC_CPU
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFCompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFDecompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/output_bit_stream.cc
    ${CMAKE_SOURCE_DIR}/src/utils/input_bit_stream.cc
)

set(TEST_SOURCES
    CDF_test.cpp
    data/dataset_utils.cpp
)

set(BENCHMARK_SOURCES
    benchmark.cpp
    data/dataset_utils.cpp
)

# 添加测试文件和可执行文件
add_executable(test_cpu ${SRC_CPU} ${TEST_SOURCES})
add_executable(benchmark_tests ${SRC_CPU} ${BENCHMARK_SOURCES})
add_executable(test_ALP test_ALP.cpp data/dataset_utils.cpp)

# 链接库时使用 PRIVATE，确保符号在不同目标之间不可见
target_link_libraries(test_cpu PRIVATE GTest::GTest GTest::Main pthread ALP)
target_link_libraries(benchmark_tests PRIVATE benchmark::benchmark GTest::GTest GTest::Main pthread ALP)
target_link_libraries(test_ALP PRIVATE GTest::GTest GTest::Main pthread ALP)

# 设置 include 目录
target_include_directories(test_cpu PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include ${CMAKE_SOURCE_DIR}/test/data ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/utils)
target_include_directories(benchmark_tests PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include ${CMAKE_SOURCE_DIR}/test/data ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/utils)
target_include_directories(test_ALP PRIVATE ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include ${CMAKE_SOURCE_DIR}/test/data ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src/utils)

# 设置编译标准
set_target_properties(test_cpu benchmark_tests test_ALP PROPERTIES CXX_STANDARD 17)

