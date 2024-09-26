
cmake_minimum_required(VERSION 3.10)

project(CompressionTests)


# 源文件
set(TEST_SOURCES
        test_compression.cpp)
# 创建可执行测试文件

# Auto fetch GoogleTest source code
FetchContent_Declare(
        googletest
        DOWNLOAD_EXTRACT_TIMESTAMP true
        URL https://github.com/google/googletest/archive/refs/tags/v${GTEST_VER}.zip)
# For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(compression_tests ${TEST_SOURCES})

