cmake_minimum_required(VERSION 3.10)

project(ALP)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含头文件目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 源文件路径
file(GLOB_RECURSE ALP_SOURCES "src/*.cpp")

# 创建 ALP 静态库
add_library(ALP STATIC ${ALP_SOURCES})
