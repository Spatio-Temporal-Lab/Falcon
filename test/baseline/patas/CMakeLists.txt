cmake_minimum_required(VERSION 3.11)

# Patas 是一个header-only库，不需要编译源文件
# 只需要提供一个接口库供其他目标使用

add_library(patas INTERFACE)

target_include_directories(patas INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

message(STATUS "Patas baseline library configured (header-only)")
