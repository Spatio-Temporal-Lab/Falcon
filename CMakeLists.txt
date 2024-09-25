cmake_minimum_required(VERSION 3.20)

project(cuCompressor)

add_subdirectory(cuSZp)

add_subdirectory(Serf)

add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark cuSZp_libs gorilla elf)