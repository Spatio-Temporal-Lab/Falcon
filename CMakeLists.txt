cmake_minimum_required(VERSION 3.20)

project(cuCompressor-all)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(cuSZp)

add_subdirectory(Serf)

add_subdirectory(src)

add_subdirectory(test)


add_executable(benchmark benchmark.cpp)

target_link_libraries(benchmark cuSZp_libs gorilla elf)