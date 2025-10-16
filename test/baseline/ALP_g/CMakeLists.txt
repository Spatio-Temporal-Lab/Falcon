# baseline/ALP_g/CMakeLists.txt
cmake_minimum_required(VERSION 3.11)
project(alp_gpu CUDA CXX)

# 设置CUDA标准
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)

# 设置CUDA架构（根据您的GPU调整）
set(CMAKE_CUDA_ARCHITECTURES 70)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/ALP/include

    ${CUDA_INCLUDE_DIRS}
)

# 源文件
set(ALP_GPU_SOURCES
    src/alp_gpu.cu
    src/alp_gpu_hybrid.cu
)

set(ALP_GPU_HEADERS
    include/alp_gpu.hpp
    include/alp_c.h
)

# 使用cuda_add_library创建静态库
cuda_add_library(alp_gpu STATIC ${ALP_GPU_SOURCES})

# 设置目标属性
set_target_properties(alp_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# 设置编译选项
target_compile_options(alp_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xcompiler=-fPIC
        -Xcompiler=-Wall
        -Xcompiler=-Wextra
    >
)

# 导出包含目录
target_include_directories(alp_gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 使用plain signature链接CUDA库（移除PUBLIC关键字）
target_link_libraries(alp_gpu
    ${CUDA_LIBRARIES}
    ${CUDA_cudart_LIBRARY}
)

# 如果使用CUDA 11或更高版本，可能需要添加这些
if(CUDA_VERSION_MAJOR GREATER_EQUAL 11)
    set_property(TARGET alp_gpu PROPERTY CUDA_ARCHITECTURES 70 75 80 86)
endif()

# 确保设备代码正确链接
set_property(TARGET alp_gpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(alp_compare
  examples/compare_main.cpp
)

target_include_directories(alp_compare
  PRIVATE ${ALP_INCLUDE_DIR}
)

target_link_libraries(alp_compare
  PRIVATE alp_gpu
)
