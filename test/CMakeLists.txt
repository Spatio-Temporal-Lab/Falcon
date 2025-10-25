cmake_minimum_required(VERSION 3.11)

project(CompressionTests)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")

# 查找 nvcomp
find_package(nvcomp CONFIG PATHS 
    /usr/lib/x86_64-linux-gnu/nvcomp/11/cmake/nvcomp
    /usr/lib/x86_64-linux-gnu/cmake/nvcomp
    NO_DEFAULT_PATH
)

if(nvcomp_FOUND)
    message(STATUS "Found nvcomp using CMake config")
    set(NVCOMP_LIBRARIES nvcomp::nvcomp)
    set(NVCOMP_INCLUDE_DIRS "")  # 由目标自己处理
endif()

# 添加子项目
add_subdirectory(baseline/ALP)
add_subdirectory(baseline/ALP_g)
add_subdirectory(baseline/ndzip)
add_subdirectory(baseline/elf)
add_subdirectory(baseline/elf_star)
add_subdirectory(baseline/elf_star_g)
add_subdirectory(baseline/mpc)

# 查找依赖包
find_package(GTest REQUIRED)
find_package(benchmark REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(CUDA REQUIRED)

message(STATUS "CUDA Include Directories: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 70)

# ==================== 重要修复：创建共享的CUDA对象库 ====================
# 这避免了多个目标重复编译相同的CUDA文件

# CPU源文件库
add_library(cdf_cpu_lib STATIC
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFCompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/cpu/CDFDecompressor.cpp
    ${CMAKE_SOURCE_DIR}/src/utils/output_bit_stream.cc
    ${CMAKE_SOURCE_DIR}/src/utils/input_bit_stream.cc
)

target_include_directories(cdf_cpu_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
)

# GPU源文件编译为对象库（关键修复）
cuda_add_library(Falcon_gpu_lib STATIC
    ${CMAKE_SOURCE_DIR}/src/gpu/FalconCompressor_1024.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/FalconDecompressor_1024.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_compressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_decompressor.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_pipeline.cu
    ${CMAKE_SOURCE_DIR}/src/gpu/Falcon_float_pipeline.cu
)

target_include_directories(Falcon_gpu_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
)

# 数据工具库
add_library(dataset_utils_lib STATIC
    data/dataset_utils.cpp
)

target_include_directories(dataset_utils_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/test/data
)

# ==================== 全局包含目录 ====================
include_directories(
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP/include
    ${CMAKE_SOURCE_DIR}/test/data
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/utils
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include
    ${CMAKE_SOURCE_DIR}/test/baseline/elf
    ${CMAKE_SOURCE_DIR}/test/baseline/elf_star
    ${CMAKE_SOURCE_DIR}/test/baseline/elf_star_g
    ${CMAKE_SOURCE_DIR}/test/baseline/ALP_g/include
)

# ==================== CPU测试目标 ====================
add_executable(test_cpu 
    CDF_test.cpp
)
target_link_libraries(test_cpu PRIVATE 
    cdf_cpu_lib
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# ==================== GPU测试目标（使用共享库） ====================

# test_gpu_nopack
cuda_add_executable(test_gpu_nopack 
    Falcon_test_nopack.cu
)
target_link_libraries(test_gpu_nopack 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# test_gpu_spare
cuda_add_executable(test_gpu_spare
    Falcon_test_spare.cu
)
target_link_libraries(test_gpu_spare
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# test_gpu_br
cuda_add_executable(test_gpu_br 
    Falcon_test_br.cu
)
target_link_libraries(test_gpu_br 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# test_gpu
cuda_add_executable(test_gpu 
    Falcon_test.cu
)
target_link_libraries(test_gpu 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# test_muti_stream
cuda_add_executable(test_muti_stream 
    Falcon_test_muti_steam_f64.cu
)
target_link_libraries(test_muti_stream 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)


# test_muti_3step_block
cuda_add_executable(test_muti_3step_block 
    muti_type_3step_block.cu
)
target_link_libraries(test_muti_3step_block 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)

# test_muti_3step_noblock
cuda_add_executable(test_muti_3step_noblock 
    muti_type_3step_noblock.cu
)
target_link_libraries(test_muti_3step_noblock 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)

# ==================== Benchmark目标 ====================
cuda_add_executable(benchmark_tests 
    benchmark.cu
)
target_link_libraries(benchmark_tests 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    benchmark::benchmark 
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# ==================== 其他测试目标 ====================

# ALP测试
add_executable(test_ALP 
    test_ALP.cpp
)
target_link_libraries(test_ALP PRIVATE 
    cdf_cpu_lib
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

# ndzip测试
cuda_add_executable(test_ndzip 
    test_ndzip.cpp
)
target_link_libraries(test_ndzip 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    ndzip-cuda 
    io 
    Boost::program_options 
    GTest::Main 
    ndzip 
    pthread
)
target_include_directories(test_ndzip PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/src/io
)
target_compile_options(test_ndzip PRIVATE ${NDZIP_CUDA_FLAGS})

# ELF测试
add_executable(test_elf 
    test_elf.cpp
)
target_link_libraries(test_elf PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf
)

# ELF Star测试
add_executable(test_elf_star 
    test_elf_star.cpp
)
target_link_libraries(test_elf_star PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star
)

# ALP GPU测试
cuda_add_executable(test_ALP_g 
    test_ALP_g.cu
)
target_link_libraries(test_ALP_g 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    alp_gpu
    ALP
)

# ELF Star GPU测试
cuda_add_executable(test_elf_star_g 
    test_elf_star_g.cu
)
target_link_libraries(test_elf_star_g 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star_g
)
#=====================float=============================
# test_float_muti_stream
cuda_add_executable(test_muti_stream_opt 
    Falcon_test_muti_steam_opt_f64.cu
)
target_link_libraries(test_muti_stream_opt 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)


# test_float_muti_stream
cuda_add_executable(test_float_muti_stream 
    test_float_falcon.cu
)
target_link_libraries(test_float_muti_stream 
    Falcon_gpu_lib
    cdf_cpu_lib
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread
)

# ALP测试
add_executable(test_float_ALP 
    test_float_ALP.cpp
)
target_link_libraries(test_float_ALP PRIVATE 
    cdf_cpu_lib
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    ALP
)

cuda_add_executable(test_float_ALP_g
    test_float_ALP_g.cu
)
target_link_libraries(test_float_ALP_g
    ${CUDA_LIBRARIES} 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    alp_gpu
    ALP
)


# test_float_muti_stream
cuda_add_executable(test_float_Lz4 
    test_float_Lz4.cu
)
target_link_libraries(test_float_Lz4
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )

add_executable(test_float_elf
test_float_elf.cpp
)
target_link_libraries(test_float_elf PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf
)

# ELF Star测试
add_executable(test_float_elf_star 
    test_float_elf_star.cpp
)
target_link_libraries(test_float_elf_star PRIVATE 
    dataset_utils_lib
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star
)


# ELF Star GPU测试
cuda_add_executable(test_float_elf_star_g 
    test_float_elf_star_g.cu
)
target_link_libraries(test_float_elf_star_g 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    GTest::Main 
    pthread 
    elf_star_g
)

# gdeflate测试
    cuda_add_executable(test_float_gdeflate 
        test_float_gdeflate.cu
    )
    target_link_libraries(test_float_gdeflate
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    target_include_directories(test_float_gdeflate PRIVATE 
        ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include
    )

    

        # Snappy测试
    add_executable(test_float_Snappy 
        test_float_Snappy.cpp
    )
    target_link_libraries(test_float_Snappy
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    target_include_directories(test_float_Snappy PRIVATE 
        ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include
    )

    cuda_add_executable(test_float_ndzip 
    test_float_ndzip.cpp
)
target_link_libraries(test_float_ndzip 
    dataset_utils_lib
    ${CUDA_LIBRARIES} 
    GTest::GTest 
    ndzip-cuda 
    io 
    Boost::program_options 
    GTest::Main 
    ndzip 
    pthread
)
target_include_directories(test_float_ndzip PRIVATE
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/include
    ${CMAKE_SOURCE_DIR}/test/baseline/ndzip/src/io
)

# bitcomp测试
    cuda_add_executable(test_float_bitcomp 
        test_float_bitcomp.cu
    )
    target_link_libraries(test_float_bitcomp
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    target_include_directories(test_float_bitcomp PRIVATE 
        ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include
    )

# ==================== nvcomp相关测试（如果找到） ====================
if(nvcomp_FOUND)
    # LZ4测试
    cuda_add_executable(test_LZ4 
        test_LZ4.cu
    )
    target_link_libraries(test_LZ4
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    target_include_directories(test_LZ4 PRIVATE 
        ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include
    )

    # gdeflate测试
    cuda_add_executable(test_gdeflate 
        test_gdeflate.cu
    )
    target_link_libraries(test_gdeflate
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    target_include_directories(test_gdeflate PRIVATE 
        ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include
    )

    # Snappy测试
    add_executable(test_Snappy 
        test_Snappy.cpp
    )
    target_link_libraries(test_Snappy
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    target_include_directories(test_Snappy PRIVATE 
        ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include
    )

    # bitcomp测试
    cuda_add_executable(test_bitcomp 
        test_bitcomp.cu
    )
    target_link_libraries(test_bitcomp
        dataset_utils_lib
        ${CUDA_LIBRARIES}
        ${NVCOMP_LIBRARIES}
        GTest::GTest
        GTest::Main
        pthread
    )
    target_include_directories(test_bitcomp PRIVATE 
        ${CMAKE_SOURCE_DIR}/test/baseline/nvcomp/include
    )
endif()

# ==================== 设置输出目录（避免冲突） ====================
set_target_properties(
    test_cpu test_gpu test_gpu_nopack test_gpu_spare test_gpu_br 
    # test_gpu_string
    test_muti_stream test_muti_3step_block test_muti_3step_noblock
    benchmark_tests test_ALP test_ALP_g test_ndzip test_elf test_elf_star test_elf_star_g
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

# ==================== 添加自定义命令用于清理 ====================
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/lib
    COMMENT "Cleaning all build files..."
)

# ==================== 打印配置信息 ====================
message(STATUS "=================================")
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA Version: ${CUDA_VERSION}")
message(STATUS "  CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  nvcomp Found: ${nvcomp_FOUND}")
message(STATUS "=================================")