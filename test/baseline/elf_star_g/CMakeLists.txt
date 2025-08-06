cmake_minimum_required(VERSION 3.20)

project(Elf_star_g LANGUAGES CXX CUDA)

# Find CUDA package - this is essential for using CUDA:: targets
find_package(CUDAToolkit REQUIRED)

# Set C++ standard version
set(CMAKE_CXX_STANDARD 17)

# -O3 Optimization for release version
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_ARCHITECTURES "75;80;86" CACHE STRING "CUDA architectures")
# Set parallel compilation level as 4
set(CMAKE_BUILD_PARALLEL_LEVEL 4)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(ELF_STAR_SOURCES
        utils.cu
        utils/post_office_solver.cu
        defs.cuh
        Elf_Star_g_Kernel.cuh
        ElfStarComKernel.cu
        ElfStarDecHost.cu
        ElfStarDecKernel.cu
        ElfStarComHost.cu
)

# Create shared library
add_library(elf_star_g SHARED ${ELF_STAR_SOURCES})

# Set CUDA separable compilation to resolve device function linking issues
set_property(TARGET elf_star_g PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Set CUDA resolve device symbols
set_property(TARGET elf_star_g PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# --- 最佳实践：使用 target_include_directories ---
# 这会告诉 'elf_star_g' 这个目标去哪里寻找它需要的头文件。
target_include_directories(elf_star_g PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/BitStream
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# --- 最佳实践：链接必要的库 ---
# 将CUDA运行时库链接到你的目标
target_link_libraries(elf_star_g PRIVATE CUDA::cudart)

# Optional: Set CUDA architectures if needed (suppress deprecation warnings)
set_target_properties(elf_star_g PROPERTIES CUDA_ARCHITECTURES "75;80;86")

# Add compiler flags to suppress deprecation warnings
target_compile_options(elf_star_g PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

# ===== 添加测试程序 =====

# Create test executable
add_executable(elf_star_test test_elf_star.cu)

# Set CUDA separable compilation for test
set_property(TARGET elf_star_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Link test with the library
target_link_libraries(elf_star_test PRIVATE elf_star_g CUDA::cudart)

# Set same include directories for test
target_include_directories(elf_star_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/BitStream
        ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Set CUDA architectures for test
set_target_properties(elf_star_test PROPERTIES CUDA_ARCHITECTURES "75;80;86")

# Add compiler flags for test
target_compile_options(elf_star_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)

# Optional: Enable testing with CTest
enable_testing()
add_test(NAME ElfStarCompressionTest COMMAND elf_star_test)