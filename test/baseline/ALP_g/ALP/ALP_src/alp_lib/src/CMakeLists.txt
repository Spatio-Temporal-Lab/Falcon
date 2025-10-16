# 定义 ALP 库并添加源文件
add_library(ALP SHARED
    falp.cpp
    fastlanes_ffor.cpp
    fastlanes_generated_ffor.cpp
    fastlanes_generated_unffor.cpp
    fastlanes_unffor.cpp
)

# 设置头文件的公开包含目录
target_include_directories(ALP PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# 设置编译选项（可选）
set_target_properties(ALP PROPERTIES CXX_STANDARD 17)

# 添加 Clang-Tidy 静态代码检查（可选）
find_program(CLANG_TIDY_EXE NAMES clang-tidy)
if (CLANG_TIDY_EXE)
    set_target_properties(ALP PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif ()
