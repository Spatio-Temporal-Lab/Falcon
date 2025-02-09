project(cudpp C CUDA)

set(cudpp_VERSION_MAJOR 2)
set(cudpp_VERSION_MINOR 3)
set(cudpp_VERSION_PATCH 0)

cmake_minimum_required(VERSION 2.8)

option(CUDPP_BUILD_SHARED_LIBS
  "On to build shared libraries, off for static libraries."
  OFF
  )

option(CUDPP_BUILD_APPLICATIONS
  "If on, builds the sample applications."
  OFF
  )

set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  )

set(cudpp_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/ext/moderngpu/include/
  )
include_directories(${cudpp_INCLUDE_DIRS})

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(bits 64)
else ()
  set(bits 32)
endif (CMAKE_SIZEOF_VOID_P EQUAL 8)
string(TOUPPER "${CMAKE_CONFIGURATION_TYPES}" configs)
foreach (config ${configs})
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${EXECUTABLE_OUTPUT_PATH}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${LIBRARY_OUTPUT_PATH}")
  set(CMAKE_${config}_POSTFIX ${bits})
endforeach(config {configs})
set(CMAKE_DEBUG_POSTFIX "${bits}d")

if(CUDPP_BUILD_SHARED_LIBS)
  # User wants to build Dynamic Libraries, so change the LIB_TYPE variable to CMake keyword 'SHARED'
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
  set(CUDPP_STATIC_LIB 1)
endif(CUDPP_BUILD_SHARED_LIBS)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cudpp_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/include/cudpp_config.h
  )

set (CUDPP_APP_DATA_DIR
  "${CMAKE_SOURCE_DIR}/apps/data/"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/common/include/common_config.h.in
  ${CMAKE_CURRENT_SOURCE_DIR}/apps/common/include/common_config.h
  )

# copy WIN32 DLLs
if (WIN32)
if (bits EQUAL 64)
file(COPY bin/win32/glut32.dll bin/win64/glew64.dll DESTINATION bin/)
elseif (bits EQUAL 32)
file(COPY bin/win32/glut32.dll bin/win32/glew32.dll DESTINATION bin/)
endif (bits EQUAL 64)
endif (WIN32)

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  exec_program(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
  string(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
  # message(STATUS "Detected Darwin kernel version: ${DARWIN_VERSION}")
  if ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
    set(APPLE_MAVERICKS_OR_LATER 1 INTERNAL)
    set(CMAKE_CXX_FLAGS -stdlib=libstdc++)
    set(CMAKE_C_FLAGS -stdlib=libstdc++)
    if(CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR LESS 5)
      set(CUDA_6_0 1)
    endif(CUDA_VERSION_MAJOR EQUAL 6 AND CUDA_VERSION_MINOR LESS 5)
    if(CUDA_VERSION_MAJOR LESS 6 OR CUDA_6_0)
      if(NOT CUDA_NVCC_FLAGS MATCHES "ccbin")
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ccbin;/usr/bin/clang)
      endif(NOT CUDA_NVCC_FLAGS MATCHES "ccbin")
    endif(CUDA_VERSION_MAJOR LESS 6 OR CUDA_6_0)
  endif ((DARWIN_VERSION GREATER 13) OR (DARWIN_VERSION EQUAL 13))
  # current CUDA on Mavericks uses libstdc++, not libc++
  # https://github.com/cudpp/cudpp/wiki/BuildingCUDPPWithMavericks
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

add_subdirectory(src/cudpp)
