#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cuSZp::cuSZp" for configuration "Debug"
set_property(TARGET cuSZp::cuSZp APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(cuSZp::cuSZp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CUDA;CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcuSZpd.a"
  )

list(APPEND _cmake_import_check_targets cuSZp::cuSZp )
list(APPEND _cmake_import_check_files_for_cuSZp::cuSZp "${_IMPORT_PREFIX}/lib/libcuSZpd.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
