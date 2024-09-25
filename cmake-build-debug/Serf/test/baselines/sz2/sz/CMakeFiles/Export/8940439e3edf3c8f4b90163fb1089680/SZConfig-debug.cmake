#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "sz" for configuration "Debug"
set_property(TARGET sz APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(sz PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libSZ.so"
  IMPORTED_SONAME_DEBUG "libSZ.so"
  )

list(APPEND _cmake_import_check_targets sz )
list(APPEND _cmake_import_check_files_for_sz "${_IMPORT_PREFIX}/lib/libSZ.so" )

# Import target "zstd" for configuration "Debug"
set_property(TARGET zstd APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(zstd PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libzstd.so"
  IMPORTED_SONAME_DEBUG "libzstd.so"
  )

list(APPEND _cmake_import_check_targets zstd )
list(APPEND _cmake_import_check_files_for_zstd "${_IMPORT_PREFIX}/lib/libzstd.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
