#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cuSZp::cuSZp_shared" for configuration "Debug"
set_property(TARGET cuSZp::cuSZp_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(cuSZp::cuSZp_shared PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcuSZp.so"
  IMPORTED_SONAME_DEBUG "libcuSZp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS cuSZp::cuSZp_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_cuSZp::cuSZp_shared "${_IMPORT_PREFIX}/lib/libcuSZp.so" )

# Import target "cuSZp::cuSZp_static" for configuration "Debug"
set_property(TARGET cuSZp::cuSZp_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(cuSZp::cuSZp_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CUDA"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcuSZp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS cuSZp::cuSZp_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_cuSZp::cuSZp_static "${_IMPORT_PREFIX}/lib/libcuSZp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
