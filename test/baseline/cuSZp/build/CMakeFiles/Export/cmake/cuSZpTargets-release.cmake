#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cuSZp::cuSZp_shared" for configuration "Release"
set_property(TARGET cuSZp::cuSZp_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cuSZp::cuSZp_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcuSZp.so"
  IMPORTED_SONAME_RELEASE "libcuSZp.so"
  )

list(APPEND _IMPORT_CHECK_TARGETS cuSZp::cuSZp_shared )
list(APPEND _IMPORT_CHECK_FILES_FOR_cuSZp::cuSZp_shared "${_IMPORT_PREFIX}/lib/libcuSZp.so" )

# Import target "cuSZp::cuSZp_static" for configuration "Release"
set_property(TARGET cuSZp::cuSZp_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cuSZp::cuSZp_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CUDA"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libcuSZp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS cuSZp::cuSZp_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_cuSZp::cuSZp_static "${_IMPORT_PREFIX}/lib/libcuSZp.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
