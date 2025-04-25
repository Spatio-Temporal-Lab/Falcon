# CMake generated Testfile for 
# Source directory: /mnt/c/Users/29836/Downloads/cuSZp/cuSZp
# Build directory: /mnt/c/Users/29836/Downloads/cuSZp/cuSZp/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[cuSZpTest_f32]=] "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/install/bin/cuSZp_test_f32")
set_tests_properties([=[cuSZpTest_f32]=] PROPERTIES  WORKING_DIRECTORY "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/install/bin" _BACKTRACE_TRIPLES "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/CMakeLists.txt;94;add_test;/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/CMakeLists.txt;0;")
add_test([=[cuSZpTest_f64]=] "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/install/bin/cuSZp_test_f64")
set_tests_properties([=[cuSZpTest_f64]=] PROPERTIES  WORKING_DIRECTORY "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/install/bin" _BACKTRACE_TRIPLES "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/CMakeLists.txt;95;add_test;/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/CMakeLists.txt;0;")
add_test([=[GDFCTest_f32]=] "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/install/bin/GDFC_test_f32")
set_tests_properties([=[GDFCTest_f32]=] PROPERTIES  WORKING_DIRECTORY "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/install/bin" _BACKTRACE_TRIPLES "/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/CMakeLists.txt;96;add_test;/mnt/c/Users/29836/Downloads/cuSZp/cuSZp/CMakeLists.txt;0;")
subdirs("examples")
