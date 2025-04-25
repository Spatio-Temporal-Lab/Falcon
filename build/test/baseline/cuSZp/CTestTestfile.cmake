# CMake generated Testfile for 
# Source directory: /home/longxl/cuda/cuCompressor/test/baseline/cuSZp
# Build directory: /home/longxl/cuda/cuCompressor/build/test/baseline/cuSZp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[cuSZpTest_f32]=] "/home/longxl/cuda/cuCompressor/install/bin/cuSZp_test_f32")
set_tests_properties([=[cuSZpTest_f32]=] PROPERTIES  WORKING_DIRECTORY "/home/longxl/cuda/cuCompressor/install/bin" _BACKTRACE_TRIPLES "/home/longxl/cuda/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;94;add_test;/home/longxl/cuda/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;0;")
add_test([=[cuSZpTest_f64]=] "/home/longxl/cuda/cuCompressor/install/bin/cuSZp_test_f64")
set_tests_properties([=[cuSZpTest_f64]=] PROPERTIES  WORKING_DIRECTORY "/home/longxl/cuda/cuCompressor/install/bin" _BACKTRACE_TRIPLES "/home/longxl/cuda/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;95;add_test;/home/longxl/cuda/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;0;")
add_test([=[GDFCTest_f32]=] "/home/longxl/cuda/cuCompressor/install/bin/GDFC_test_f32")
set_tests_properties([=[GDFCTest_f32]=] PROPERTIES  WORKING_DIRECTORY "/home/longxl/cuda/cuCompressor/install/bin" _BACKTRACE_TRIPLES "/home/longxl/cuda/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;96;add_test;/home/longxl/cuda/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;0;")
subdirs("examples")
