# CMake generated Testfile for 
# Source directory: /home/lizhzz/workspace/github/study/cuCompressor/test/baseline/cuSZp
# Build directory: /home/lizhzz/workspace/github/study/cuCompressor/build/test/baseline/cuSZp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[cuSZpTest_f32]=] "/home/lizhzz/workspace/github/study/cuCompressor/install/bin/cuSZp_test_f32")
set_tests_properties([=[cuSZpTest_f32]=] PROPERTIES  WORKING_DIRECTORY "/home/lizhzz/workspace/github/study/cuCompressor/install/bin" _BACKTRACE_TRIPLES "/home/lizhzz/workspace/github/study/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;94;add_test;/home/lizhzz/workspace/github/study/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;0;")
add_test([=[cuSZpTest_f64]=] "/home/lizhzz/workspace/github/study/cuCompressor/install/bin/cuSZp_test_f64")
set_tests_properties([=[cuSZpTest_f64]=] PROPERTIES  WORKING_DIRECTORY "/home/lizhzz/workspace/github/study/cuCompressor/install/bin" _BACKTRACE_TRIPLES "/home/lizhzz/workspace/github/study/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;95;add_test;/home/lizhzz/workspace/github/study/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;0;")
add_test([=[GDFCTest_f32]=] "/home/lizhzz/workspace/github/study/cuCompressor/install/bin/GDFC_test_f32")
set_tests_properties([=[GDFCTest_f32]=] PROPERTIES  WORKING_DIRECTORY "/home/lizhzz/workspace/github/study/cuCompressor/install/bin" _BACKTRACE_TRIPLES "/home/lizhzz/workspace/github/study/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;96;add_test;/home/lizhzz/workspace/github/study/cuCompressor/test/baseline/cuSZp/CMakeLists.txt;0;")
subdirs("examples")
