# CMake generated Testfile for 
# Source directory: /home/lz/workspace/cuCompressor/Serf/test
# Build directory: /home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/buff_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/chimp_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/deflate_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/elf_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/fpc_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/gorilla_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/lz4_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/lz77_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/machete_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/serf_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/sim_piece_test[1]_include.cmake")
include("/home/lz/workspace/cuCompressor/cmake-build-debug/Serf/test/PerformanceProgram[1]_include.cmake")
subdirs("baselines/deflate")
subdirs("baselines/fpc")
subdirs("baselines/lz4")
subdirs("baselines/chimp128")
subdirs("baselines/gorilla")
subdirs("baselines/elf")
subdirs("baselines/machete")
subdirs("baselines/lz77")
subdirs("baselines/sz2")
subdirs("baselines/buff")
subdirs("baselines/snappy")
subdirs("baselines/zstd/build/cmake")
subdirs("baselines/sim_piece")
subdirs("../../_deps/googletest-build")
