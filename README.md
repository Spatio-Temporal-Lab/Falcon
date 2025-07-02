# CuCompressor

```
mkdir build
cd build
cmake ..
cmake .. \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-11
  
make -j4
./test/benchmark_tests 
./test/test_cpu 
./test/test_ALP
./test/test_gpu
```

**# 只运行流式压缩测试**
./test/test_gpu  --gtest_filter=GDFStreamTest

**# 只运行性能对比测试**
./test/test_gpu --gtest_filter=*GDFPerformanceTest.*

**# 单独文件夹测试（所有测试都适配了这个指令）**
./test/test_gpu --dir ../test/data/neon
./test/test_ALP --dir ../test/data/neon
