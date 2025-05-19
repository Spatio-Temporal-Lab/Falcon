# CuCompressor

```
mkdir build
cd build
cmake ..
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
