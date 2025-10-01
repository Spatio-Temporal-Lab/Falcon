# ALP GPU vs CPU 比较

## 1. 准备
将你的 3 个算法文件放到指定位置（覆盖占位符）：
- `include/alp_c_compression.h`
- `include/alp_gpu.hpp`
- `src/gpu/alp_gpu.cu`

## 2. 构建
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
````

## 3. 运行

* 合成数据：

```bash
./alp_compare --n 10000000 --pattern 1 --vector 1000 --block 100000
```

* 从 CSV 第一列读取：

```bash
./alp_compare --csv ../data/your.csv --vector 1000 --block 100000
```

* 运行参数：
    ALP_GPU_DIAG=1 显示 CPU & GPU 运行信息

## 4. 输出示例

```
N = 10000000 (76.294 MB)
CPU:  comp=1234.567 ms, decomp=890.123 ms, size=60.821 MB, ratio=0.797, mse=0.000, mismatches=0
GPU:  comp=210.345 ms, decomp=155.678 ms, size=60.812 MB, ratio=0.797, mse=0.000, mismatches=0
CPU vs GPU: mismatches=0, mse=0.000
```

> 注：若你的 CUDA 代码采用“一个线程一个数据块”的策略，请把 `--block` 设为与你实现中 `Params.blockSize` 一致，以获得最公平的对比。