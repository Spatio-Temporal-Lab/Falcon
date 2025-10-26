# Falcon: 基于GPU的自适应浮点数无损压缩框架

**Falcon** 是一个高性能的基于GPU加速的无损压缩框架，专门为浮点数时间序列数据设计。它通过三个关键创新利用现代GPU架构，实现了前所未有的压缩比和吞吐量：异步流水线、精确的浮点数到整数转换以及自适应稀疏位平面编码。

## 📊 性能亮点

- **压缩比**: 平均0.299（比最好的CPU竞争对手提高21%）
- **压缩吞吐量**: 平均10.82 GB/s（比最快的GPU竞争对手快2.43倍）
- **解压缩吞吐量**: 平均12.32 GB/s（比最快的GPU竞争对手快2.4倍）

## 🚀 主要特性

### 🎯 异步流水线

- **事件驱动的调度器**: 在CPU和GPU数据传输期间隐藏I/O延迟
- **多流处理**: 支持最多16个并发流
- **双向PCIe利用**: 重叠H2D和D2H通信

### 🔢 精度保持转换

- **理论保证**: 消除浮点数算术误差
- **自适应数字转换**: 处理正常（β≤15, α≤22）和异常情况
- **无损恢复**: 精确重建原始浮点数值

### 🎚️ 自适应稀疏位平面编码

- **双存储方案**: 稀疏存储用于零主导的位平面，稠密存储用于其他
- **异常值恢复**: 减轻由异常引起的稀疏性退化
- **线程束发散最小化**: 为GPU并行执行优化

## 🛠️ 环境要求

### 已验证的环境

#### 基础环境1 (WSL2)

- **操作系统**: Ubuntu 22.04.5 LTS
- **编译器**: g++ 11.4
- **构建系统**: CMake 3.22.1
- **CUDA**: nvcc 12.8/11.6
- **GPU**: NVIDIA GeForce RTX 3050

#### 基础环境2 (原生Ubuntu)

- **操作系统**: Ubuntu 24.04.2 LTS
- **编译器**: g++ 11.4
- **构建系统**: CMake 3.28.1
- **CUDA**: nvcc 12.0
- **GPU**: NVIDIA GeForce RTX 5080

### 必要依赖

#### 基本构建工具

```bash
# 适用于Ubuntu 22.04/24.04
sudo apt update && sudo apt upgrade
sudo apt install -y git build-essential
```

#### CMake 安装

```bash
# Ubuntu 22.04 (CMake 3.22)
sudo apt install -y cmake

# Ubuntu 24.04 (CMake 3.28) 或更新版本
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main'
sudo apt update
sudo apt install -y cmake
```

#### CUDA 工具包安装

```bash
# 适用于 CUDA 12.x（兼容 RTX 3050/5080）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-0

# 适用于 CUDA 11.x（如果需要兼容性）
sudo apt install -y cuda-toolkit-11-8
```

#### 必要库

```bash
# Boost (program_options 组件)
sudo apt install -y libboost-all-dev

# Google Test (GTest)
sudo apt install -y libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib

# Google Benchmark
sudo apt install -y libbenchmark-dev

# NVIDIA nvcomp (用于基线比较)
sudo apt-get -y install nvcomp-cuda-11
# or
sudo apt-get -y install nvcomp-cuda-12
```

### 环境验证

**bash**

```
# 检查编译器版本
g++ --version
cmake --version
nvcc --version

# 验证CUDA安装
nvidia-smi
```

## 🏗️ 代码架构

### 头文件结构

#### GPU 优化版本 (每个线程处理1025个元素)

* `Falcon_compressor.cuh` - 优化的GPU压缩器（1个线程处理1025个元素）
* `Falcon_decompressor.cuh` - 优化的GPU解压缩器（1个线程处理1025个元素）

#### GPU 单精度浮点数版本

* `Falcon_float_compressor.cuh` - 单精度浮点数（32位）专用的GPU压缩器
* `Falcon_float_decompressor.cuh` - 单精度浮点数（32位）专用的GPU解压缩器

#### GPU 基础版本 (每个线程处理1024个元素)

* `FalconCompressor_1024.cuh` - 基础GPU压缩器（1个线程处理1024个元素）
* `FalconDecompressor_1024.cuh` - 基础GPU解压缩器（1个线程处理1024个元素）

#### GPU 流水线版本

* `Falcon_pipeline.cuh` - 包含消融测试接口的流水线实现
* `Falcon_float_pipeline.cuh` - 单精度浮点数流水线实现

### 源代码实现

**text**

```
src/
├── gpu/           # GPU内核实现
└── utils/         # 位流工具和辅助函数
```

### 并行设计

* **块大小** : 每个GPU线程处理1024或1025个元素
* **线程映射** : 每个线程处理一个完整的块
* **线程束效率** : 为32线程的线程束执行优化
* **内存访问** : 合并的全局内存访问模式

## 🔨 构建

### 快速构建脚本

**bash**

```
#!/bin/bash
set -x
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### 手动构建

1. 克隆仓库：

   ```bash
   git clone <repository-url>
   cd Falcon
   ```
2. 生成CMake构建系统：

   ```bash
   cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
   ```
3. 构建所有目标：

   ```bash
   cmake --build ./build --config Release -j$(nproc)
   ```

## 🧪 测试

### 测试结构

**text**

```
test/
├── baseline/          # 比较算法 (ALP, ndzip, elf 等)
├── data/             # 测试数据集
├── Falcon_test_*.cu  # 主GPU测试套件
└── test_*.cpp/cu     # 特定算法测试
```

### 运行测试

#### 所有测试的基本用法

```bash
./test/test_${test_name} --dir ../test/data/use/
```

#### 基准测试 (与基线比较)

```bash
# 主GPU实现 (每个线程1024个元素)
./test/test_gpu --dir ../test/data/use/

# 无打包优化的GPU
./test/test_gpu_nopack --dir ../test/data/use/

# 位减少优化的GPU
./test/test_gpu_br --dir ../test/data/use/

# 稀疏优化的GPU
./test/test_gpu_spare --dir ../test/data/use/
```

#### 多流性能测试

```bash
# 优化的多流
./test/test_muti_stream_opt --dir ../test/data/use/
```

### 消融实验

#### 编码策略消融

* **全稀疏** : 所有位平面使用稀疏存储
* **全稠密** : 所有位平面使用稠密存储
* **暴力误差** : 不精确的十进制位置计算
* **标准** : 自适应稀疏/稠密选择（默认）

#### 流水线消融

* **单流** : 顺序处理
* **阻塞** : 同步多流
* **非阻塞** : 异步多流
* **标准** : 事件驱动调度器（默认）

### 完整测试脚本

```bash
#!/bin/bash
set -x
cd Falcon
mkdir -p build
cd build

# 编译项目
cmake ..
make -j

# 运行所有测试
run_test() {
    local test_name=$1
    echo "===== 运行 ${test_name} ====="
    ./test/test_${test_name} --dir ../test/data/use/
}

# 核心GPU测试
run_test "gpu"
run_test "gpu_nopack"
run_test "gpu_br"
run_test "gpu_spare"

# 多流测试
run_test "muti_3step_block"
run_test "muti_3step_noblock"
run_test "muti_stream_opt"
```

## 📊 实验结果

### 压缩比比较

| 方法             | 平均压缩比      | 相对于Falcon的改进 |
| ---------------- | --------------- | ------------------ |
| **Falcon** | **0.299** | -                  |
| ALP              | 0.329           | 差9.1%             |
| Elf*             | 0.339           | 差13.4%            |
| Elf              | 0.380           | 差27.1%            |
| ndzip            | 0.996           | 差233%             |

### 吞吐量性能

| 操作   | Falcon     | 最佳竞争对手         | 加速比 |
| ------ | ---------- | -------------------- | ------ |
| 压缩   | 10.82 GB/s | 4.46 GB/s (GDeflate) | 2.43× |
| 解压缩 | 12.32 GB/s | 5.13 GB/s (GPU:Elf*) | 2.4×  |

## 🔧 配置

### 默认参数

* **块大小** : 每个线程1024或1025个元素
* **批大小** : 1025 × 1024 × 4 个元素
* **流水线流数** : 16
* **GPU架构** : 计算能力7.0+

### 块大小考虑

* **1024个元素** : 与2的幂对齐以便内存寻址
* **1025个元素** : 优化内存空间利用，减少内存浪费
* **线程映射** : 每个GPU线程处理恰好一个块

### 构建选项

* `-DCMAKE_BUILD_TYPE=Release` 用于优化性能
* `-DCMAKE_CUDA_ARCHITECTURES=70` 用于特定GPU架构

## 📚 引用

如果您在研究中使用了Falcon，请引用：

**bibtex**

```
@article{falcon2025,
  title={Falcon: GPU-Based Floating-point Adaptive Lossless Compression},
  author={Li, Zheng and Wang, Weiyan and Li, Ruiyuan and Chen, Chao and Long, Xianlei and Zheng, Linjiang and Xu, Quanqing and Yang, Chuanhui},
  journal={PVLDB},
  volume={14},
  number={1},
  pages={XXX--XXX},
  year={2025},
  publisher={VLDB Endowment}
}
```

## 👥 作者

* **李政** (重庆大学) - zhengli@cqu.edu.cn
* **王伟俨** (重庆大学) - weiyan.wang@stu.cqu.edu.cn
* **李瑞远** (重庆大学) - ruiyuan.li@cqu.edu.cn
* **陈超** (重庆大学) - cschaochen@cqu.edu.cn
* **龙宪磊** (重庆大学) - xianlei.long@cqu.edu.cn
* **郑林江** (重庆大学) - zlj_cqu@cqu.edu.cn
* **徐泉清** (OceanBase, 蚂蚁集团) - xuquanqing.xqq@oceanbase.com
* **杨传辉** (OceanBase, 蚂蚁集团) - rizhao.ych@oceanbase.com

## 📄 许可证

此项目可用于学术和研究用途。请参考仓库中的特定许可证条款。

## 🔗 相关出版物

* [Elf: 基于擦除的无损浮点数压缩](https://doi.org/10.14778/3594512.3594523)
* [ALP: 自适应无损浮点数压缩](https://doi.org/10.1145/3614332)
* [Serf: 流式误差有界浮点数压缩](https://doi.org/10.1145/3725353)

---

 **注意** : 本项目已在WSL2 (Ubuntu 22.04) 和原生Ubuntu 24.04环境下验证通过，并具有指定的依赖项。关于具体实现或性能特性的问题，请参考相应的头文件和测试用例。
