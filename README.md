# Falcon: GPU-Based Floating-point Adaptive Lossless Compression

**Falcon** is a high-performance GPU-accelerated lossless compression framework specifically designed for floating-point time series data. It achieves unprecedented compression ratios and throughput by leveraging modern GPU architectures through three key innovations: asynchronous pipeline, precise float-to-integer conversion, and adaptive sparse bit-plane encoding.

## 📊 Performance Highlights

- **Compression Ratio**: Average 0.299 (21% improvement over best CPU competitors)
- **Compression Throughput**: Average 10.82 GB/s (2.43× faster than fastest GPU competitors)
- **Decompression Throughput**: Average 12.32 GB/s (2.4× faster than fastest GPU competitors)

## 🚀 Key Features

### 🎯 Asynchronous Pipeline

- **Event-Driven Scheduler**: Hides I/O latency during CPU-GPU data transmission
- **Multi-stream Processing**: Supports up to 16 concurrent streams
- **Bidirectional PCIe Utilization**: Overlaps H2D and D2H communications

### 🔢 Precision-Preserving Conversion

- **Theoretical Guarantees**: Eliminates floating-point arithmetic errors
- **Adaptive Digit Transformation**: Handles both normal (β≤15, α≤22) and exceptional cases
- **Lossless Recovery**: Exact reconstruction of original floating-point values

### 🎚️ Adaptive Sparse Bit-Plane Encoding

- **Dual Storage Schemes**: Sparse storage for zero-dominated planes, dense storage for others
- **Outlier Resilience**: Mitigates sparsity degradation caused by anomalies
- **Warp Divergence Minimization**: Optimized for GPU parallel execution

## 🛠️ Prerequisite

### Verified Environments

#### Base Environment 1 (WSL2)

- **OS**: Ubuntu 22.04.5 LTS
- **Compiler**: g++ 11.4
- **Build System**: CMake 3.22.1
- **CUDA**: nvcc 12.8/11.6
- **GPU**: NVIDIA GeForce RTX 3050

#### Base Environment 2 (Native Ubuntu)

- **OS**: Ubuntu 24.04.2 LTS
- **Compiler**: g++ 11.4
- **Build System**: CMake 3.28.1
- **CUDA**: nvcc 12.0
- **GPU**: NVIDIA GeForce RTX 5080

### Required Dependencies

#### Essential Build Tools

```bash
# For Ubuntu 22.04/24.04
sudo apt update && sudo apt upgrade
sudo apt install -y git build-essential
```

#### CMake Installation

```bash
# Ubuntu 22.04 (CMake 3.22)
sudo apt install -y cmake

# Ubuntu 24.04 (CMake 3.28) or for newer version
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main'
sudo apt update
sudo apt install -y cmake
```

#### CUDA Toolkit Installation

```bash
# For CUDA 12.x (compatible with RTX 3050/5080)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-0

# For CUDA 11.x (if needed for compatibility)
sudo apt install -y cuda-toolkit-11-8
```

#### Required Libraries

```bash
# Boost (program_options component)
sudo apt install -y libboost-all-dev

# Google Test (GTest)
sudo apt install -y libgtest-dev
cd /usr/src/gtest
sudo cmake .
sudo make
sudo cp lib/*.a /usr/lib

# Google Benchmark
sudo apt install -y libbenchmark-dev

# NVIDIA nvcomp (for baseline comparisons)
sudo apt-get -y install nvcomp-cuda-11
# or
sudo apt-get -y install nvcomp-cuda-12
```

### Environment Verification

```bash
# Check compiler versions
g++ --version
cmake --version
nvcc --version

# Verify CUDA installation
nvidia-smi
```

## 🏗️ Code Architecture

### Header Files Structure

#### GPU Optimized Version (1025 elements per thread)

* `Falcon_compressor.cuh` - Optimized GPU compressor (1 thread processes 1025 elements)
* `Falcon_decompressor.cuh` - Optimized GPU decompressor (1 thread processes 1025 elements)

#### GPU Single Precision Version

* `Falcon_float_compressor.cuh` - Single precision floating-point GPU compressor
* `Falcon_float_decompressor.cuh` - Single precision floating-point GPU decompressor

#### GPU Base Version (1024 elements per thread)

* `FalconCompressor_1024.cuh` - Base GPU compressor (1 thread processes 1024 elements)
* `FalconDecompressor_1024.cuh` - Base GPU decompressor (1 thread processes 1024 elements)

#### GPU Pipeline Version

* `Falcon_pipeline.cuh` - Pipeline implementation with ablation interfaces
* `Falcon_float_pipeline.cuh` - Single precision floating-point pipeline implementation

### Source Implementation

**text**

```
src/
├── gpu/           # GPU kernel implementations
└── utils/         # Bit stream utilities and helper functions
```

### Parallelism Design

- **Chunk Size**: 1024 or 1025 elements per GPU thread
- **Thread Mapping**: Each thread processes one complete chunk
- **Warp Efficiency**: Optimized for 32-thread warp execution
- **Memory Access**: Coalesced global memory access patterns

## 🔨 Building

### Quick Build Script

```bash
#!/bin/bash
set -x
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

### Manual Building

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Falcon
   ```
2. Generate CMake building system:

   ```bash
   cmake -S . -B ./build -DCMAKE_BUILD_TYPE=Release
   ```
3. Build all targets:

   ```bash
   cmake --build ./build --config Release -j$(nproc)
   ```

## 🧪 Testing

### Test Structure

```
test/
├── baseline/          # Comparison algorithms (ALP, ndzip, elf, etc.)
├── data/             # Test datasets
├── Falcon_test_*.cu  # Main GPU test suites
└── test_*.cpp/cu     # Specific algorithm tests
```

### Running Tests

#### Basic Usage for All Tests

```bash
./test/test_${test_name} --dir ../test/data/use/
```

#### Benchmark Tests (vs Baselines)

```bash
# Main GPU implementation (1024 elements/thread)
./test/test_gpu --dir ../test/data/use/

# GPU without packing optimization
./test/test_gpu_nopack --dir ../test/data/use/

# GPU with bit-reduction optimization
./test/test_gpu_br --dir ../test/data/use/

# GPU with sparse optimization
./test/test_gpu_spare --dir ../test/data/use/
```

#### Multi-stream Performance Tests

```bash
# Multi-stream with 3-step blocking
./test/test_muti_3step_block --dir ../test/data/use/

# Multi-stream with 3-step non-blocking
./test/test_muti_3step_noblock --dir ../test/data/use/

# Optimized multi-stream
./test/test_muti_stream_opt --dir ../test/data/use/
```

### Ablation Studies

#### Encoding Strategy Ablation

- **Full Sparse**: All bit-planes use sparse storage
- **Full Dense**: All bit-planes use dense storage
- **Brute-force Error**: Inaccurate decimal place calculation
- **Standard**: Adaptive sparse/dense selection (default)

#### Pipeline Ablation

- **Single-stream**: Sequential processing
- **Blocking**: Synchronous multi-stream
- **Non-blocking**: Asynchronous multi-stream
- **Standard**: Event-driven scheduler (default)

### Complete Test Script

```bash
#!/bin/bash
set -x
cd Falcon
mkdir -p build
cd build

# Compile project
cmake ..
make -j

# Run all tests
run_test() {
    local test_name=$1
    echo "===== Running ${test_name} ====="
    ./test/test_${test_name} --dir ../test/data/use/
}

# Core GPU tests
run_test "gpu"
run_test "gpu_nopack"
run_test "gpu_br"
run_test "gpu_spare"

# Multi-stream tests
run_test "muti_3step_block"
run_test "muti_3step_noblock"
run_test "muti_stream_opt"
```

## 📊 Experimental Results

### Compression Ratio Comparison

| Method           | Average Ratio   | Improvement vs Falcon |
| ---------------- | --------------- | --------------------- |
| **Falcon** | **0.299** | -                     |
| ALP              | 0.329           | 9.1% worse            |
| Elf*             | 0.339           | 13.4% worse           |
| Elf              | 0.380           | 27.1% worse           |
| ndzip            | 0.996           | 233% worse            |

### Throughput Performance

| Operation     | Falcon     | Best Competitor      | Speedup |
| ------------- | ---------- | -------------------- | ------- |
| Compression   | 10.82 GB/s | 4.46 GB/s (GDeflate) | 2.43×  |
| Decompression | 12.32 GB/s | 5.13 GB/s (GPU:Elf*) | 2.4×   |

## 🔧 Configuration

### Default Parameters

- **Chunk Size**: 1024 or 1025 elements per thread
- **Batch Size**: 1025 × 1024 × 4 elements
- **Pipeline Streams**: 16
- **GPU Architecture**: Compute Capability 7.0+

### Chunk Size Considerations

- **1024 elements**: Aligns with power-of-two for memory addressing
- **1025 elements**: Optimized for memory space utilization
- **Thread Mapping**: Each GPU thread processes exactly one chunk

### Build Options

- `-DCMAKE_BUILD_TYPE=Release` for optimized performance
- `-DCMAKE_CUDA_ARCHITECTURES=70` for specific GPU architecture

## 📚 Citation

If you use Falcon in your research, please cite:

```bibtex
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

## 👥 Authors

- **Zheng Li** (Chongqing University) - zhengli@cqu.edu.cn
- **Weiyan Wang** (Chongqing University) - weiyan.wang@stu.cqu.edu.cn
- **Ruiyuan Li** (Chongqing University) - ruiyuan.li@cqu.edu.cn
- **Chao Chen** (Chongqing University) - cschaochen@cqu.edu.cn
- **Xianlei Long** (Chongqing University) - xianlei.long@cqu.edu.cn
- **Linjiang Zheng** (Chongqing University) - zlj_cqu@cqu.edu.cn
- **Quanqing Xu** (OceanBase, Ant Group) - xuquanqing.xqq@oceanbase.com
- **Chuanhui Yang** (OceanBase, Ant Group) - rizhao.ych@oceanbase.com

## 📄 License

This project is available for academic and research use. Please refer to the specific license terms in the repository.

## 🔗 Related Publications

- [Elf: Erasing-Based Lossless Floating-Point Compression](https://doi.org/10.14778/3594512.3594523)
- [ALP: Adaptive Lossless Floating-Point Compression](https://doi.org/10.1145/3614332)
- [Serf: Streaming Error-Bounded Floating-Point Compression](https://doi.org/10.1145/3725353)

---

**Note**: This project has been verified to work on both WSL2 (Ubuntu 22.04) and native Ubuntu 24.04 environments with the specified dependencies. For questions about specific implementations or performance characteristics, please refer to the corresponding header files and test cases.
