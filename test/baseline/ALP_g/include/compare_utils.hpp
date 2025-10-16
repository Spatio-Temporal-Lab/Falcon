#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>

struct Metrics {
    size_t n = 0;                      // 元素个数
    size_t bytes_in = 0;               // 原始字节
    size_t bytes_cpu = 0;              // CPU 压缩字节
    size_t bytes_gpu = 0;              // GPU 压缩字节
    double cpu_comp_ms = 0.0;          // CPU 压缩耗时
    double cpu_decomp_ms = 0.0;        // CPU 解压耗时
    double gpu_comp_ms = 0.0;          // GPU 压缩耗时（含 H2D/D2H）
    double gpu_decomp_ms = 0.0;        // GPU 解压耗时（含 D2H）
    size_t mismatches_cpu = 0;         // CPU 解压与原始不匹配个数（按阈值）
    size_t mismatches_gpu = 0;         // GPU 解压与原始不匹配个数（按阈值）
    double mse_cpu = 0.0;              // 均方误差
    double mse_gpu = 0.0;              // 均方误差
};

// 简单计时器
struct Timer {
    using clk = std::chrono::high_resolution_clock;
    clk::time_point t0;
    void tic() { t0 = clk::now(); }
    double toc_ms() const {
        auto t1 = clk::now();
        return std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// CSV 读取：第一列数值
inline std::vector<double> read_csv_first_column(const std::string& path) {
    std::ifstream fin(path);
    if (!fin) throw std::runtime_error("cannot open: "+path);
    std::vector<double> out; out.reserve(1<<20);
    std::string line;
    while (std::getline(fin, line)) {
        if(line.empty()) continue;
        std::stringstream ss(line);
        std::string tok;
        if (std::getline(ss, tok, ',')) {
            try { out.push_back(std::stod(tok)); } catch(...){}
        }
    }
    return out;
}

// 合成数据：多种模式
inline std::vector<double> synth(size_t n, int pattern = 0, unsigned seed = 42) {
    std::vector<double> x(n); std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::normal_distribution<double> G(0.0,1.0);
    switch(pattern){
        case 0: // 线性+弱噪声
            for(size_t i=0;i<n;++i) x[i] = 0.001*i + 0.01*G(rng);
            break;
        case 1: // 正弦+漂移
            for(size_t i=0;i<n;++i) x[i] = std::sin(0.001*i) + 0.0005*i + 0.01*G(rng);
            break;
        case 2: // 分段常数+少量异常
            for(size_t i=0;i<n;++i){
                double base = (i/1000)%2 ? 10.0 : 10.5;
                x[i] = base + 0.005*G(rng);
                if (U(rng)<0.0008) x[i] += (U(rng)<0.5? -50.0:50.0); // outlier
            }
            break;
        case 3: // 纯高斯
            for(size_t i=0;i<n;++i) x[i] = G(rng);
            break;
        default:
            for(size_t i=0;i<n;++i) x[i] = U(rng);
    }
    return x;
}

// 误差统计
inline void diff_stats(const std::vector<double>& a, const std::vector<double>& b,
                       size_t& mismatches, double& mse, double atol=0){
    mismatches = 0; mse = 0.0;
    size_t n = a.size();
    for(size_t i=0;i<n;++i){
        double d = a[i]-b[i];
        if (std::fabs(d) > atol) ++mismatches;
        mse += d*d;
    }
    mse /= std::max<size_t>(1,n);
}

// 写一行结果
inline void print_metrics(const Metrics& m){
    auto mb = [](size_t bytes){ return double(bytes)/(1024.0*1024.0); };
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "N = " << m.n << " (" << mb(m.bytes_in) << " MB)\n";
    std::cout << "CPU:  "
              << "comp=" << m.cpu_comp_ms << " ms, decomp=" << m.cpu_decomp_ms << " ms, "
              << "size=" << mb(m.bytes_cpu) << " MB, ratio=" << (double)m.bytes_cpu/m.bytes_in << ", "
              << "mse=" << m.mse_cpu << ", mismatches=" << m.mismatches_cpu << "\n";
    std::cout << "GPU:  "
              << "comp=" << m.gpu_comp_ms << " ms, decomp=" << m.gpu_decomp_ms << " ms, "
              << "size=" << mb(m.bytes_gpu) << " MB, ratio=" << (double)m.bytes_gpu/m.bytes_in << ", "
              << "mse=" << m.mse_gpu << ", mismatches=" << m.mismatches_gpu << "\n";
}
