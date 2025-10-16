#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include "compare_utils.hpp"

// ==== 你的算法头文件 ====
#include "alp_c.h"   // CPU 版（类：ALPCompression<T>, ALPDecompression<T>）
#include "alp_gpu.hpp"           // GPU 版（命名空间 alp_gpu::{compress_*,decompress_*}）

// ===== CPU 端比特流解析：仅用于诊断打印，不依赖库内部 =====
namespace cpu_diag {
    static constexpr int DICT_BW = 3;
    static constexpr int DICT_SZ = 1 << DICT_BW;

    struct BitReader {
        const uint8_t* buf;
        uint64_t bitpos;
        inline int get1() {
            uint64_t byte = bitpos >> 3;
            int off = 7 - int(bitpos & 7ULL);
            int b = (buf[byte] >> off) & 1;
            ++bitpos; return b;
        }
        inline uint64_t getN(int bits) {
            uint64_t v=0; for(int i=0;i<bits;++i){ v=(v<<1)|get1(); } return v;
        }
        inline void skip(uint64_t bits) { bitpos += bits; }
    };

    // 解析一个“CPU 压缩块”的比特流，统计 useALP 分布，并打印前若干个 ALP 向量的 (e,f)
    inline void print_cpu_modes_one_block(const uint8_t* bytes, size_t byteCount,
                                          int block_idx, size_t max_alp_print = 8)
    {
        (void)byteCount; // 我们仅顺序读取，不强校验边界
        BitReader br{bytes, 0};
        int numVec = (int)br.getN(8);  // 行组向量数 (8 bit)

        size_t cntALP = 0, cntALPrd = 0, printed = 0;
        for (int v = 0; v < numVec; ++v) {
            int useALP = br.get1();    // 1 bit
            if (useALP) {
                ++cntALP;
                uint8_t e  = (uint8_t)br.getN(8);
                uint8_t f  = (uint8_t)br.getN(8);
                uint16_t bw= (uint16_t)br.getN(16);
                uint64_t FOR = br.getN(64);
                uint32_t n   = (uint32_t)br.getN(32);

                // if (printed < max_alp_print) {
                //     std::cout << "  [CPU-Diag] Block " << block_idx
                //               << " vec#" << v << " -> ALP(e=" << (int)e
                //               << ", f=" << (int)f << ", bitw=" << bw
                //               << ", FOR=" << (long long)FOR
                //               << ", n=" << n << ")\n";
                //     ++printed;
                // }

                // 跳过 packed 值
                br.skip((uint64_t)n * bw);
                // 异常表：个数(16) + 每个 异常值(双精度64) + 位置(16)
                uint16_t exc = (uint16_t)br.getN(16);
                br.skip((uint64_t)exc * (64 + 16));
            } else {
                ++cntALPrd;
                // ALPrd 头：n(32) + rbw(8)
                uint32_t n   = (uint32_t)br.getN(32);
                uint8_t rbw  = (uint8_t)br.getN(8);

                // 主体：每值 3bit(字典索引) + rbw(右半)
                br.skip((uint64_t)n * (DICT_BW + rbw));

                // 字典：DICT_SZ * lbw，lbw = 64 - rbw（double）
                uint8_t lbw = (uint8_t)(64 - rbw);
                br.skip((uint64_t)DICT_SZ * lbw);

                // 异常：个数(16) + 每个 lbw(异常左半) + 位置(16)
                uint16_t exc = (uint16_t)br.getN(16);
                br.skip((uint64_t)exc * (lbw + 16));
            }
        }

        std::cout << "[CPU-Diag] Block " << block_idx
                  << " vector-mode distribution: ALP=" << cntALP
                  << ", ALPrd=" << cntALPrd
                  << ", totalVec=" << numVec << "\n";
    }
} // namespace cpu_diag

// 将一段 [s,e) 按 vectorSize 切为向量组
static std::vector<std::vector<double>> make_row_group(const std::vector<double>& data,
                                                       size_t s, size_t e, int vectorSize){
    std::vector<std::vector<double>> group;
    size_t n = e - s;
    if (n == 0) return group;
    size_t numVec = (n + vectorSize - 1) / vectorSize;
    group.reserve(numVec);
    for(size_t v=0; v<numVec; ++v){
        size_t vs = s + v*vectorSize;
        size_t ve = std::min(e, vs + (size_t)vectorSize);
        group.emplace_back(data.begin()+vs, data.begin()+ve);
    }
    return group;
}

// ===== CPU 压缩/解压（按块）=====
static void run_cpu(const std::vector<double>& data, int vectorSize, int blockSize,
                    std::vector<uint8_t>& bytes_out, std::vector<double>& decoded_out,
                    double& t_comp_ms, double& t_decomp_ms)
{
    Timer tim;
    bytes_out.clear(); decoded_out.assign(data.size(), 0.0);

    // 压缩
    tim.tic();
    size_t n = data.size();
    for(size_t s=0; s<n; s += blockSize){
        size_t e = std::min(n, s + (size_t)blockSize);
        auto group = make_row_group(data, s, e, vectorSize);
        ALPCompression<double> enc(vectorSize);
        enc.entry(group);
        auto bytes = enc.getOutput();
        // 追加到总缓冲
        bytes_out.insert(bytes_out.end(), bytes.begin(), bytes.end());
    }
    t_comp_ms = tim.toc_ms();

    // 解压（再次逐块解，块边界靠顺序遍历 group 的大小恢复）
    // 注意：CPU 压缩结果没有显式块边界，这里简单按同样切分逐块重压并立刻解压
    tim.tic();
    size_t offset = 0; // 字节偏移
    for(size_t s=0; s<n; s += blockSize){
        size_t e = std::min(n, s + (size_t)blockSize);
        auto group = make_row_group(data, s, e, vectorSize);

        // 为了知道该块字节数：重新局部压缩一次（开销很小，相比全量），只用于提取该块长度
        ALPCompression<double> enc_len(vectorSize);
        enc_len.entry(group);
        auto blk_bytes = enc_len.getOutput();

        if (std::getenv("ALP_GPU_DIAG")) {
            int block_idx = int(s / (size_t)blockSize);
            cpu_diag::print_cpu_modes_one_block(bytes_out.data()+offset, blk_bytes.size(), block_idx, /*max_alp_print=*/8);
        }
        ALPDecompression<double> dec(bytes_out.data()+offset, blk_bytes.size());
        auto vecs = dec.entry();
        // 把解压向量组平铺回 decoded_out
        size_t pos = s;
        for (auto& v : vecs) {
            std::copy(v.begin(), v.end(), decoded_out.begin()+pos);
            pos += v.size();
        }
        offset += blk_bytes.size();
    }
    t_decomp_ms = tim.toc_ms();
}

// ===== GPU 压缩/解压（按块，一个线程一块由你的实现负责）=====
static void run_gpu(const std::vector<double>& data, int vectorSize, int blockSize,
                    std::vector<uint8_t>& bytes_out, std::vector<double>& decoded_out,
                    double& t_comp_ms, double& t_decomp_ms)
{
    Timer tim;
    decoded_out.assign(data.size(), 0.0);

    alp_gpu::Params p;
    p.vectorSize = vectorSize;
    p.blockSize  = blockSize;

    // 压缩
    tim.tic();
    auto c = alp_gpu::compress_double(data.data(), data.size(), p);
    t_comp_ms = tim.toc_ms();

    // ！！！关键顺序：先解压，再把字节流 move 出去
    tim.tic();
    alp_gpu::decompress_double(c, decoded_out.data(), data.size(), p);
    t_decomp_ms = tim.toc_ms();

    // 压缩字节流最后再搬走（避免额外拷贝，也不影响上面的解压）
    bytes_out = std::move(c.data);
}


static void usage(){
    std::cout << "Usage: alp_compare [--n N | --csv path] [--pattern k] [--vector V] [--block B]\n";
    std::cout << "  --n N        生成 N 个样本（与 --csv 互斥）\n";
    std::cout << "  --csv path   从 CSV 第一列读取数据\n";
    std::cout << "  --pattern k  合成数据模式(0..3)，默认0\n";
    std::cout << "  --vector V   向量长度，默认1000\n";
    std::cout << "  --block B    块大小   ，默认100000\n";
}

int main(int argc, char** argv){
    size_t N = 1'000'000; // 缺省数据量
    std::string csv;
    int pattern = 0;
    int vectorSize = 1000;
    int blockSize  = 100000;

    for(int i=1;i<argc;++i){
        std::string a = argv[i];
        auto need = [&](int more){ if(i+more>=argc) throw std::runtime_error("bad args"); };
        if(a=="--n"){ need(1); N = std::stoull(argv[++i]); }
        else if(a=="--csv"){ need(1); csv = argv[++i]; }
        else if(a=="--pattern"){ need(1); pattern = std::stoi(argv[++i]); }
        else if(a=="--vector"){ need(1); vectorSize = std::stoi(argv[++i]); }
        else if(a=="--block"){ need(1); blockSize  = std::stoi(argv[++i]); }
        else { usage(); return 1; }
    }

    std::vector<double> data = csv.empty() ? synth(N, pattern) : read_csv_first_column(csv);

    Metrics M; M.n = data.size(); M.bytes_in = data.size()*sizeof(double);

    std::vector<uint8_t> cpu_bytes, gpu_bytes;
    std::vector<double> cpu_out,   gpu_out;

    run_cpu(data, vectorSize, blockSize, cpu_bytes, cpu_out, M.cpu_comp_ms, M.cpu_decomp_ms);
    run_gpu(data, vectorSize, blockSize, gpu_bytes, gpu_out, M.gpu_comp_ms, M.gpu_decomp_ms);

    M.bytes_cpu = cpu_bytes.size();
    M.bytes_gpu = gpu_bytes.size();

    diff_stats(data, cpu_out, M.mismatches_cpu, M.mse_cpu);
    diff_stats(data, gpu_out, M.mismatches_gpu, M.mse_gpu);

    print_metrics(M);

    // 简单一致性（可选）：比较 CPU vs GPU 解压结果是否一致
    size_t mis_cpu_vs_gpu = 0; double mse_cpu_vs_gpu = 0.0;
    diff_stats(cpu_out, gpu_out, mis_cpu_vs_gpu, mse_cpu_vs_gpu);
    std::cout << "CPU vs GPU: mismatches=" << mis_cpu_vs_gpu << ", mse=" << mse_cpu_vs_gpu << "\n";

    return 0;
}