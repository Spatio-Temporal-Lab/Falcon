#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuSZp.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include "data/dataset_utils.hpp"
namespace fs = std::filesystem;

#define NUM_STREAMS 16 // 使用3个CUDA Stream实现流水线
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

cudaEvent_t global_start_event, global_end_event;//全局开始和结束事件

// 高精度计时器
class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
public:
    void Start() { start = std::chrono::high_resolution_clock::now(); }
    float Elapsed() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(end - start).count();
    }
};

//内存池
class MemoryPool
{
public:
    MemoryPool(size_t poolSize, size_t chunkSize) : poolSize(poolSize), chunkSize(chunkSize)
    {
        cudaMalloc(&pool, poolSize); // 为整个内存池分配显存
        freeList = (void**)malloc(poolSize / chunkSize * sizeof(void*)); // 用于管理空闲块
        freeBlockCount = poolSize / chunkSize;

        // 初始化空闲块列表
        for (size_t i = 0; i < freeBlockCount; ++i)
        {
            freeList[i] = (void*)((char*)pool + i * chunkSize);
        }
    }

    // 从内存池中获取内存
    void* allocate()
    {
        if (freeBlockCount == 0)
        {
            std::cerr << "No available memory blocks in the pool!" << std::endl;
            return nullptr;
        }
        void* block = freeList[--freeBlockCount]; // 获取一个空闲块
        return block;
    }

    // 归还内存块
    void deallocate(void* block)
    {
        if (freeBlockCount >= poolSize / chunkSize)
        {
            std::cerr << "Memory pool is full!" << std::endl;
            return;
        }
        freeList[freeBlockCount++] = block; // 将块放回空闲列表
    }

    ~MemoryPool()
    {
        cudaFree(pool); // 释放内存池
        free(freeList); // 释放空闲列表
    }

private:
    void* pool{}; // 内存池的起始地址
    void** freeList; // 空闲块链表
    size_t poolSize; // 内存池的大小
    size_t chunkSize; // 每个内存块的大小
    size_t freeBlockCount; // 当前空闲块的数量
};

// 流水线每个阶段的计时事件
struct StreamTiming {
    cudaEvent_t start_event, h2d_event, comp_event, d2h_event, end_event;
    float begin_time = 0;
    float h2d_time = 0;
    float comp_time = 0;
    float d2h_time = 0;
    float total_time = 0;
    float end_time = 0;
};

// 流水线总体性能分析
struct PipelineAnalysis {
    float total_h2d = 0;
    float total_comp = 0;
    float total_d2h = 0;
    float end_time = 0;
    float sequential_time = 0;
    float speedup = 0;
};

// 流水线操作类，负责管理单个数据块的处理
class PipelineOperator {
private:
    int index;
    cudaStream_t stream;
    StreamTiming timing;
    
public:
    PipelineOperator() {
        // 创建流和事件
        cudaCheckError(cudaStreamCreate(&stream));
        cudaCheckError(cudaEventCreate(&timing.start_event));
        cudaCheckError(cudaEventCreate(&timing.h2d_event));
        cudaCheckError(cudaEventCreate(&timing.comp_event));
        cudaCheckError(cudaEventCreate(&timing.d2h_event));
        cudaCheckError(cudaEventCreate(&timing.end_event));
    }
    
    ~PipelineOperator() {
        // 销毁流和事件
        cudaStreamDestroy(stream);
        cudaEventDestroy(timing.start_event);
        cudaEventDestroy(timing.h2d_event);
        cudaEventDestroy(timing.comp_event);
        cudaEventDestroy(timing.d2h_event);
        cudaEventDestroy(timing.end_event);
    }
    
    void set_index(int idx) { index = idx; }
    int get_index() const { return index; }
    cudaStream_t get_stream() const { return stream; }
    
    // 执行一次异步操作（H2D -> Compression -> D2H）
    void process_chunk(
        double* h_data, unsigned char* h_cmpBytes, size_t& cmpSize,
        double* d_data, unsigned char* d_cmpBytes,
        size_t chunkEle, size_t offset) {
        
        // 记录开始事件
        cudaCheckError(cudaEventRecord(timing.start_event, stream));
        
        // 主机到设备的拷贝 (H2D)
        cudaCheckError(cudaMemcpyAsync(d_data, h_data + offset, chunkEle * sizeof(double), 
                                      cudaMemcpyHostToDevice, stream));
        cudaCheckError(cudaEventRecord(timing.h2d_event, stream));
        
        // 执行压缩计算
        size_t chunkCmpSize = 0;
        GDFC_compress_plain_f64(d_data, d_cmpBytes, chunkEle, &chunkCmpSize, stream);
        cudaCheckError(cudaEventRecord(timing.comp_event, stream));
        
        // 设备到主机的拷贝 (D2H) - 压缩结果
        cudaCheckError(cudaMemcpyAsync(h_cmpBytes + cmpSize, d_cmpBytes, 
                                      chunkCmpSize * sizeof(unsigned char), 
                                      cudaMemcpyDeviceToHost, stream));
        cudaCheckError(cudaEventRecord(timing.d2h_event, stream));
        
        // 记录结束事件
        cudaCheckError(cudaEventRecord(timing.end_event, stream));
        
        // 更新总压缩大小
        cmpSize += chunkCmpSize;
    }
    
    // 计算并返回每个阶段的时间
    StreamTiming calculate_timing() {
        float tmp;
        // 计算在全局的起始时间
        cudaCheckError(cudaEventElapsedTime(&tmp, global_start_event, timing.start_event));
        timing.begin_time = tmp;

        // 计算H2D时间
        cudaCheckError(cudaEventElapsedTime(&tmp, timing.start_event, timing.h2d_event));
        timing.h2d_time = tmp;
        
        // 计算压缩时间
        cudaCheckError(cudaEventElapsedTime(&tmp, timing.h2d_event, timing.comp_event));
        timing.comp_time = tmp;
        
        // 计算D2H时间
        cudaCheckError(cudaEventElapsedTime(&tmp, timing.comp_event, timing.d2h_event));
        timing.d2h_time = tmp;
        
        // 计算总时间
        cudaCheckError(cudaEventElapsedTime(&tmp, timing.start_event, timing.end_event));
        timing.total_time = tmp;
        
        // 计算在全局中结束时间
        timing.end_time = timing.total_time+timing.begin_time;
        return timing;
    }
};

// 分析流水线性能
void analyze_pipeline(const std::vector<StreamTiming>& timings, PipelineAnalysis& analysis) {
    analysis.total_h2d = 0;
    analysis.total_comp = 0;
    analysis.total_d2h = 0;
    analysis.end_time = 0;
    
    // 计算各阶段总时间和最大总时间
    for (const auto& timing : timings) {
        analysis.total_h2d += timing.h2d_time;
        analysis.total_comp += timing.comp_time;
        analysis.total_d2h += timing.d2h_time;
        analysis.end_time = std::max(analysis.end_time, timing.end_time);
    }
    
    // 计算顺序执行理论时间和加速比
    analysis.sequential_time = analysis.total_h2d + analysis.total_comp + analysis.total_d2h;
    analysis.speedup = analysis.sequential_time / analysis.end_time;
    
    // 打印流水线分析结果
    printf("\n===== 流水线执行分析 =====\n");
    printf("各阶段总时间:\n");
    printf("- H2D总时间: %.2f ms\n", analysis.total_h2d);
    printf("- 压缩计算总时间: %.2f ms\n", analysis.total_comp);
    printf("- D2H总时间: %.2f ms\n", analysis.total_d2h);
    printf("- 顺序执行理论时间: %.2f ms\n", analysis.sequential_time);
    printf("- 实际并行执行时间: %.2f ms\n", analysis.end_time);
    printf("- 流水线加速比: %.2fx\n", analysis.speedup);
}

// 可视化流水线执行时间线
void visualize_timeline(const std::vector<StreamTiming>& timings) {
    printf("\n===== 流水线时间线可视化 =====\n");
    printf("图例: |H2D|->|Comp|->|D2H|\n\n");
    
    // 查找最长的执行时间
    float max_time = 0;
    for (const auto& timing : timings) {
        max_time = std::max(max_time, timing.total_time);
    }
    
    const int TIME_SCALE = 2; // 每毫秒显示的字符数
    
    // 为每个流/操作显示时间线
    for (size_t i = 0; i < timings.size(); i++) {
        const auto& timing = timings[i];
        
        // 计算各阶段的开始和结束位置
        float h2d_start = 0;
        float h2d_end = timing.h2d_time;
        float comp_start = h2d_end;
        float comp_end = comp_start + timing.comp_time;
        float d2h_start = comp_end;
        float d2h_end = d2h_start + timing.d2h_time;
        
        // 打印操作ID和流ID
        printf("操作%2zu(流%zu): ", i, i % NUM_STREAMS);
        
        //绘制全局前偏移
        int beg_len = std::max(1, (int)(timing.begin_time * TIME_SCALE));
        for (int j = 0; j < beg_len; j+=2) printf(" ");
        // 绘制H2D
        printf("|H2D|");
        int h2d_len = std::max(1, (int)(timing.h2d_time * TIME_SCALE));
        for (int j = 0; j < h2d_len; j+=2) printf("=");
        
        // 绘制Comp
        printf("|Comp|");
        int comp_len = std::max(1, (int)(timing.comp_time * TIME_SCALE));
        for (int j = 0; j < comp_len; j+=2) printf("~");
        
        // 绘制D2H
        printf("|D2H|");
        int d2h_len = std::max(1, (int)(timing.d2h_time * TIME_SCALE));
        for (int j = 0; j < d2h_len; j+=2) printf("-");
        
        printf(" (%.2f ms)\n", timing.total_time);
    }
}

// 生成随机数据的函数
std::vector<double> generate_test_data(size_t nbEle, int pattern_type = 0) {
    std::vector<double> data(nbEle);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    switch(pattern_type) {
        case 0: { // 随机数据
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
            break;
        }
        case 1: { // 线性增长数据
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<double>(i) * 0.01;
            }
            break;
        }
        case 2: { // 正弦波数据
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = 1000.0 * sin(0.01 * i);
            }
            break;
        }
        case 3: { // 多步阶数据
            int step_size = nbEle / 10;
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<double>((i / step_size) * 100);
            }
            break;
        }
        default: {
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
        }
    }
    
    return data;
}

// 数据预处理函数
struct ProcessedData {
    double* oriData;
    unsigned char* cmpBytes;
    size_t nbEle;
};

// 准备数据函数，支持文件和生成数据两种模式
ProcessedData prepare_data(const std::string& source_path = "", size_t generate_size = 0, int pattern_type = 0) {
    ProcessedData result;
    std::vector<double> data;
    
    // 决定数据来源
    if (generate_size > 0) {
        // 生成指定大小的数据
        printf("生成 %zu 个元素的测试数据 (模式: %d)\n", generate_size, pattern_type);
        data = generate_test_data(generate_size, pattern_type);
        result.nbEle = generate_size;
    } else if (!source_path.empty()) {
        // 从文件读取数据
        printf("从文件加载数据: %s\n", source_path.c_str());
        data = read_data(source_path);
        result.nbEle = data.size();
    } else {
        printf("错误: 未指定数据源\n");
        result.nbEle = 0;
        result.oriData = nullptr;
        result.cmpBytes = nullptr;
        return result;
    }
    
    // 分配固定内存
    cudaCheckError(cudaHostAlloc(&result.oriData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpBytes, result.nbEle * sizeof(double) * 2, cudaHostAllocDefault));
    
    // 将数据拷贝到固定内存
    #pragma omp parallel for
    for(size_t i = 0; i < result.nbEle; ++i) {
        result.oriData[i] = data[i];
    }
    
    return result;
}

// 清理资源函数
void cleanup_data(ProcessedData& data) {
    if (data.oriData != nullptr) {
        cudaFreeHost(data.oriData);
        data.oriData = nullptr;
    }
    
    if (data.cmpBytes != nullptr) {
        cudaFreeHost(data.cmpBytes);
        data.cmpBytes = nullptr;
    }
}

// 设置GPU内存池函数
size_t setup_gpu_memory_pool(size_t nbEle, size_t& chunkSize) {
    // 检查GPU可用内存
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    std::cout << "GPU可用内存: " << freeMem / (1024*1024) << " MB / 总内存: " 
              << totalMem / (1024*1024) << " MB" << std::endl;
    
    // 设置内存池大小
    size_t poolSize = freeMem * 0.4;
    poolSize = (poolSize + 1024*2*sizeof(double)-1) & ~(1024*2*sizeof(double)-1);  // 向上对齐
    chunkSize = poolSize * 0.5 * 0.5 / sizeof(double);
    
    // 设置分块大小为2的幂，以优化对齐
    int temp = 2;
    while (temp*2 < chunkSize) {
        temp *= 2;
    }
    chunkSize = temp/2;//调整
    
    return poolSize;
}

// 执行压缩流程函数
void execute_pipeline(ProcessedData& data, size_t chunkSize, size_t poolSize) {
    // 创建时间线记录事件
    cudaEventCreate(&global_start_event);  
    cudaEventCreate(&global_end_event);  
    
    // 创建内存池
    MemoryPool ori_data_pool(poolSize, chunkSize * sizeof(double)); // 原始数据显存池
    MemoryPool cmp_bytes_pool(poolSize, chunkSize * sizeof(double)); // 压缩结果显存池
    
    std::cout << "数据总大小: " << data.nbEle * sizeof(double) / (1024*1024) << " MB" << std::endl;
    std::cout << "块大小: " << chunkSize * sizeof(double) / (1024*1024) << " MB" << std::endl;
    
    // 创建流水线操作器
    std::vector<PipelineOperator> operators(NUM_STREAMS);
    
    // 创建流水线分析对象
    std::vector<StreamTiming> timings;
    PipelineAnalysis analysis;
    
    // 启动计时器
    Timer timer;
    timer.Start();
    
    // 执行流水线处理
    size_t processedEle = 0;
    int chunkIndex = 0;
    size_t cmpSize = 0;
    
    while (processedEle < data.nbEle) {
        if (processedEle == 0) {  
            cudaEventRecord(global_start_event);  //全局时间记录
        }  
        // 计算当前块的大小
        size_t chunkEle = (data.nbEle - processedEle) > chunkSize ? chunkSize : (data.nbEle - processedEle);
        if (chunkEle == 0) break;
        
        // 选择当前流
        int streamIdx = chunkIndex % NUM_STREAMS;
        
        // 分配GPU内存
        double* d_oriData = (double*)ori_data_pool.allocate();
        unsigned char* d_cmpBytes = (unsigned char*)cmp_bytes_pool.allocate();
        
        if (!d_oriData || !d_cmpBytes) {
            std::cerr << "内存池分配失败!" << std::endl;
            break;
        }
        
        std::cout << "处理数据块 " << chunkIndex << " (流 " << streamIdx << "): " 
                  << processedEle << " 到 " << (processedEle + chunkEle) 
                  << " 块大小: " << chunkEle << std::endl;
        
        // 设置当前块的索引
        operators[streamIdx].set_index(chunkIndex);
        
        // 处理当前数据块（异步操作）
        operators[streamIdx].process_chunk(
            data.oriData, data.cmpBytes, cmpSize,
            d_oriData, d_cmpBytes,
            chunkEle, processedEle
        );
        
        // 处理下一个数据块
        processedEle += chunkEle;
        chunkIndex++;
        
        // 若已经处理了NUM_STREAMS个数据块，则需要等待最早的流完成
        if (chunkIndex >= NUM_STREAMS && chunkIndex % NUM_STREAMS == 0) {
            // 等待每个流完成一轮操作
            for (int i = 0; i < NUM_STREAMS; i++) {
                cudaCheckError(cudaStreamSynchronize(operators[i].get_stream()));
                
                // 计算并记录时间
                StreamTiming timing = operators[i].calculate_timing();
                timings.push_back(timing);
                
                // 释放资源回内存池（已经在流中完成的操作所使用的内存）
                int completedChunkIdx = chunkIndex - NUM_STREAMS + i;
                int chunkOffset = completedChunkIdx * chunkSize;
                if (chunkOffset < data.nbEle) {
                    size_t completedChunkSize = (data.nbEle - chunkOffset) > chunkSize ? 
                                              chunkSize : (data.nbEle - chunkOffset);
                }
            }
        }
    }
    
    // 等待所有未完成的流操作
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaStreamSynchronize(operators[i].get_stream()));
        
        // 收集最后一批操作的时间数据
        if ((chunkIndex - 1) % NUM_STREAMS >= i) {
            StreamTiming timing = operators[i].calculate_timing();
            timings.push_back(timing);
        }
    }
    cudaEventRecord(global_end_event);
    
    // 记录总执行时间
    float totalTime = timer.Elapsed();
    
    // 分析流水线性能
    analyze_pipeline(timings, analysis);
    
    // 可视化时间线
    visualize_timeline(timings);
    
    // 输出统计信息
    printf("\n===== 压缩统计 =====\n");
    printf("压缩大小: %lu 字节\n", cmpSize);
    printf("原始大小: %lu 字节\n", data.nbEle * sizeof(double));
    printf("压缩比: %.2f\n", static_cast<double>(data.nbEle * sizeof(double)) / cmpSize);
    
    printf("\n===== 性能统计 =====\n");
    printf("端到端总时间: %.2f ms\n", totalTime);
    printf("压缩吞吐量: %.2f GB/s\n", 
           (data.nbEle * sizeof(double) / 1024.0 / 1024.0 / 1024.0) / (totalTime / 1000.0));
    printf("流水线效率: %.2f%%\n", analysis.speedup * 100.0 / NUM_STREAMS);
    
    // 清理事件
    cudaEventDestroy(global_start_event);
    cudaEventDestroy(global_end_event);
}

// 从文件测试函数
int test_from_file(const std::string& file_path) {
    printf("=================================================\n");
    printf("=========Testing GDFC with File Data============\n");
    printf("=================================================\n");
    
    // 准备数据
    ProcessedData data = prepare_data(file_path);
    if (data.nbEle == 0) {
        printf("错误: 无法读取文件数据\n");
        return 1;
    }
    
    // 设置GPU内存
    size_t chunkSize;
    size_t poolSize = setup_gpu_memory_pool(data.nbEle, chunkSize);
    
    // 执行压缩
    execute_pipeline(data, chunkSize, poolSize);
    
    // 清理资源
    cleanup_data(data);
    
    return 0;
}

// 生成数据测试函数
int test_with_generated_data(size_t data_size_mb, int pattern_type = 0) {
    printf("=================================================\n");
    printf("=======Testing GDFC with Generated Data=========\n");
    printf("=================================================\n");
    
    // 将MB转换为元素数量 (每个元素是double类型，8字节)
    size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
    
    // 准备数据
    ProcessedData data = prepare_data("", nbEle, pattern_type);
    if (data.nbEle == 0) {
        printf("错误: 无法生成测试数据\n");
        return 1;
    }
    
    // 设置GPU内存
    size_t chunkSize;
    size_t poolSize = setup_gpu_memory_pool(data.nbEle, chunkSize);
    
    // 执行压缩
    execute_pipeline(data, chunkSize, poolSize);
    
    // 清理资源
    cleanup_data(data);
    
    return 0;
}

// 主要测试函数 - 支持文件路径或生成数据
int test(const std::string& file_path = "", size_t data_size_mb = 0, int pattern_type = 0) {
    if (!file_path.empty()) {
        return test_from_file(file_path);
    } else if (data_size_mb > 0) {
        return test_with_generated_data(data_size_mb, pattern_type);
    } else {
        printf("错误: 必须提供文件路径或数据生成参数\n");
        return 1;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2) {
        printf("使用方法:\n");
        printf("  %s --file <file_path> : 从文件测试\n", argv[0]);
        printf("  %s --dir <directory_path> : 测试目录中所有文件\n", argv[0]);
        printf("  %s --generate <size_in_mb> [pattern_type] : 生成数据测试\n", argv[0]);
        printf("    pattern_type: 0=随机数据, 1=线性增长, 2=正弦波, 3=阶梯\n");
        return 1;
    }
    
    std::string arg = argv[1];
    
    if (arg == "--file" && argc >= 3) {
        std::string file_path = argv[2];
        test(file_path);
    }
    else if (arg == "--dir" && argc >= 3) {
        std::string dir_path = argv[2];
        
        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                test(file_path);
                std::cout << "---------------------------------------------" << std::endl;
            }
        }
    }
    else if (arg == "--generate" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;
        
        test("", data_size_mb, pattern_type);
    }
    else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }
    
    return 0;
}