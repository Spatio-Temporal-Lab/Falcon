#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include "data/dataset_utils.hpp"
#include "Falcon_pipeline.cuh"
#include <thread>
namespace fs = std::filesystem;

std::string title = "";
int NUM_STREAM=16; // 使用16个CUDA Stream实现流水线
#define NUM_STREAMS 16 // 使用16个CUDA Stream实现流水线
// 生成随机数据的函数
std::vector<double> generate_test_data(size_t nbEle, int pattern_type = 0) {
    std::vector<double> data(nbEle);

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (pattern_type) {
        case 0: {
            // 随机数据
            std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = dist(gen);
            }
            break;
        }
        case 1: {
            // 线性增长数据
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = static_cast<double>(i) * 0.01;
            }
            break;
        }
        case 2: {
            // 正弦波数据
            for (size_t i = 0; i < nbEle; ++i) {
                data[i] = 1000.0 * sin(0.01 * i);
            }
            break;
        }
        case 3: {
            // 多步阶数据
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


CompressionInfo test_compression(ProcessedData data, size_t chunkSize);
// 准备数据函数，支持文件和生成数据两种模式
ProcessedData prepare_data(const std::string &source_path = "", size_t generate_size = 0, int pattern_type = 0,int fig=-1) {
    ProcessedData result;
    std::vector<double> data;

    // 决定数据来源
    if (generate_size > 0) {
        // 生成指定大小的数据
        // printf("生成 %zu 个元素的测试数据 (模式: %d)\n", generate_size, pattern_type);
        data = generate_test_data(generate_size, pattern_type);
        result.nbEle = generate_size;
    } else if (!source_path.empty()) {
        // 从文件读取数据
        // printf("从文件加载数据: %s\n", source_path.c_str());
        data = read_data(source_path,fig);
        result.nbEle = data.size();
    } else {
        printf("错误: 未指定数据源\n");
        result.nbEle = 0;
        result.oriData = nullptr;
        result.cmpBytes = nullptr;
        return result;
    }
    if(result.nbEle <= 0)
    {
        printf("wrong");
    }
    // 分配固定内存
    cudaCheckError(cudaHostAlloc(&result.oriData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpBytes, result.nbEle * sizeof(double), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc((void**)&result.cmpSize, sizeof(unsigned int), cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&result.decData, result.nbEle * sizeof(double), cudaHostAllocDefault));
    // 将数据拷贝到固定内存
#pragma omp parallel for
    for (size_t i = 0; i < result.nbEle; ++i) {
        result.oriData[i] = data[i];
    }

    return result;
}


CompressionInfo test_streams_compression(ProcessedData data, size_t chunkSize)
{

    FalconPipeline ex;

    CompressionResult compResult = ex.executeCompressionPipeline(data, chunkSize, NUM_STREAM);
    cudaDeviceSynchronize(); 

    PipelineAnalysis decompAnalysis = ex.executeDecompressionPipeline(compResult, data, NUM_STREAM);
    cudaDeviceSynchronize(); 


    return CompressionInfo{
        decompAnalysis.total_size,
        decompAnalysis.total_compressed_size,
        compResult.analysis.compression_ratio,
        0,
        compResult.analysis.comp_time,
        compResult.analysis.comp_throughout,
        0,
        decompAnalysis.decomp_time,
        decompAnalysis.decomp_throughout};
}


CompressionInfo test_compression(ProcessedData data, size_t chunkSize)
{
    FalconPipeline ex;


    CompressionResult compResult = ex.executeCompressionPipeline(data, chunkSize);
    cudaDeviceSynchronize(); 

    PipelineAnalysis decompAnalysis = ex.executeDecompressionPipeline(compResult, data);
    cudaDeviceSynchronize(); 


    return CompressionInfo{
        decompAnalysis.total_size,
        decompAnalysis.total_compressed_size,
        compResult.analysis.compression_ratio,
        0,
        compResult.analysis.comp_time,
        compResult.analysis.comp_throughout,
        0,
        decompAnalysis.decomp_time,
        decompAnalysis.decomp_throughout};
}

// 从文件测试多个块大小
int test_multiple_blocksizes(const std::string &file_path, const std::vector<size_t> &block_sizes_kb)
{
    printf("=================================================\n");
    printf("=====Testing Block Sizes : %d muti streams ======\n", NUM_STREAMS);
    printf("=================================================\n");

    // 对每个块大小进行测试
    for (size_t block_size_kb : block_sizes_kb)
    {
        // 将KB转换为元素数量 (double = 8 bytes)
        cudaDeviceReset();

        size_t chunkSize = (block_size_kb * 1025) / sizeof(double);

        printf("\n[测试块大小: %zu MB (%zu 元素)]\n", block_size_kb / 1024, chunkSize);

        CompressionInfo a;
        for (int i = 0; i < 3; i++)
        {
            // 每次循环重置GPU并重新准备数据
            cudaDeviceReset();

            // 重新准备CUDA资源
            ProcessedData data = prepare_data(file_path);

            // 执行测试
            auto tmp = test_compression(data, chunkSize);
            cudaDeviceSynchronize();
            a += tmp;

            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);

            // 清理本次迭代的资源
            cleanup_data(data);
            // cudaFree(0);
        }
        a = a / 3;
        a.print();

    }

    return 0;
}

// 生成数据测试多个块大小
int test_multiple_blocksizes_generated(size_t data_size_mb, const std::vector<size_t> &block_sizes_kb,
                                       int pattern_type = 0) {
    printf("=================================================\n");
    printf("=======Testing FALCON with Different Block Sizes===\n");
    printf("=================================================\n");


    // 准备数据
    // 将MB转换为元素数量 (每个元素是double类型，8字节)
    size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
    ProcessedData data = prepare_data("", nbEle, pattern_type);
    if (data.nbEle == 0) {
        printf("错误: 无法读取文件数据\n");
        return 1;
    }
    printf("GPU预热中...\n");
    size_t warmup_chunk = data.nbEle; // 单流
    FalconPipeline ex;

    ex.executeCompressionPipeline(data, warmup_chunk);
    cudaDeviceSynchronize(); // 确保预热完成


    // 检查GPU可用内存
    size_t freeMem, totalMem;
    cudaMemGetInfo(&freeMem, &totalMem);
    // size_t poolSize = freeMem * 0.4;
    // poolSize = (poolSize + 1024 * 2 * sizeof(double) - 1) & ~(1024 * 2 * sizeof(double) - 1); // 向上对齐

    std::vector<PipelineAnalysis> results;

    // 对每个块大小进行测试
    for (size_t block_size_kb: block_sizes_kb) {
        // 将KB转换为元素数量 (double = 8 bytes)
        size_t chunkSize = (block_size_kb * 1024) / sizeof(double);

        printf("\n[测试块大小: %zu KB (%zu 元素)]\n", block_size_kb, chunkSize);

        // 执行压缩并获取结果
        for(int i=0;i<3;i++)
        {
            FalconPipeline ex;
            CompressionResult compResult = ex.executeCompressionPipeline(data, chunkSize);
            PipelineAnalysis result = compResult.analysis;
            results.push_back(result);
        }

        // 稍微延迟一下，让GPU冷却
        std::this_thread::sleep_for(std::chrono::milliseconds(500*3));
    }

    // 可视化块大小与阶段时间的关系
    // visualize_stage_timing_relationship(results);

    // 输出CSV数据以便外部绘图分析
    // output_blocksize_timing_csv(results, "block_size_timing_analysis_generated.csv");

    // 清理资源
    cleanup_data(data);

    return 0;
}

// 生成二次方增长的块大小序列
std::vector<size_t> generate_power2_blocksizes(size_t min_kb, size_t max_kb) {
    std::vector<size_t> sizes;
    for (size_t size = min_kb; size <= max_kb; size *= 2) {
        sizes.push_back(size);
    }
    return sizes;
}

// 生成线性增长的块大小序列
std::vector<size_t> generate_linear_blocksizes(size_t min_kb, size_t max_kb, size_t step_kb) {
    std::vector<size_t> sizes;
    for (size_t size = min_kb; size <= max_kb; size += step_kb) {
        sizes.push_back(size);
    }
    return sizes;
}


void warmup()
{
        size_t nbEle = (1*1024 * 1024) / sizeof(double);
        ProcessedData data = prepare_data("", nbEle, 0);
        if (data.nbEle == 0) {
            printf("错误: 无法读取文件数据\n");
            return ;
        }
        printf("GPU预热中...\n");
        size_t warmup_chunk = data.nbEle; // 单流
        FalconPipeline ex;
        ex.executeCompressionPipeline(data, warmup_chunk);
        cudaDeviceSynchronize(); // 确保预热完成
}

int setChunk(int nbEle)
{
    size_t chunkSize=1025;
    size_t temp=nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
    //可用的显存
    // size_t availableMemory = getAvailableGPUMemory();
    size_t availableMemory, totalMem;
    cudaMemGetInfo(&availableMemory, &totalMem);
    // size_t limit=availableMemory/(4 * NUM_STREAMS * sizeof(double) * 2);
    size_t limit=64*1024*1024/sizeof(double);
    //最多同时有16流 chunkSize*NUM_STREAMS*8*sizeof(double) * 2<availableMemory/8*
    while(chunkSize<=limit//MAX_NUMS_PER_CHUNK l
            && chunkSize<=temp)
    {
        chunkSize*=2;
    }
    chunkSize=chunkSize>limit?chunkSize/2:chunkSize;
    printf("chunkSize:%zu MB\n",chunkSize*sizeof(double)/1024/1025);
    return chunkSize;
}

// 主要测试函数 - 支持文件路径或生成数据
int test(const std::string &file_path = "", size_t data_size_mb = 0, int pattern_type = 0) {
    // warmup();
    cudaDeviceReset();

    if (!file_path.empty()) {
        std::vector<double> file_data = read_data(file_path);
        size_t nbEle = file_data.size();
        
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            // 每次循环重置GPU并重新准备数据
            cudaDeviceReset();
            
            // 重新准备CUDA资源
            ProcessedData data = prepare_data(file_path);
            
            // 计算chunk大小
            size_t chunkSize = setChunk(data.nbEle);
            
            // 执行测试
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
            
            // 清理本次迭代的资源
            cleanup_data(data);
        }
        a=a/3;
        a.print();
        // cleanup_data(data);
        return 0;
    } else if (data_size_mb > 0) {
        size_t nbEle = (data_size_mb * 1024 * 1024) / sizeof(double);
        
        // ProcessedData data = prepare_data("", nbEle, pattern_type);
        // chunkSize=setChunk(data.nbEle);
        // CompressionInfo a;
        // for(int i=0;i<3;i++)
        // {
        //     a+=test_compression(data,chunkSize);
        // }
        // a=a/3;
        // a.print();
        // cleanup_data(data);
        CompressionInfo a;
        for(int i = 0; i < 3; i++) {
            // 每次循环重置GPU并重新准备数据
            cudaDeviceReset();
            
            // 重新准备CUDA资源
            ProcessedData data = prepare_data("", nbEle, pattern_type);
            
            // 计算chunk大小
            size_t chunkSize = setChunk(data.nbEle);
            
            // 执行测试
            auto tmp = test_compression(data, chunkSize);
            a += tmp;
            
            // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
            
            // 清理本次迭代的资源
            cleanup_data(data);
        }
        a=a/3;
        a.print();
        return 0;
    } else {
        printf("错误: 必须提供文件路径或数据生成参数\n");
        return 1;
    }
}

int main(int argc, char *argv[]) {
    cudaSetDevice(0);
    if (argc < 2) {
        printf("使用方法:\n");
        printf("  %s --file <file_path> : 从文件测试\n", argv[0]);
        printf("  %s --dir <directory_path> : 测试目录中所有文件\n", argv[0]);
        printf("  %s --generate <size_in_mb> [pattern_type] : 生成数据测试\n", argv[0]);
        printf("    pattern_type: 0=随机数据, 1=线性增长, 2=正弦波, 3=阶梯\n");
        printf("  %s --analyze-blocks <file_path> : 分析不同块大小的性能\n", argv[0]);
        printf("  %s --analyze-blocks-gen <size_in_mb> [pattern_type] : 使用生成数据分析不同块大小的性能\n", argv[0]);
        return 1;
    }

    std::string arg = argv[1];

    if (arg == "--file" && argc >= 3) {
        std::string file_path = argv[2];
        test(file_path);
    } else if (arg == "--dir" && argc >= 3) {
        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path)) {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        warmup();
        for (const auto &entry: fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                test(file_path);
                std::cout << "---------------------------------------------" << std::endl;
            }
        }
    }
    else if(arg == "--file-beta" && argc >= 3){
        std::string file_path = argv[2];
        warmup();
        for(int beta=16;beta<18;beta++)
        {
            std::cout << "\n正在处理文件: " << file_path << " beta :" << beta << std::endl;
            
            CompressionInfo a;
            for(int i = 0; i < 3; i++) {
                // 每次循环重置GPU并重新准备数据
                cudaDeviceReset();
                
                // 重新准备CUDA资源
                ProcessedData data = prepare_data(file_path,0,0,beta);
                
                // 计算chunk大小
                size_t chunkSize = setChunk(data.nbEle);
                
                // 执行测试
                auto tmp = test_compression(data, chunkSize);
                a += tmp;
                
                // printf("Iteration %d - a:%.6f, get:%.6f\n", i+1, a.compression_ratio/(i+1), tmp.compression_ratio);
                
                // 清理本次迭代的资源
                cleanup_data(data);
            }
            a=a/3;
            a.print();
            std::cout << "---------------------------------------------" << std::endl;

        }

        // cleanup_data(data);

    } 
    else if (arg == "--generate" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

        test("", data_size_mb, pattern_type);
    } else if (arg == "--analyze-blocks" && argc >= 3) {
        std::string file_path = argv[2];
        title = "analyze-blocks " + file_path;
        // 创建不同大小的块序列
        // 从16mbB到512MB，以二次方增长
        std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024/4, 2*64*1024);

        test_multiple_blocksizes(file_path, block_sizes);
    } else if (arg == "--analyze-blocks-gen" && argc >= 3) {
        size_t data_size_mb = std::stoul(argv[2]);
        int pattern_type = (argc >= 4) ? std::stoi(argv[3]) : 0;

        std::string pattern_str = (argc >= 4) ? argv[3] : "0";
        title = std::string("analyze-blocks-gen ") + argv[2] + " " + pattern_str;

        // 创建不同大小的块序列
        // 从16mbB到512MB，以二次方增长
        std::vector<size_t> block_sizes = generate_power2_blocksizes(16*1024, 8*64*1024);

        test_multiple_blocksizes_generated(data_size_mb, block_sizes, pattern_type);
    }else if (arg == "--analyze-blocks-dir" && argc >= 3)
    {
        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path))
        {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }
        // 从16mbB到512MB，以二次方增长
        std::vector<size_t> block_sizes = generate_power2_blocksizes(4*1024, 64*1024);
        // std::vector<size_t> block_sizes = {32 * 1024};
        // int streams[5]={1,4,8,16,32};
        for (const auto &entry : fs::directory_iterator(dir_path))
        {
            if (entry.is_regular_file())
            {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                // for(int i=0;i<5;i++)
                // {
                //     NUM_STREAM=streams[i];
                    test_multiple_blocksizes(file_path, block_sizes);
                    std::cout << "---------------------------------------------" << std::endl;
                // }
            }
        }
    } 
    else if(arg == "--analyze-streams-dir" && argc >= 3){
        std::string dir_path = argv[2];

        // 检查目录是否存在
        if (!fs::exists(dir_path))
        {
            std::cerr << "指定的数据目录不存在: " << dir_path << std::endl;
            return 1;
        }

        int streams[5]={1,4,8,16,32};


        for (const auto &entry : fs::directory_iterator(dir_path))
        {
            if (entry.is_regular_file())
            {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << std::endl;
                for(int i=0;i<5;i++)
                {
                    NUM_STREAM=streams[i];
                    printf("=================================================\n");
                    printf("=====Testing Stream : %d muti streams ======\n", NUM_STREAM);
                    printf("=================================================\n");
                    CompressionInfo a;
                    for(int i = 0; i < 3; i++) {
                        // 每次循环重置GPU并重新准备数据
                        cudaDeviceReset();
                        
                        // 重新准备CUDA资源
                        ProcessedData data = prepare_data(file_path);
                        
                        // 计算chunk大小
                        size_t chunkSize = setChunk(data.nbEle);
                        
                        // 执行测试
                        auto tmp = test_streams_compression(data, chunkSize);
                        a += tmp;

                        cleanup_data(data);
                    }
                    a=a/3;
                    a.print();
                    std::cout << "---------------------------------------------" << std::endl;
                }
            }
        }

    }
    else {
        printf("无效的参数. 使用 %s 查看用法\n", argv[0]);
        return 1;
    }

    return 0;
}

