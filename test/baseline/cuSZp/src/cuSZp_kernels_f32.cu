#include "cuSZp_kernels_f32.h"

#include <stdint.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <thrust/device_vector.h>

__device__ inline int quantization(float data, float recipPrecision)
{
    int result;
    asm("{\n\t"
        ".reg .f32 dataRecip;\n\t"
        ".reg .f32 temp1;\n\t"
        ".reg .s32 s;\n\t"
        ".reg .pred p;\n\t"
        "mul.f32 dataRecip, %1, %2;\n\t"
        "setp.ge.f32 p, dataRecip, -0.5;\n\t"
        "selp.s32 s, 0, 1, p;\n\t"
        "add.f32 temp1, dataRecip, 0.5;\n\t"
        "cvt.rzi.s32.f32 %0, temp1;\n\t"
        "sub.s32 %0, %0, s;\n\t"
        "}": "=r"(result) : "f"(data), "f"(recipPrecision)
    );
    return result;
}


__device__ inline int get_bit_num(unsigned int x)
{
    int leading_zeros;
    asm("clz.b32 %0, %1;" : "=r"(leading_zeros) : "r"(x));
    return 32 - leading_zeros;
}

#define BLOCK_SIZE_G 32
#define POW_NUM_G ((1L << 51) + (1L << 52))
#define DATA_PER_THREAD 1024

#define DATA_PER_ONE 32

#define MAX_BITCOUNT 64
#define MAX_BITSIZE_PER_BLOCK (64 + 64 + 8 + 8 + 64 + (DATA_PER_THREAD) * MAX_BITCOUNT)
#define MAX_BYTES_PER_BLOCK ((MAX_BITSIZE_PER_BLOCK + 7) / 8)

// pow10_table 和 POW_NUM_G
__constant__ double pow10_table[17] = {
    1.0,                    // 10^0
    10.0,                   // 10^1
    100.0,                  // 10^2
    1000.0,                 // 10^3
    10000.0,                // 10^4
    100000.0,               // 10^5
    1000000.0,              // 10^6
    10000000.0,             // 10^7
    100000000.0,            // 10^8
    1000000000.0,           // 10^9
    10000000000.0,          // 10^10
    100000000000.0,         // 10^11
    1000000000000.0,        // 10^12
    10000000000000.0,       // 10^13
    100000000000000.0,      // 10^14
    1000000000000000.0,     // 10^15
    10000000000000000.0     // 10^16
};

// ZigZag 编码函数
// __device__ static uint64_t zigzag_encode_cuda(int64_t value) {
//     return (value << 1) ^ (value >> 63);
// }
__device__ static unsigned long zigzag_encode_cuda(long value) {
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}

__device__ static int getDecimalPlaces(double value) {
    double trac = value + POW_NUM_G - POW_NUM_G;
    double temp = value;
    int digits = 0;
    int64_t int_temp, trac_temp;
    memcpy(&int_temp, &temp, sizeof(double));
    memcpy(&trac_temp, &trac, sizeof(double));
    while (llabs(trac_temp - int_temp) > 1.0 && digits < 16) { // 使用 llabs 替代 std::abs
        digits++;
        double td = pow10_table[digits]; // 使用查找表替代 pow
        temp = value * td;
        memcpy(&int_temp, &temp, sizeof(double));
        trac = temp + POW_NUM_G - POW_NUM_G;
        memcpy(&trac_temp, &trac, sizeof(double));
    }
    return digits;
}



// // 辅助函数：打印缓冲区的指定范围，以十六进制格式显示
// __device__ void print_bytes(const unsigned char* buffer, size_t start, size_t length, const char* label) {
//     printf("%s: ", label);
//     for (size_t i = start; i < start + length; ++i) {
//         // 打印每个字节的十六进制表示
//         printf("%02x ", buffer[i]);
//     }
//     printf("\n");
// }

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


__device__ inline int device_min(int a, int b) {
    return (a < b) ? a : b;
}

__device__ inline int device_max(int a, int b) {
    return (a > b) ? a : b;
}

__device__ inline uint64_t device_min_uint64(uint64_t a, uint64_t b) {
    return (a < b) ? a : b;
}

__device__ inline uint64_t device_max_uint64(uint64_t a, uint64_t b) {
    return (a > b) ? a : b;
}


__device__ inline long double2long(double data,int maxDecimalPlaces)
{
    if (maxDecimalPlaces > 15) {
        unsigned long long bits = __double_as_longlong(data);
        return static_cast<long>(bits);//当前线程在块中的数据位置
        } else {
        return static_cast<long>(round(data * pow10_table[maxDecimalPlaces]));
    }
}

/*
const double* input,
int nbEle,
unsigned char* output,
// uint64_t* bitSizes,
volatile unsigned int* const __restrict__ cmpOffset, // 压缩数据偏移量数组（输出）
volatile unsigned int* const __restrict__ locOffset, // 局部偏移量数组（输出）
volatile int* const __restrict__ flag             // 标志数组，用于同步不同warp的状态（输出
 */
__global__ void GDFC_compress_kernel_plain_f32(
    const float* const __restrict__ input, 
    unsigned char* const __restrict__ output, 
    volatile unsigned int* const __restrict__ cmpOffset, // 压缩数据偏移量数组（输出）
    volatile unsigned int* const __restrict__ locOffset, // 局部偏移量数组（输出）
    volatile int* const __restrict__ flag,             // 标志数组，用于同步不同warp的状态（输出
    const size_t nbEle
)
{
        // 共享内存，用于在线程块内共享数据
    __shared__ unsigned int excl_sum; // 排他性前缀和，用于偏移量计算
    //__shared__ unsigned int base_idx; // 当前warp的基地址索引

    // 获取线程和块信息
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;                   // 当前线程在warp中的位置（0-31）
    const int warp = idx >> 5;                     // 当前线程所属的warp编号
    
    // 每个线程处理1024个数据项
    int startIdx = idx * DATA_PER_THREAD;
    int endIdx = device_min((startIdx + DATA_PER_THREAD),static_cast<int>(nbEle));
    int numDatas = endIdx - startIdx;
    uint64_t deltas[DATA_PER_THREAD];
    //int numDeltas = numDatas - 1;
    if(numDatas<=0)
    {
        return;
    }
    // 局部变量
    int maxDecimalPlaces = 0;
    long firstValue = 0;
    int bitCount = 0;

    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    // int block_idx; // 如果不使用，可以移除

    long currQuant;
    long lorenQuant;
    long prevQuant;

    unsigned int thread_ofs = 0;
    double4 tmp_buffer;

    // 1. 采样 
    for (int i = 0; i < numDatas; i++) {
        maxDecimalPlaces = device_max(maxDecimalPlaces, getDecimalPlaces(input[startIdx + i]));
    }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    // 2. FOR + zigzag（用4向量化进行实现）
    uint64_t maxDelta = 0;
    for(int j = 0; j < (numDatas+31) / 32; j++) { // 每个线程的数量 / 一个数据批次（32）
        base_block_start_idx = startIdx + j * 32;           //每一组32个数据的起始位置
        base_block_end_idx = base_block_start_idx + 32;     //每一组32个数据的结束位置

        if(base_block_start_idx == startIdx) { // 针对第一个数据
            firstValue = double2long(input[base_block_start_idx], maxDecimalPlaces); // 量化当前数据点
            prevQuant = firstValue;// 初始化第一个量化值
            //printf("firstValue:%d\n",firstValue);
        }

        if(base_block_end_idx < nbEle) {
            #pragma unroll 8 //循环展开8次，就是4*8=32个数据
            for(int i = base_block_start_idx; i < base_block_end_idx; i += 4) {
                
                tmp_buffer = reinterpret_cast<const double4*>(input)[i / 4];
                quant_chunk_idx = j * 32 + (i % 32); //处理的每一组的第几个数据

                // 处理x分量
                if(i == startIdx) { // 针对第一个数据
                    // firstValue = double2long(tmp_buffer.x, maxDecimalPlaces); // 量化当前数据点
                    // prevQuant = firstValue;
                    // printf("firstValue:%d\n",firstValue);
                    deltas[quant_chunk_idx] = 0;//填充第一个数据为0保证1024个数据
                }
                else {
                    currQuant = double2long(tmp_buffer.x, maxDecimalPlaces); // 量化当前数据点
                    lorenQuant = currQuant - prevQuant; // 计算差分

                    deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
                    prevQuant = currQuant; // 更新前一个量化值
                    maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
                }

                // 处理y分量
                currQuant = double2long(tmp_buffer.y, maxDecimalPlaces);   
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

                // 处理z分量
                currQuant = double2long(tmp_buffer.z, maxDecimalPlaces);   
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);     

                // 处理w分量
                currQuant = double2long(tmp_buffer.w, maxDecimalPlaces);   
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+3],quant_chunk_idx+3,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);          
            }
        }
        else {
            // 处理当前数据块超出数据范围的情况
            if(base_block_start_idx >= endIdx) {
                // 如果整个数据块都超出范围，将absQuant设置为0
                quant_chunk_idx = j * 32 + (base_block_start_idx % 32);
                for(int i = quant_chunk_idx; i < quant_chunk_idx + 32; i++) 
                    deltas[i] = 0;
            }
            else {
                // 部分数据块在范围内，部分超出范围
                int remainbEle = nbEle - base_block_start_idx;  // 剩余有效数据元素数
                int zeronbEle = base_block_end_idx - nbEle;     // 超出范围的数据元素数

                // 处理剩余有效数据元素
                for(int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++) {
                    if(i==startIdx)
                    {
                        deltas[0]=0;
                        continue;
                    }
                    quant_chunk_idx = j * 32 + (i % 32);
                    currQuant = double2long(input[i], maxDecimalPlaces); 

                    lorenQuant = currQuant - prevQuant;

                    deltas[ quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    // printf("I:%d delta:%ld currQuant:%ld  prevQuant:%ld ",
                    //     quant_chunk_idx,  
                    //     lorenQuant,
                    //     currQuant,
                    //     prevQuant);
                    // printf("zigzag:%02x \n", deltas[quant_chunk_idx]);
                    prevQuant = currQuant;
                    maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]);  
                }

                quant_chunk_idx = j * 32 + (nbEle % 32);
                for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++) 
                    deltas[i] = 0;
            }  
        }
    }

    // 3. 得到编码过后的最大delta值，并且得到最大bit位
    bitCount = 0;

    while (maxDelta > 0) {
        maxDelta >>= 1;
        bitCount++;
    }

    // 防止 bitCount 为 0（所有 deltas 都为 0）
    if (bitCount == 0) {
        bitCount = 1;
    }

    // 限制 bitCount 不超过 MAX_BITCOUNT
    bitCount = device_min(bitCount, MAX_BITCOUNT);

    // 4. 稀疏列处理
    
        // 4.1 shuffle 
        // 原 numData个数据 每个数据占bitCount位
        // 变成 bitCount行 numData列 第i行是所有数据的第i位bit

        // 计算需要多少列，每列最多包含8个数据的bit位
        int numByte = (numDatas + 7) / 8;
        uint8_t result[64][128];
        // 初始化二维数组

        // 遍历每个uint64_t的数据
        for (int i = 0; i < bitCount; ++i) {//行
            
            for (int j = 0; j < numDatas; ++j) {//列numBytes
                //计算当前行（即bit位）
                // if(i==0)
                // {
                //     printf("0x %02x ", deltas[j]); // 打印为十六进制，且确保每个字节以2位输出
                // }
                int byteIndex = j / 8;  // 当前bit属于第几个字节
                int bitIndex = j % 8;   // 当前bit在字节中的位置

                // 提取当前bit位并存入结果数组
                result[i][byteIndex] |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7 - bitIndex));
            }
        }

        // 4.2 设置稀疏列，并且进行标记，同时计算bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;              // 用于记录每一列是否为稀疏列
        uint8_t flag2[64][16];          // 对于稀疏列统计稀疏位置,最多1024个数据，所以最多1024bit，即128byte,
        memset(flag2, 0, sizeof(flag2));
                                        //每一个byte用1bit标识，所以最多16byte
        //很重要的一点：flag2是byte单位
        for( int i=0;i<bitCount;++i)
        {
            int b0=0;
            int b1=0;
    //            int s=0;
            for(int j=0;j<numByte;++j)
            {
                int m_byte =j/8; //flag2的第几个byte位
                int m_bit =j%8;  //flag2的这个byte位的第几个bit
                if(result[i][j]==0)//如果是0可以稀疏化
                {
                    b0++;   //只用1个标识位，减少了7位
                    flag2[i][m_byte] &= ~(1ULL << m_bit);//设置flag2的第i行（和flag对应）的对应字节的bit位为0
                }
                else
                {
                    b1++;//多用一位进行标识
                    flag2[i][m_byte] |= (1ULL << m_bit); //设置flag2的第i行（和flag对应）的对应字节的bit位为1
                }
            }
            if (((numByte+7)/8+b1)>= numByte) {
                // 浪费了，设置 flag1 的第 i 位为 0
                flag1 &= ~(1ULL << i); // 使用按位与和按位取反清除第 i 位

                // 4.3 非稀疏列处理
                bitSize += 8 * numByte;

            } else {
                // 设置 flag1 的第 i 位为 1
                flag1 |= (1ULL << i); // 使用按位或设置第 i 位为 1

                // 4.4 稀疏列处理
                bitSize += ((numByte+7)/8+b1)*8;    //用byte为单位
            }
        }

        // for (int i = 0; i < bitCount; ++i) {
        //     for (int j = 0; j < numBytes; ++j) {
        //         printf("0x %02x ", result[i][j]); // 打印为十六进制，且确保每个字节以2位输出
        //     }
        //     printf("\n");
        // }

    // 5. 前缀和计算
        thread_ofs+=bitSize;//bitSize是每一个线程处理后需要写入的数据量所占的bit位

        // 5.1. Warp(块)内前缀和计算，确定每个线程的字节偏移量
        #pragma unroll 5
        for(int i = 1; i < 32; i <<= 1)
        {
            int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i); 
            if(lane >= i) thread_ofs += tmp;                      // 累加偏移量
        }
        __syncthreads(); // 同步线程，确保前缀和计算完成

        // 5.2 Warp(块)内最后一个线程更新locOffset和flag数组
        if(lane == 31) 
        {
            locOffset[warp + 1] = thread_ofs; // 更新下一warp的局部偏移量
            __threadfence();                  // 确保全局内存中的写操作完成
            if(warp == 0)
            {
                flag[0] = 2;                   // 标记第一个warp完成前缀和计算
                __threadfence();              
                flag[1] = 1;                   // 标记下一个warp可以开始
                __threadfence();
            }
            else
            {
                flag[warp + 1] = 1;            // 标记下一个warp可以开始
                __threadfence();    
            }
            //printf("flag[%d] ready\n",warp + 1);
        }
        __syncthreads(); // 同步线程，确保flag更新完成

        // 5.3 对于非第一个warp，计算排他性前缀和（有问题）
        if(warp > 0)
        {
            if(!lane) // 每个warp的第一个线程
            {
                int lookback = warp;          // 查找前一个warp(块)的状态
                int loc_excl_sum = 0;         // 本地排他性前缀和

                while(lookback > 0)//向前计算得到当前wrap（块）的起始位置
                {
                    int status;
                    do{
                        status = flag[lookback]; // 获取一个warp的状态
                    //    printf(" loop flag[%d]:%d\n",lookback,status);
                        __threadfence();         // 确保读取到最新的状态
                    } while(status == 0);

                    if(status == 2)
                    {
                        loc_excl_sum += cmpOffset[lookback]; // 累加前一个warp的cmpOffset
                        __threadfence();
                        break;
                    }
                    if(status == 1) 
                        loc_excl_sum += locOffset[lookback]; // 累加前一个warp的locOffset
                    lookback--;
                    __threadfence();
                   // printf(" turn flag[%d]:%d\n",lookback,status);
                }
                //printf(" loop out warp:%d\n",warp);
                excl_sum = loc_excl_sum; // 存储排他性前缀和
                //__syncthreads(); // 这个同步有毛病，用了就卡死，同步线程，确保排他性前缀和计算完成
                
                //printf("flag[%d] over0\n",warp);
                // 2.3 更新cmpOffset数组
                cmpOffset[warp] = excl_sum; // 更新当前warp的cmpOffset
                __threadfence();           // 确保写操作完成
                
                //printf("flag[%d] over1\n",warp);
                if(warp == gridDim.x - 1) 
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; // 更新最后一个warp的cmpOffset
                    __threadfence();
                }
                flag[warp] = 2;             // 标记当前warp完成
                //printf("flag[%d] over2\n",warp);
                __threadfence(); 
            } 
        }
        __syncthreads(); // 同步线程，确保cmpOffset更新完成
        
        // 5.4 得到写入位置
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap偏移+wrap内偏移 得到当前压缩后数据应该写入的起始位置
        int outputIdx = (outputIdxBit+7)/8;
        //printf("idx:%d outputIdx:%d bitSize :%d \n",idx,outputIdx,bitSize );
        // 6 开始写入
        // printf("excl_sum:%d\n",excl_sum);
        // printf("thread_ofs:%d\n",thread_ofs);
        // printf("bitSize:%d\n",bitSize);
        // printf("firstValue:%d\n",firstValue);
        // printf("maxDecimalPlaces:%d\n",maxDecimalPlaces);
        // printf("bitCount:%d\n",bitCount);
        // printf("flag1:%016llx\n",flag1);
        

        //bitSizes[idx] = bitSize;
        // 6.1 写入 bitSize (8 字节)
        for(int i = 0; i < 8; i++) {
            output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

        }


        // 6.2. 写入 firstValue (8 字节)
        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

        }

        // 6.3. 写入 maxDecimalPlaces 和 bitCount (各1字节)
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);

        output[outputIdx + 17] = static_cast<unsigned char>(bitCount);

        // 6.4 写入flag1(8字节 标识稀疏)
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 18 + i] = (flag1 >> (i * 8)) & 0xFF;
        }
        // 6.5 写入每一列 
        int flag2Byte=(numByte+7)/8;
        int ofs=outputIdx + 26;
        //int res=0;              //byte中剩余的bit位
        for(int i=0;i<bitCount;i++)
        {
            if((flag1 & (1ULL << i)) != 0)//flag第i个bit不为0:稀疏
            {
                // 6.5.1 稀疏列写入flag2+data
                for(int j=0;j<flag2Byte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(flag2[i][j]);
                    // printf("flag2[%d][%d]:0x%llx\n",i,j,flag2[i][j]);
                }
                for(int j=0;j<numByte;j++)
                {   
                    if(result[i][j])
                    {
                        output[ofs++] = static_cast<unsigned char>(result[i][j]);
                    }
                }
            }
            else{
                // 6.5.2 非稀疏列写入data
                for(int j=0;j<numByte;j++)
                {   
                    output[ofs++] = static_cast<unsigned char>(result[i][j]);
                }
            }

        }
}
__global__ void cuSZp_compress_kernel_outlier_f32(const float* const __restrict__ oriData, 
                                                  unsigned char* const __restrict__ cmpData, 
                                                  volatile unsigned int* const __restrict__ cmpOffset, 
                                                  volatile unsigned int* const __restrict__ locOffset, 
                                                  volatile int* const __restrict__ flag, 
                                                  const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk >> 5;
    const int rate_ofs = (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
    const float recipPrecision = 0.5f/eb;

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant;
    int absQuant[cmp_chunk];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0; // Thread-level prefix-sum, double check for overflow in large data (can be resolved by using size_t type).
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * cmp_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        fixed_rate[j] = 0;
        block_idx = base_block_start_idx/32;
        prevQuant = 0;
        int maxQuant = 0;
        int maxQuan2 = 0;
        int outlier = 0;

        if(base_block_end_idx < nbEle)
        {
            #pragma unroll 8
            for(int i=base_block_start_idx; i<base_block_end_idx; i+=4)
            {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i/4];
                quant_chunk_idx = j * 32 + i % 32;

                currQuant = quantization(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = i % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                if(sign_ofs) maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx] ? maxQuan2 : absQuant[quant_chunk_idx];
                else outlier = absQuant[quant_chunk_idx];

                currQuant = quantization(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+1) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+1] ? maxQuan2 : absQuant[quant_chunk_idx+1];

                currQuant = quantization(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+2) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+2] ? maxQuan2 : absQuant[quant_chunk_idx+2];

                currQuant = quantization(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+3) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
                maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx+3] ? maxQuan2 : absQuant[quant_chunk_idx+3];
            }
        }
        else
        {
            if(base_block_start_idx >= nbEle)
            {
                quant_chunk_idx = j * 32 + base_block_start_idx % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+32; i++) absQuant[i] = 0;
            }
            else
            {
                int remainbEle = nbEle - base_block_start_idx;
                int zeronbEle = base_block_end_idx - nbEle;

                for(int i=base_block_start_idx; i<base_block_start_idx+remainbEle; i++)
                {
                    quant_chunk_idx = j * 32 + i % 32;
                    currQuant = quantization(oriData[i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_ofs = i % 32;
                    sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                    if(sign_ofs) maxQuan2 = maxQuan2 > absQuant[quant_chunk_idx] ? maxQuan2 : absQuant[quant_chunk_idx];
                    else outlier = absQuant[quant_chunk_idx];
                }

                quant_chunk_idx = j * 32 + nbEle % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+zeronbEle; i++) absQuant[i] = 0;
            }  
        }

        int fr1 = get_bit_num(maxQuant);
        int fr2 = get_bit_num(maxQuan2);
        outlier = (get_bit_num(outlier)+7)/8;
        int temp_rate = 0;
        int temp_ofs1 = fr1 ? 4 + fr1 * 4 : 0;
        int temp_ofs2 = fr2 ? 4 + fr2 * 4 + outlier: 4 + outlier;
        if(temp_ofs1<=temp_ofs2) 
        {
            thread_ofs += temp_ofs1;
            temp_rate = fr1;
        }
        else 
        {
            thread_ofs += temp_ofs2;
            temp_rate = fr2 | 0x80 | ((outlier-1) << 5);
        }

        fixed_rate[j] = temp_rate;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        } 
    }
    __syncthreads();
    
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j=0; j<block_num; j++)
    {
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
        fixed_rate[j] &= 0x1f;
        int chunk_idx_start = j*32;

        if(!encoding_selection) tmp_byte_ofs = (fixed_rate[j]) ? (4 + fixed_rate[j] * 4) : 0;
        else tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(encoding_selection)
        {
            for(int i=0; i<outlier_byte_num; i++)
            {
                cmpData[cmp_byte_ofs++] = (unsigned char)(absQuant[chunk_idx_start] & 0xff);
                absQuant[chunk_idx_start] >>= 8;
            }

            if(!fixed_rate[j])
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);
                cmpData[cmp_byte_ofs++] = 0xff & sign_flag[j];
            }
        }

        if(fixed_rate[j])
        {
            int vec_ofs = cmp_byte_ofs % 4;
            if(vec_ofs==0)
            {
                tmp_char.x = 0xff & (sign_flag[j] >> 24);
                tmp_char.y = 0xff & (sign_flag[j] >> 16);
                tmp_char.z = 0xff & (sign_flag[j] >> 8);
                tmp_char.w = 0xff & sign_flag[j];
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;

                    if(!encoding_selection) tmp_char.x = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
                    tmp_char.x = tmp_char.x | 
                                (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                    tmp_char.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                    tmp_char.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                    
                    tmp_char.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                    mask <<= 1;
                }
            }
            else if(vec_ofs==1)
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 8);

                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;
                tmp_char.x = 0xff & sign_flag[j];
                if(!encoding_selection) tmp_char.y = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char.y = tmp_char.y | 
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);
                tmp_char.z = ((absQuant[chunk_idx_start+8] & 1) << 7) |
                            ((absQuant[chunk_idx_start+9] & 1) << 6) |
                            ((absQuant[chunk_idx_start+10] & 1) << 5) |
                            ((absQuant[chunk_idx_start+11] & 1) << 4) |
                            ((absQuant[chunk_idx_start+12] & 1) << 3) |
                            ((absQuant[chunk_idx_start+13] & 1) << 2) |
                            ((absQuant[chunk_idx_start+14] & 1) << 1) |
                            ((absQuant[chunk_idx_start+15] & 1) << 0);
                tmp_char.w = ((absQuant[chunk_idx_start+16] & 1) << 7) |
                            ((absQuant[chunk_idx_start+17] & 1) << 6) |
                            ((absQuant[chunk_idx_start+18] & 1) << 5) |
                            ((absQuant[chunk_idx_start+19] & 1) << 4) |
                            ((absQuant[chunk_idx_start+20] & 1) << 3) |
                            ((absQuant[chunk_idx_start+21] & 1) << 2) |
                            ((absQuant[chunk_idx_start+22] & 1) << 1) |
                            ((absQuant[chunk_idx_start+23] & 1) << 0);
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;
                    
                    tmp_char.x = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);
                    mask <<= 1;

                    if(!encoding_selection) tmp_char.y = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
                    tmp_char.y = tmp_char.y | 
                                (((absQuant[chunk_idx_start+0] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    tmp_char.z = (((absQuant[chunk_idx_start+8] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> (i+1)) << 0);

                    tmp_char.w = (((absQuant[chunk_idx_start+16] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> (i+1)) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                }

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+24] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+25] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+26] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+27] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+28] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+29] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+30] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+31] & mask) >> (fixed_rate[j]-1)) << 0);
            }
            else if(vec_ofs==2)
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 16);

                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;
                tmp_char.x = 0xff & (sign_flag[j] >> 8);
                tmp_char.y = 0xff & sign_flag[j];
                if(!encoding_selection) tmp_char.z = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char.z = tmp_char.z | 
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);
                tmp_char.w = ((absQuant[chunk_idx_start+8] & 1) << 7) |
                            ((absQuant[chunk_idx_start+9] & 1) << 6) |
                            ((absQuant[chunk_idx_start+10] & 1) << 5) |
                            ((absQuant[chunk_idx_start+11] & 1) << 4) |
                            ((absQuant[chunk_idx_start+12] & 1) << 3) |
                            ((absQuant[chunk_idx_start+13] & 1) << 2) |
                            ((absQuant[chunk_idx_start+14] & 1) << 1) |
                            ((absQuant[chunk_idx_start+15] & 1) << 0);
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;

                    tmp_char.x = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                    
                    tmp_char.y = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);
                    mask <<= 1;

                    if(!encoding_selection) tmp_char.z = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
                    tmp_char.z = tmp_char.z | 
                                (((absQuant[chunk_idx_start+0] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    tmp_char.w = (((absQuant[chunk_idx_start+8] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> (i+1)) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                }

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+16] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+17] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+18] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+19] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+20] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+21] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+22] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+23] & mask) >> (fixed_rate[j]-1)) << 0);
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+24] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+25] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+26] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+27] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+28] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+29] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+30] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+31] & mask) >> (fixed_rate[j]-1)) << 0);
            }
            else
            {
                cmpData[cmp_byte_ofs++] = 0xff & (sign_flag[j] >> 24);

                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;
                tmp_char.x = 0xff & (sign_flag[j] >> 16);
                tmp_char.y = 0xff & (sign_flag[j] >> 8);
                tmp_char.z = 0xff & sign_flag[j];
                if(!encoding_selection) tmp_char.w = ((absQuant[chunk_idx_start] & 1) << 7);
                tmp_char.w = tmp_char.w | 
                            ((absQuant[chunk_idx_start+1] & 1) << 6) |
                            ((absQuant[chunk_idx_start+2] & 1) << 5) |
                            ((absQuant[chunk_idx_start+3] & 1) << 4) |
                            ((absQuant[chunk_idx_start+4] & 1) << 3) |
                            ((absQuant[chunk_idx_start+5] & 1) << 2) |
                            ((absQuant[chunk_idx_start+6] & 1) << 1) |
                            ((absQuant[chunk_idx_start+7] & 1) << 0);
                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;

                int mask = 1;
                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char.x = 0;
                    tmp_char.y = 0;
                    tmp_char.z = 0;
                    tmp_char.w = 0;

                    tmp_char.x = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                    tmp_char.y = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                    
                    tmp_char.z = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                                (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                                (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                                (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                                (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                                (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                                (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                                (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);
                    mask <<= 1;

                    if(!encoding_selection) tmp_char.w = (((absQuant[chunk_idx_start] & mask) >> i) << 7);
                    tmp_char.w = (((absQuant[chunk_idx_start+0] & mask) >> (i+1)) << 7) |
                                (((absQuant[chunk_idx_start+1] & mask) >> (i+1)) << 6) |
                                (((absQuant[chunk_idx_start+2] & mask) >> (i+1)) << 5) |
                                (((absQuant[chunk_idx_start+3] & mask) >> (i+1)) << 4) |
                                (((absQuant[chunk_idx_start+4] & mask) >> (i+1)) << 3) |
                                (((absQuant[chunk_idx_start+5] & mask) >> (i+1)) << 2) |
                                (((absQuant[chunk_idx_start+6] & mask) >> (i+1)) << 1) |
                                (((absQuant[chunk_idx_start+7] & mask) >> (i+1)) << 0);

                    reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                    cmp_byte_ofs+=4;
                }

                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+8] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+9] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+10] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+11] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+12] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+13] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+14] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+15] & mask) >> (fixed_rate[j]-1)) << 0);
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+16] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+17] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+18] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+19] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+20] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+21] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+22] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+23] & mask) >> (fixed_rate[j]-1)) << 0);
                cmpData[cmp_byte_ofs++] = (((absQuant[chunk_idx_start+24] & mask) >> (fixed_rate[j]-1)) << 7) |
                                        (((absQuant[chunk_idx_start+25] & mask) >> (fixed_rate[j]-1)) << 6) |
                                        (((absQuant[chunk_idx_start+26] & mask) >> (fixed_rate[j]-1)) << 5) |
                                        (((absQuant[chunk_idx_start+27] & mask) >> (fixed_rate[j]-1)) << 4) |
                                        (((absQuant[chunk_idx_start+28] & mask) >> (fixed_rate[j]-1)) << 3) |
                                        (((absQuant[chunk_idx_start+29] & mask) >> (fixed_rate[j]-1)) << 2) |
                                        (((absQuant[chunk_idx_start+30] & mask) >> (fixed_rate[j]-1)) << 1) |
                                        (((absQuant[chunk_idx_start+31] & mask) >> (fixed_rate[j]-1)) << 0);
            }
        }
        
        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_decompress_kernel_outlier_f32(float* const __restrict__ decData, 
                                                    const unsigned char* const __restrict__ cmpData, 
                                                    volatile unsigned int* const __restrict__ cmpOffset, 
                                                    volatile unsigned int* const __restrict__ locOffset, 
                                                    volatile int* const __restrict__ flag, 
                                                    const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = dec_chunk >> 5;
    const int rate_ofs = (nbEle+dec_tblock_size*dec_chunk-1)/(dec_tblock_size*dec_chunk)*(dec_tblock_size*dec_chunk)/32;

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;    
    int absQuant[32];
    int currQuant, lorenQuant, prevQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char;
    float4 dec_buffer;

    for(int j=0; j<block_num; j++)
    {
        block_idx = warp * dec_chunk + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];

        int encoding_selection = fixed_rate[j] >> 7;
        int outlier = ((fixed_rate[j] & 0x60) >> 5) + 1;
        int temp_rate = fixed_rate[j] & 0x1f;
        if(!encoding_selection) thread_ofs += temp_rate ? (4 + temp_rate * 4) : 0;
        else thread_ofs += 4 + temp_rate * 4 + outlier;
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * dec_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        int encoding_selection = fixed_rate[j] >> 7;
        int outlier_byte_num = ((fixed_rate[j] & 0x60) >> 5) + 1;
        fixed_rate[j] &= 0x1f;
        int outlier_buffer = 0;
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        unsigned int sign_flag = 0;

        if(!encoding_selection) tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        else tmp_byte_ofs = 4 + outlier_byte_num + fixed_rate[j] * 4;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(encoding_selection)
        {
            for(int i=0; i<outlier_byte_num; i++)
            {
                int buffer = cmpData[cmp_byte_ofs++] << (8*i);
                outlier_buffer |= buffer;
            }

            if(!fixed_rate[j])
            {
                sign_flag = (0xff000000 & (cmpData[cmp_byte_ofs++] << 24)) |
                            (0x00ff0000 & (cmpData[cmp_byte_ofs++] << 16)) |
                            (0x0000ff00 & (cmpData[cmp_byte_ofs++] << 8))  |
                            (0x000000ff & cmpData[cmp_byte_ofs++]);
                absQuant[0] = outlier_buffer;
                for(int i=1; i<32; i++) absQuant[i] = 0;

                prevQuant = 0;
                if(base_block_end_idx < nbEle)
                {
                    #pragma unroll 8
                    for(int i=0; i<32; i+=4)
                    {
                        sign_ofs = i % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.x = currQuant * eb * 2;

                        sign_ofs = (i+1) % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.y = currQuant * eb * 2;

                        sign_ofs = (i+2) % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.z = currQuant * eb * 2;

                        sign_ofs = (i+3) % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        dec_buffer.w = currQuant * eb * 2;
                        
                        reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                    }
                }
                else
                {
                    for(int i=0; i<32; i++)
                    {
                        sign_ofs = i % 32;
                        lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                        currQuant = lorenQuant + prevQuant;
                        prevQuant = currQuant;
                        if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                    }
                }
            }
        }

        if(fixed_rate[j])
        {
            int vec_ofs = cmp_byte_ofs % 4;
            if(vec_ofs==0)
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                            (0x00ff0000 & (tmp_char.y << 16)) |
                            (0x0000ff00 & (tmp_char.z << 8))  |
                            (0x000000ff & tmp_char.w);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    absQuant[8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                    absQuant[9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                    absQuant[10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                    absQuant[11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                    absQuant[12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                    absQuant[13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                    absQuant[14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                    absQuant[15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                    absQuant[16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
                }
            }
            else if(vec_ofs==1)
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag = (0xff000000 & cmpData[cmp_byte_ofs++] << 24) |
                            (0x00ff0000 & cmpData[cmp_byte_ofs++] << 16) |
                            (0x0000ff00 & cmpData[cmp_byte_ofs++] << 8);

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag |= (0x000000ff & tmp_char.x);

                if(!encoding_selection) absQuant[0] |= ((tmp_char.y >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char.y >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char.y >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char.y >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char.y >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char.y >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char.y >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char.y >> 0) & 0x00000001);

                absQuant[8] |= ((tmp_char.z >> 7) & 0x00000001);
                absQuant[9] |= ((tmp_char.z >> 6) & 0x00000001);
                absQuant[10] |= ((tmp_char.z >> 5) & 0x00000001);
                absQuant[11] |= ((tmp_char.z >> 4) & 0x00000001);
                absQuant[12] |= ((tmp_char.z >> 3) & 0x00000001);
                absQuant[13] |= ((tmp_char.z >> 2) & 0x00000001);
                absQuant[14] |= ((tmp_char.z >> 1) & 0x00000001);
                absQuant[15] |= ((tmp_char.z >> 0) & 0x00000001);

                absQuant[16] |= ((tmp_char.w >> 7) & 0x00000001);
                absQuant[17] |= ((tmp_char.w >> 6) & 0x00000001);
                absQuant[18] |= ((tmp_char.w >> 5) & 0x00000001);
                absQuant[19] |= ((tmp_char.w >> 4) & 0x00000001);
                absQuant[20] |= ((tmp_char.w >> 3) & 0x00000001);
                absQuant[21] |= ((tmp_char.w >> 2) & 0x00000001);
                absQuant[22] |= ((tmp_char.w >> 1) & 0x00000001);
                absQuant[23] |= ((tmp_char.w >> 0) & 0x00000001);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    absQuant[24] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.y >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char.y >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char.y >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char.y >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char.y >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char.y >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char.y >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char.y >> 0) & 0x00000001) << (i+1);

                    absQuant[8] |= ((tmp_char.z >> 7) & 0x00000001) << (i+1);
                    absQuant[9] |= ((tmp_char.z >> 6) & 0x00000001) << (i+1);
                    absQuant[10] |= ((tmp_char.z >> 5) & 0x00000001) << (i+1);
                    absQuant[11] |= ((tmp_char.z >> 4) & 0x00000001) << (i+1);
                    absQuant[12] |= ((tmp_char.z >> 3) & 0x00000001) << (i+1);
                    absQuant[13] |= ((tmp_char.z >> 2) & 0x00000001) << (i+1);
                    absQuant[14] |= ((tmp_char.z >> 1) & 0x00000001) << (i+1);
                    absQuant[15] |= ((tmp_char.z >> 0) & 0x00000001) << (i+1);

                    absQuant[16] |= ((tmp_char.w >> 7) & 0x00000001) << (i+1);
                    absQuant[17] |= ((tmp_char.w >> 6) & 0x00000001) << (i+1);
                    absQuant[18] |= ((tmp_char.w >> 5) & 0x00000001) << (i+1);
                    absQuant[19] |= ((tmp_char.w >> 4) & 0x00000001) << (i+1);
                    absQuant[20] |= ((tmp_char.w >> 3) & 0x00000001) << (i+1);
                    absQuant[21] |= ((tmp_char.w >> 2) & 0x00000001) << (i+1);
                    absQuant[22] |= ((tmp_char.w >> 1) & 0x00000001) << (i+1);
                    absQuant[23] |= ((tmp_char.w >> 0) & 0x00000001) << (i+1);
                }

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);
            }
            else if(vec_ofs==2)
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag = (0xff000000 & cmpData[cmp_byte_ofs++] << 24) |
                            (0x00ff0000 & cmpData[cmp_byte_ofs++] << 16);

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag |= (0x0000ff00 & tmp_char.x << 8) |
                             (0x000000ff & tmp_char.y);

                if(!encoding_selection) absQuant[0] |= ((tmp_char.z >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char.z >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char.z >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char.z >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char.z >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char.z >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char.z >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char.z >> 0) & 0x00000001);

                absQuant[8] |= ((tmp_char.w >> 7) & 0x00000001);
                absQuant[9] |= ((tmp_char.w >> 6) & 0x00000001);
                absQuant[10] |= ((tmp_char.w >> 5) & 0x00000001);
                absQuant[11] |= ((tmp_char.w >> 4) & 0x00000001);
                absQuant[12] |= ((tmp_char.w >> 3) & 0x00000001);
                absQuant[13] |= ((tmp_char.w >> 2) & 0x00000001);
                absQuant[14] |= ((tmp_char.w >> 1) & 0x00000001);
                absQuant[15] |= ((tmp_char.w >> 0) & 0x00000001);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    absQuant[16] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.z >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char.z >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char.z >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char.z >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char.z >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char.z >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char.z >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char.z >> 0) & 0x00000001) << (i+1);

                    absQuant[8] |= ((tmp_char.w >> 7) & 0x00000001) << (i+1);
                    absQuant[9] |= ((tmp_char.w >> 6) & 0x00000001) << (i+1);
                    absQuant[10] |= ((tmp_char.w >> 5) & 0x00000001) << (i+1);
                    absQuant[11] |= ((tmp_char.w >> 4) & 0x00000001) << (i+1);
                    absQuant[12] |= ((tmp_char.w >> 3) & 0x00000001) << (i+1);
                    absQuant[13] |= ((tmp_char.w >> 2) & 0x00000001) << (i+1);
                    absQuant[14] |= ((tmp_char.w >> 1) & 0x00000001) << (i+1);
                    absQuant[15] |= ((tmp_char.w >> 0) & 0x00000001) << (i+1);                    
                }

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[16] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[17] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[18] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[19] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[20] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[21] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[22] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[23] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);
            }
            else
            {
                for(int i=0; i<32; i++) absQuant[i] = 0;

                if(encoding_selection) absQuant[0] = outlier_buffer;

                sign_flag = (0xff000000 & cmpData[cmp_byte_ofs++] << 24);                            

                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                sign_flag |= (0x00ff0000 & tmp_char.x << 16) |
                             (0x0000ff00 & tmp_char.y << 8)  |
                             (0x000000ff & tmp_char.z);

                if(!encoding_selection) absQuant[0] |= ((tmp_char.w >> 7) & 0x00000001);
                absQuant[1] |= ((tmp_char.w >> 6) & 0x00000001);
                absQuant[2] |= ((tmp_char.w >> 5) & 0x00000001);
                absQuant[3] |= ((tmp_char.w >> 4) & 0x00000001);
                absQuant[4] |= ((tmp_char.w >> 3) & 0x00000001);
                absQuant[5] |= ((tmp_char.w >> 2) & 0x00000001);
                absQuant[6] |= ((tmp_char.w >> 1) & 0x00000001);
                absQuant[7] |= ((tmp_char.w >> 0) & 0x00000001);
                cmp_byte_ofs+=4;

                for(int i=0; i<fixed_rate[j]-1; i++)
                {
                    tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                    cmp_byte_ofs+=4;

                    absQuant[8] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                    absQuant[9] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                    absQuant[10] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                    absQuant[11] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                    absQuant[12] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                    absQuant[13] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                    absQuant[14] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                    absQuant[15] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                    absQuant[16] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                    absQuant[17] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                    absQuant[18] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                    absQuant[19] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                    absQuant[20] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                    absQuant[21] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                    absQuant[22] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                    absQuant[23] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                    absQuant[24] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                    absQuant[25] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                    absQuant[26] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                    absQuant[27] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                    absQuant[28] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                    absQuant[29] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                    absQuant[30] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                    absQuant[31] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                    if(!encoding_selection) absQuant[0] |= ((tmp_char.w >> 7) & 0x00000001) << (i+1);
                    absQuant[1] |= ((tmp_char.w >> 6) & 0x00000001) << (i+1);
                    absQuant[2] |= ((tmp_char.w >> 5) & 0x00000001) << (i+1);
                    absQuant[3] |= ((tmp_char.w >> 4) & 0x00000001) << (i+1);
                    absQuant[4] |= ((tmp_char.w >> 3) & 0x00000001) << (i+1);
                    absQuant[5] |= ((tmp_char.w >> 2) & 0x00000001) << (i+1);
                    absQuant[6] |= ((tmp_char.w >> 1) & 0x00000001) << (i+1);
                    absQuant[7] |= ((tmp_char.w >> 0) & 0x00000001) << (i+1);                    
                }

                int uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[8] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[9] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[10] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[11] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[12] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[13] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[14] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[15] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[16] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[17] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[18] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[19] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[20] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[21] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[22] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[23] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);

                uchar_buffer = cmpData[cmp_byte_ofs++];
                absQuant[24] |= ((uchar_buffer >> 7) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[25] |= ((uchar_buffer >> 6) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[26] |= ((uchar_buffer >> 5) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[27] |= ((uchar_buffer >> 4) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[28] |= ((uchar_buffer >> 3) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[29] |= ((uchar_buffer >> 2) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[30] |= ((uchar_buffer >> 1) & 0x00000001) << (fixed_rate[j]-1);
                absQuant[31] |= ((uchar_buffer >> 0) & 0x00000001) << (fixed_rate[j]-1);
            }
            
            prevQuant = 0;
            if(base_block_end_idx < nbEle)
            {
                #pragma unroll 8
                for(int i=0; i<32; i+=4)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.x = currQuant * eb * 2;

                    sign_ofs = (i+1) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.y = currQuant * eb * 2;

                    sign_ofs = (i+2) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.z = currQuant * eb * 2;

                    sign_ofs = (i+3) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.w = currQuant * eb * 2;
                    
                    reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                }
            }
            else
            {
                for(int i=0; i<32; i++)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                }
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

__global__ void cuSZp_compress_kernel_plain_f32(const float* const __restrict__ oriData, 
                                                unsigned char* const __restrict__ cmpData, 
                                                volatile unsigned int* const __restrict__ cmpOffset, 
                                                volatile unsigned int* const __restrict__ locOffset, 
                                                volatile int* const __restrict__ flag, 
                                                const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = cmp_chunk >> 5;
    const int rate_ofs = (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
    const float recipPrecision = 0.5f/eb;

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    int block_idx;
    int currQuant, lorenQuant, prevQuant, maxQuant;
    int absQuant[cmp_chunk];
    unsigned int sign_flag[block_num];
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0; // Thread-level prefix-sum, double check for overflow in large data (can be resolved by using size_t type).
    float4 tmp_buffer;
    uchar4 tmp_char;

    base_start_idx = warp * cmp_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        sign_flag[j] = 0;
        block_idx = base_block_start_idx/32;
        prevQuant = 0;
        maxQuant = 0;

        if(base_block_end_idx < nbEle)
        {
            #pragma unroll 8
            for(int i=base_block_start_idx; i<base_block_end_idx; i+=4)
            {
                tmp_buffer = reinterpret_cast<const float4*>(oriData)[i/4];
                quant_chunk_idx = j * 32 + i % 32;

                currQuant = quantization(tmp_buffer.x, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = i % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];

                currQuant = quantization(tmp_buffer.y, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+1) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+1] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+1] ? maxQuant : absQuant[quant_chunk_idx+1];

                currQuant = quantization(tmp_buffer.z, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+2) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+2] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+2] ? maxQuant : absQuant[quant_chunk_idx+2];

                currQuant = quantization(tmp_buffer.w, recipPrecision);
                lorenQuant = currQuant - prevQuant;
                prevQuant = currQuant;
                sign_ofs = (i+3) % 32;
                sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                absQuant[quant_chunk_idx+3] = abs(lorenQuant);
                maxQuant = maxQuant > absQuant[quant_chunk_idx+3] ? maxQuant : absQuant[quant_chunk_idx+3];
            }
        }
        else
        {
            if(base_block_start_idx >= nbEle)
            {
                quant_chunk_idx = j * 32 + base_block_start_idx % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+32; i++) absQuant[i] = 0;
            }
            else
            {
                int remainbEle = nbEle - base_block_start_idx;
                int zeronbEle = base_block_end_idx - nbEle;

                for(int i=base_block_start_idx; i<base_block_start_idx+remainbEle; i++)
                {
                    quant_chunk_idx = j * 32 + i % 32;
                    currQuant = quantization(oriData[i], recipPrecision);
                    lorenQuant = currQuant - prevQuant;
                    prevQuant = currQuant;
                    sign_ofs = i % 32;
                    sign_flag[j] |= (lorenQuant < 0) << (31 - sign_ofs);
                    absQuant[quant_chunk_idx] = abs(lorenQuant);
                    maxQuant = maxQuant > absQuant[quant_chunk_idx] ? maxQuant : absQuant[quant_chunk_idx];
                }

                quant_chunk_idx = j * 32 + nbEle % 32;
                for(int i=quant_chunk_idx; i<quant_chunk_idx+zeronbEle; i++) absQuant[i] = 0;
            }  
        }

        fixed_rate[j] = get_bit_num(maxQuant);
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        cmpData[block_idx] = (unsigned char)fixed_rate[j];
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane)
        {
            cmpOffset[warp] = excl_sum;
            __threadfence();
            if(warp==gridDim.x-1) cmpOffset[warp+1] = cmpOffset[warp] + locOffset[warp+1];
            __threadfence();
            flag[warp] = 2;
            __threadfence(); 
        }
    }
    __syncthreads();
    
    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    for(int j=0; j<block_num; j++)
    {
        int chunk_idx_start = j*32;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char.x = 0xff & (sign_flag[j] >> 24);
            tmp_char.y = 0xff & (sign_flag[j] >> 16);
            tmp_char.z = 0xff & (sign_flag[j] >> 8);
            tmp_char.w = 0xff & sign_flag[j];
            reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
            cmp_byte_ofs+=4;

            int mask = 1;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char.x = 0;
                tmp_char.y = 0;
                tmp_char.z = 0;
                tmp_char.w = 0;

                tmp_char.x = (((absQuant[chunk_idx_start+0] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+1] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+2] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+3] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+4] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+5] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+6] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+7] & mask) >> i) << 0);

                tmp_char.y = (((absQuant[chunk_idx_start+8] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+9] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+10] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+11] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+12] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+13] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+14] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+15] & mask) >> i) << 0);

                tmp_char.z = (((absQuant[chunk_idx_start+16] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+17] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+18] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+19] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+20] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+21] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+22] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+23] & mask) >> i) << 0);
                
                tmp_char.w = (((absQuant[chunk_idx_start+24] & mask) >> i) << 7) |
                             (((absQuant[chunk_idx_start+25] & mask) >> i) << 6) |
                             (((absQuant[chunk_idx_start+26] & mask) >> i) << 5) |
                             (((absQuant[chunk_idx_start+27] & mask) >> i) << 4) |
                             (((absQuant[chunk_idx_start+28] & mask) >> i) << 3) |
                             (((absQuant[chunk_idx_start+29] & mask) >> i) << 2) |
                             (((absQuant[chunk_idx_start+30] & mask) >> i) << 1) |
                             (((absQuant[chunk_idx_start+31] & mask) >> i) << 0);

                reinterpret_cast<uchar4*>(cmpData)[cmp_byte_ofs/4] = tmp_char;
                cmp_byte_ofs+=4;
                mask <<= 1;
            }
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}


__global__ void cuSZp_decompress_kernel_plain_f32(float* const __restrict__ decData, 
                                                  const unsigned char* const __restrict__ cmpData, 
                                                  volatile unsigned int* const __restrict__ cmpOffset, 
                                                  volatile unsigned int* const __restrict__ locOffset, 
                                                  volatile int* const __restrict__ flag, 
                                                  const float eb, const size_t nbEle)
{
    __shared__ unsigned int excl_sum;
    __shared__ unsigned int base_idx;

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    const int lane = idx & 0x1f;
    const int warp = idx >> 5;
    const int block_num = dec_chunk >> 5;
    const int rate_ofs = (nbEle+dec_tblock_size*dec_chunk-1)/(dec_tblock_size*dec_chunk)*(dec_tblock_size*dec_chunk)/32;

    int base_start_idx;
    int base_block_start_idx, base_block_end_idx;
    int block_idx;    
    int absQuant[32];
    int currQuant, lorenQuant, prevQuant;
    int sign_ofs;
    int fixed_rate[block_num];
    unsigned int thread_ofs = 0;
    uchar4 tmp_char;
    float4 dec_buffer;

    for(int j=0; j<block_num; j++)
    {
        block_idx = warp * dec_chunk + j * 32 + lane;
        fixed_rate[j] = (int)cmpData[block_idx];
        thread_ofs += (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        __syncthreads();
    }

    #pragma unroll 5
    for(int i=1; i<32; i<<=1)
    {
        int tmp = __shfl_up_sync(0xffffffff, thread_ofs, i);
        if(lane >= i) thread_ofs += tmp;
    }
    __syncthreads();

    if(lane==31) 
    {
        locOffset[warp+1] = thread_ofs;
        __threadfence();
        if(warp==0)
        {
            flag[0] = 2;
            __threadfence();
            flag[1] = 1;
            __threadfence();
        }
        else
        {
            flag[warp+1] = 1;
            __threadfence();    
        }
    }
    __syncthreads();

    if(warp>0)
    {
        if(!lane)
        {
            int lookback = warp;
            int loc_excl_sum = 0;
            while(lookback>0)
            {
                int status;
                do{
                    status = flag[lookback];
                    __threadfence();
                } while (status==0);
                if(status==2)
                {
                    loc_excl_sum += cmpOffset[lookback];
                    __threadfence();
                    break;
                }
                if(status==1) loc_excl_sum += locOffset[lookback];
                lookback--;
                __threadfence();
            }
            excl_sum = loc_excl_sum;
        }
        __syncthreads();
    }
    
    if(warp>0)
    {
        if(!lane) cmpOffset[warp] = excl_sum;
        __threadfence();
        if(!lane) flag[warp] = 2;
        __threadfence();  
    }
    __syncthreads();

    if(!lane) base_idx = excl_sum + rate_ofs;
    __syncthreads();

    unsigned int base_cmp_byte_ofs = base_idx;
    unsigned int cmp_byte_ofs;
    unsigned int tmp_byte_ofs = 0;
    unsigned int cur_byte_ofs = 0;
    base_start_idx = warp * dec_chunk * 32;
    for(int j=0; j<block_num; j++)
    {
        base_block_start_idx = base_start_idx + j * 1024 + lane * 32;
        base_block_end_idx = base_block_start_idx + 32;
        unsigned int sign_flag = 0;

        tmp_byte_ofs = (fixed_rate[j]) ? (4+fixed_rate[j]*4) : 0;
        #pragma unroll 5
        for(int i=1; i<32; i<<=1)
        {
            int tmp = __shfl_up_sync(0xffffffff, tmp_byte_ofs, i);
            if(lane >= i) tmp_byte_ofs += tmp;
        }
        unsigned int prev_thread = __shfl_up_sync(0xffffffff, tmp_byte_ofs, 1);
        if(!lane) cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs;
        else cmp_byte_ofs = base_cmp_byte_ofs + cur_byte_ofs + prev_thread;

        if(fixed_rate[j])
        {
            tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
            sign_flag = (0xff000000 & (tmp_char.x << 24)) |
                        (0x00ff0000 & (tmp_char.y << 16)) |
                        (0x0000ff00 & (tmp_char.z << 8))  |
                        (0x000000ff & tmp_char.w);
            cmp_byte_ofs+=4;
            
            for(int i=0; i<32; i++) absQuant[i] = 0;
            for(int i=0; i<fixed_rate[j]; i++)
            {
                tmp_char = reinterpret_cast<const uchar4*>(cmpData)[cmp_byte_ofs/4];
                cmp_byte_ofs+=4;

                absQuant[0] |= ((tmp_char.x >> 7) & 0x00000001) << i;
                absQuant[1] |= ((tmp_char.x >> 6) & 0x00000001) << i;
                absQuant[2] |= ((tmp_char.x >> 5) & 0x00000001) << i;
                absQuant[3] |= ((tmp_char.x >> 4) & 0x00000001) << i;
                absQuant[4] |= ((tmp_char.x >> 3) & 0x00000001) << i;
                absQuant[5] |= ((tmp_char.x >> 2) & 0x00000001) << i;
                absQuant[6] |= ((tmp_char.x >> 1) & 0x00000001) << i;
                absQuant[7] |= ((tmp_char.x >> 0) & 0x00000001) << i;

                absQuant[8] |= ((tmp_char.y >> 7) & 0x00000001) << i;
                absQuant[9] |= ((tmp_char.y >> 6) & 0x00000001) << i;
                absQuant[10] |= ((tmp_char.y >> 5) & 0x00000001) << i;
                absQuant[11] |= ((tmp_char.y >> 4) & 0x00000001) << i;
                absQuant[12] |= ((tmp_char.y >> 3) & 0x00000001) << i;
                absQuant[13] |= ((tmp_char.y >> 2) & 0x00000001) << i;
                absQuant[14] |= ((tmp_char.y >> 1) & 0x00000001) << i;
                absQuant[15] |= ((tmp_char.y >> 0) & 0x00000001) << i;

                absQuant[16] |= ((tmp_char.z >> 7) & 0x00000001) << i;
                absQuant[17] |= ((tmp_char.z >> 6) & 0x00000001) << i;
                absQuant[18] |= ((tmp_char.z >> 5) & 0x00000001) << i;
                absQuant[19] |= ((tmp_char.z >> 4) & 0x00000001) << i;
                absQuant[20] |= ((tmp_char.z >> 3) & 0x00000001) << i;
                absQuant[21] |= ((tmp_char.z >> 2) & 0x00000001) << i;
                absQuant[22] |= ((tmp_char.z >> 1) & 0x00000001) << i;
                absQuant[23] |= ((tmp_char.z >> 0) & 0x00000001) << i;

                absQuant[24] |= ((tmp_char.w >> 7) & 0x00000001) << i;
                absQuant[25] |= ((tmp_char.w >> 6) & 0x00000001) << i;
                absQuant[26] |= ((tmp_char.w >> 5) & 0x00000001) << i;
                absQuant[27] |= ((tmp_char.w >> 4) & 0x00000001) << i;
                absQuant[28] |= ((tmp_char.w >> 3) & 0x00000001) << i;
                absQuant[29] |= ((tmp_char.w >> 2) & 0x00000001) << i;
                absQuant[30] |= ((tmp_char.w >> 1) & 0x00000001) << i;
                absQuant[31] |= ((tmp_char.w >> 0) & 0x00000001) << i;
            }
            
            prevQuant = 0;
            if(base_block_end_idx < nbEle)
            {
                #pragma unroll 8
                for(int i=0; i<32; i+=4)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.x = currQuant * eb * 2;

                    sign_ofs = (i+1) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+1] * -1 : absQuant[i+1];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.y = currQuant * eb * 2;

                    sign_ofs = (i+2) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+2] * -1 : absQuant[i+2];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.z = currQuant * eb * 2;

                    sign_ofs = (i+3) % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i+3] * -1 : absQuant[i+3];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    dec_buffer.w = currQuant * eb * 2;
                    
                    reinterpret_cast<float4*>(decData)[(base_block_start_idx+i)/4] = dec_buffer;
                }
            }
            else
            {
                for(int i=0; i<32; i++)
                {
                    sign_ofs = i % 32;
                    lorenQuant = sign_flag & (1 << (31 - sign_ofs)) ? absQuant[i] * -1 : absQuant[i];
                    currQuant = lorenQuant + prevQuant;
                    prevQuant = currQuant;
                    if(base_block_start_idx+i < nbEle) decData[base_block_start_idx+i] = currQuant * eb * 2;
                }
            }      
        }

        cur_byte_ofs += __shfl_sync(0xffffffff, tmp_byte_ofs, 31);
    }
}

// __device__ inline int quantization(float data, float recipPrecision)
// {
//     float dataRecip = data*recipPrecision;
//     int s = dataRecip>=-0.5f?0:1;
//     return (int)(dataRecip+0.5f) - s;
// }


// __device__ inline int get_bit_num(unsigned int x)
// {
//     return (sizeof(unsigned int)*8) - __clz(x);
// }