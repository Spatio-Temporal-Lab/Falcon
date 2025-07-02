//
// Created by lz on 24-9-26.
//
#include "GDFCompressor.cuh"
#include <iomanip> // 用于设置输出格式
// 定义常量

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
__device__ __forceinline__ static unsigned long zigzag_encode_cuda(long value) {
    return (value << 1) ^ (value >> (sizeof(long) * 8 - 1));
}

__device__ static int getDecimalPlaces(double value,int sp) {
    double trac = value + POW_NUM_G - POW_NUM_G;
    double temp = value;

    int digits = 0;
    double td = 1;
    double deltaBound = abs(value) * pow(2, -52);
    while (abs(temp - trac) >= deltaBound * td && digits < 16 - sp - 1)
    {
        digits++;
        td = pow10_table[digits];
        temp = value * td;
        trac = temp + POW_NUM_G - POW_NUM_G;
    }

    return digits;
}



// 辅助函数：打印缓冲区的指定范围，以十六进制格式显示
__device__ void print_bytes(const unsigned char* buffer, size_t start, size_t length, const char* label) {
    printf("%s: ", label);
    for (size_t i = start; i < start + length; ++i) {
        // 打印每个字节的十六进制表示
        printf("%02x ", buffer[i]);
    }
    printf("\n");
}

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

__device__ long encodeDoubleWithSignLast(double x) {
    union {
        double d;
        long u;
    } val;

    val.d = x;

    return (val.u << 1) ^ (val.u >> (sizeof(long) * 8 - 1));
}

__device__ inline long double2long(double data,int maxDecimalPlaces, int maxBeta)
{
    // if (maxBeta > 15) {
    //     unsigned long long bits = __double_as_longlong(data);
    //     return static_cast<long>(bits);//当前线程在块中的数据位置
    //     } else {
    //     return static_cast<long>(round(data * pow10_table[maxDecimalPlaces]));
    // }
    return (maxBeta > 15)
    ? (encodeDoubleWithSignLast(data))
    : static_cast<long>(round(data * pow10_table[maxDecimalPlaces]));
}
// 核函数实现
/*
实现逻辑：
1.采样
    1.1 获取所有小数的最大小数位数
    1.2 所有数字转化为整数
    1.3 得到第一个值
2.（原差分序列）FOR + zigzag
    2.1 用FOR的话就可以进一步提高并行度，可以用··向量化··（8或者4）来处理一个线程的数据
        会比delta更快
    2.2 zigzag处理负数
3. 得到编码过后的最大delta值，从而得到最大bit位数
4. 稀疏列和非稀疏列（需要问一下）
    4.1 先shuffle
    4.2 根据每一个数（shuffle过后的每一个8bit就是8个数的对应bit）中0的位数进行添加标识位
5. 前缀和计算，模仿cuszp（学一下）
    找到数据存放位置
6. bit位打包（这个目前的新方法还不是很清楚）
    对于找到首地址的每个块：
    6.1 写入：总位宽（64 bit），第一个数（64 bit），最大小数位数（8 bit），最大位宽（8 bit）
    6.2 原本的方法：
        写入最佳切割点（每个数的前几位是稀疏的）
        写入flag
        写入转置稀疏列
        写入转置非稀疏列
    6.3 新方法：（问一下）
        主要是还需不需要最佳切入点了
        先转置，变成二维数组

要点：
1.内存处理：一个块中BLOCK_SIZE_G（32）个线程，一个线程处理DATA_PER_THREAD（64）个数据

2.向量化加速
3.
*/
/*
input：输入的完整数据
totalSize:输入数据的完整大小
output:输出数据
bitSizes:每个块的bitSize
*/
__global__ void compressBlockKernel(
    const double* input,
    int totalSize,
    unsigned char* output,
    uint64_t* bitSizes,
    volatile unsigned int* const __restrict__ cmpOffset, // 压缩数据偏移量数组（输出）
    volatile unsigned int* const __restrict__ locOffset, // 局部偏移量数组（输出）
    volatile int* const __restrict__ flag             // 标志数组，用于同步不同warp的状态（输出
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
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = endIdx - startIdx;
    uint64_t deltas[DATA_PER_THREAD];
    //int numDeltas = numDatas - 1;
    if(numDatas<=0)
    {
        return;
    }
    // 局部变量
    int maxDecimalPlaces = 0;
    int maxBeta =0;
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
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);

        double alpha = getDecimalPlaces(value, sp);// 得到小数位数
        double beta =  alpha + sp + 1;
        maxBeta = device_max(maxBeta,beta);
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    // 2. FOR + zigzag（用4向量化进行实现）
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); // 量化第一个
    prevQuant = firstValue;// 初始化第一个量化值
    for(int j = 0; j < (numDatas+31) / 32; j++) { // 每个线程的数量 / 一个数据批次（32）
        base_block_start_idx = startIdx + j * 32;           //每一组32个数据的起始位置
        base_block_end_idx = base_block_start_idx + 32;     //每一组32个数据的结束位置

        if(base_block_end_idx < totalSize) {
                int i = base_block_start_idx;
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
                    currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); // 量化当前数据点
                    lorenQuant = currQuant - prevQuant; // 计算差分

                    deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
                    prevQuant = currQuant; // 更新前一个量化值
                    maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
                }

                // 处理y分量
                currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

                // 处理z分量
                currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

                // 处理w分量
                currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+3],quant_chunk_idx+3,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);
                i+=4;
            #pragma unroll 7 //循环展开8次，就是4*8=32个数据,修改为7次，把第一次提取出来
            for(; i < base_block_end_idx; i += 4) {

                tmp_buffer = reinterpret_cast<const double4*>(input)[i / 4];
                quant_chunk_idx = j * 32 + (i % 32); //处理的每一组的第几个数据

                // 处理x分量
                // if(i == startIdx) { // 针对第一个数据
                //     // firstValue = double2long(tmp_buffer.x, maxDecimalPlaces); // 量化当前数据点
                //     // prevQuant = firstValue;
                //     // printf("firstValue:%d\n",firstValue);
                //     deltas[quant_chunk_idx] = 0;//填充第一个数据为0保证1024个数据
                // }
                // else {
                currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); // 量化当前数据点
                lorenQuant = currQuant - prevQuant; // 计算差分

                deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant; // 更新前一个量化值
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
                // }

                // 处理y分量
                currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

                // 处理z分量
                currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

                // 处理w分量
                currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
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
                int remainbEle = totalSize - base_block_start_idx;  // 剩余有效数据元素数
                int zeronbEle = base_block_end_idx - totalSize;     // 超出范围的数据元素数

                // 处理剩余有效数据元素
                for(int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++) {
                    if(i==startIdx)
                    {
                        deltas[0]=0;
                        continue;
                    }
                    quant_chunk_idx = j * 32 + (i % 32);
                    currQuant = double2long(input[i], maxDecimalPlaces,maxBeta);

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

                quant_chunk_idx = j * 32 + (totalSize % 32);
                for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
                    deltas[i] = 0;
            }
        }
    }

    // 3. 得到编码过后的最大delta值，并且得到最大bit位
    // bitCount = 0;

    // while (maxDelta > 0) {
    //     maxDelta >>= 1;
    //     bitCount++;
    // }

    // // 防止 bitCount 为 0（所有 deltas 都为 0）
    // if (bitCount == 0) {
    //     bitCount = 1;
    // }
    // // 限制 bitCount 不超过 MAX_BITCOUNT
    // bitCount = device_min(bitCount, MAX_BITCOUNT);
    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//用内置函数 替代处理循环
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

    // // 使用寄存器存储局部变量
    // register int local_bitCount = bitCount;

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
            int j=0;
                // while(j+32<numDatas)
                // {
                //     int byteIndex = j / 8;  // 当前bit属于第几个字节
                //     result[i][byteIndex] = result[i][byteIndex] |
                //                             (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << (0));

                //     result[i][byteIndex+1] = result[i][byteIndex+1] |
                //                             (((deltas[j+8] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+9] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+10] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+11] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+12] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+13] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+14] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+15] >> (bitCount - 1 - i)) & 1) << (0));

                //     result[i][byteIndex+2] = result[i][byteIndex+2] |
                //                             (((deltas[j+16] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+17] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+18] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+19] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+20] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+21] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+22] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+23] >> (bitCount - 1 - i)) & 1) << (0));

                //     result[i][byteIndex+3] = result[i][byteIndex+3] |
                //                             (((deltas[j+24] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+25] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+26] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+27] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+28] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+29] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+30] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+31] >> (bitCount - 1 - i)) & 1) << (0));
                //     j+=32;
                // }
            while(j+8<numDatas)//有效 0.0027->0.0023
            {
                int byteIndex = j / 8;  // 当前bit属于第几个字节
                result[i][byteIndex] = result[i][byteIndex] |
                                        (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7))|
                                        (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << (6))|
                                        (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << (5))|
                                        (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << (4))|
                                        (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << (3))|
                                        (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << (2))|
                                        (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << (1))|
                                        (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << (0));
                j+=8;
            }
            for (; j <numDatas ; ++j) {//列numBytes
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
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;              // 用于记录每一列是否为稀疏列
        uint8_t flag2[64][16];          // 对于稀疏列统计稀疏位置,最多1024个数据，所以最多1024bit，即128byte,
        memset(flag2, 0, sizeof(flag2));
                                        //每一个byte用1bit标识，所以最多16byte
        //很重要的一点：flag2是byte单位
        // for( int i=0;i<bitCount;++i)
        // {
        //     int b0=0;
        //     int b1=0;
        //     // int s=0;
        //     for(int j=0;j<numByte;++j)
        //     {
        //         int m_byte =j/8; //flag2的第几个byte位
        //         int m_bit =j%8;  //flag2的这个byte位的第几个bit
        //         if(result[i][j]==0)//如果是0可以稀疏化
        //         {
        //             b0++;   //全0byte 只用1个标识位
        //             flag2[i][m_byte] &= ~(1ULL << m_bit);//设置flag2的第i行（和flag对应）的对应字节的bit位为0
        //         }
        //         else
        //         {
        //             b1++;//有效的byte 多用一位进行标识
        //             flag2[i][m_byte] |= (1ULL << m_bit); //设置flag2的第i行（和flag对应）的对应字节的bit位为1
        //         }
        //     }
        //     if (((numByte+7)/8+b1)>= numByte) {//存储flag2的单位也是BYTE,有numByte位
        //         // 浪费了，设置 flag1 的第 i 位为 0
        //         flag1 &= ~(1ULL << i); // 使用按位与和按位取反清除第 i 位

        //         // 4.3 非稀疏列处理
        //         bitSize += 8 * numByte;

        //     } else {
        //         // 设置 flag1 的第 i 位为 1
        //         flag1 |= (1ULL << i); // 使用按位或设置第 i 位为 1

        //         // 4.4 稀疏列处理
        //         bitSize += ((numByte+7)/8+b1)*8;    //用byte为单位*8得到bit
        //     }
        // }
        int BITS_PER_THREAD=4;
        for(int i = 0; i < bitCount; i += BITS_PER_THREAD) { // 每次处理4个比特位
            for(int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b) {
                int bit = i + b;
                int b0 = 0;
                int b1 = 0;
                for(int j = 0; j < numByte; j++) {
                    int m_byte = j / 8;
                    int m_bit = j % 8;
                    uint8_t current_result = result[bit][j];
                    b0 += (current_result == 0);
                    b1 += (current_result != 0);
                    flag2[bit][m_byte] |= (current_result != 0) << m_bit;//设置1
                    flag2[bit][m_byte] &= ~((current_result == 0) << m_bit);//清零
                }
                // 使用掩码和算术操作代替分支(有效0.0023->0.0021)
                uint64_t is_sparse = ((numByte + 7) / 8 + b1) < numByte;
                flag1 |= (is_sparse << bit);
                flag1 &= ~((!is_sparse) << bit);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
                //flag2的长度+b1或者numByte*8
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


        bitSizes[idx] = bitSize;

        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        // if (outputIdx % 8 != 0) {
        // 6.1 写入 bitSize (8 字节)
            for(int i = 0; i < 8; i++) {
                output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            }


            // 6.2. 写入 firstValue (8 字节)
            for(int i = 0; i < 8; i++) {
                output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            }
        // }
        // else
        // {
        //     // 6.1+6.2优化 示例：一次性写入 bitSize 和 firstValueBits
        //     unsigned long long* output_ptr = reinterpret_cast<unsigned long long*>(&output[outputIdx]);
        //     output_ptr[0] = bitSize;
        //     output_ptr[1] = firstValueBits;

        // }
        // 6.3. 写入 maxDecimalPlaces 和 bitCount (各1字节)
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        // 6.4 写入flag1(8字节 标识稀疏)
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        }
        // printf("In %d  flag1 is : %llx\n",idx,flag1);
        // 6.5 写入每一列
        int flag2Byte=(numByte+7)/8;
        int ofs=outputIdx + 27;
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

    // if(idx==49)
    // {
    //     __syncthreads();
    //     printf("\nfinish idx:%d outputIdx:%d\n",idx,outputIdx);
    //     for(int i = outputIdx; i < outputIdx+(bitSize+7)/8; i++) {
    //         printf("0x%02x ", output[i]);
    //         // 每 16 个字节换行
    //         if ((i + 1) % 16 == 0) {
    //             printf("\n");
    //         }
    //     }
    // }
    //printf("\nofs:%d\n",ofs);
}


// 初始化设备内存
void GDFCompressor::setupDeviceMemory(
    const std::vector<double>& input,
    double*& d_input,
    unsigned char*& d_output,
    uint64_t*& d_bitSizes
) {
    size_t inputSize = input.size();
    int numBlocks = (inputSize + DATA_PER_THREAD - 1) / (DATA_PER_THREAD);

    // 分配输入数据的设备内存
    cudaCheckError(cudaMalloc((void**)&d_input, inputSize * sizeof(double)));
    cudaCheckError(cudaMemcpy(d_input, input.data(), inputSize * sizeof(double), cudaMemcpyHostToDevice));

    // 分配输出数据的设备内存
    cudaCheckError(cudaMalloc((void**)&d_output, numBlocks * MAX_BYTES_PER_BLOCK * sizeof(unsigned char)));

    // 分配 bitSizes 的设备内存
    cudaCheckError(cudaMalloc((void**)&d_bitSizes, numBlocks * sizeof(uint64_t)));

}

// 更新释放设备内存的函数
void GDFCompressor::freeDeviceMemory(
    double* d_input,
    unsigned char* d_output,
    uint64_t* d_bitSizes
) {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bitSizes);
}


// 主压缩函数
void GDFCompressor::compress(const std::vector<double>& input, std::vector<unsigned char>& output) {
    //std::cout<<"begin1\n";
    size_t inputSize = input.size();
    if (inputSize == 0) return;


    int blockSize = BLOCK_SIZE_G; // 32个线程每块
    size_t numBlocks = (inputSize + blockSize * DATA_PER_THREAD - 1) / (blockSize * DATA_PER_THREAD); // 多少个线程块
    size_t numthread = (inputSize + DATA_PER_THREAD - 1) / (DATA_PER_THREAD); // 多少个数据块(线程)
    double* d_input = nullptr;
    unsigned char* d_output = nullptr;
    uint64_t* d_bitSizes = nullptr;

    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    int cmpOffSize = numBlocks + 1;
    cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);

    cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);

    cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);
    // 分配设备内存
    setupDeviceMemory(input, d_input, d_output, d_bitSizes);


    size_t sharedMemSize = 64; // 确保 SharedMemory 已正确定义
    //std::cout<<"begin2\n";

    // 创建CUDA事件用于计时
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // 记录开始事件
    cudaCheckError(cudaEventRecord(start));
    
    // 启动核函数
    compressBlockKernel<<<numBlocks, blockSize, sharedMemSize>>>(
        d_input,
        inputSize,
        d_output,
        d_bitSizes,
        d_cmpOffset,
        d_locOffset,
        d_flag
    );
    // 检查错误
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());
    //std::cout<<"end2\n";

    // 记录结束事件
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaEventSynchronize(stop)); // 等待事件完成

    // 计算耗时（毫秒）
    float milliseconds = 0;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));

    // 计算吞吐量
    size_t dataSizeBytes = input.size() * sizeof(double); // 原始数据量
    float seconds = milliseconds / 1000.0f;

    // MB/s = (bytes / 1e6) / seconds
    float throughputMBs = (dataSizeBytes / 1e6) / seconds; 
    
    // GB/s = (bytes / 1e9) / seconds
    float throughputGBs = (dataSizeBytes / 1e9) / seconds;

    // 打印结果（保留两位小数）
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "压缩核函数时间" << milliseconds
              << " ms. \nThroughput: "
              << throughputMBs << " MB/s ("
              << throughputGBs << " GB/s)" 
              << std::endl;

    // 清理事件
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));


    // 复制 bitSizes 回主机

    std::vector<uint64_t> bitSizes(numthread);
    cudaCheckError(cudaMemcpy(bitSizes.data(), d_bitSizes, numthread * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    //std::cout<<"end3\n";

    // 计算每个块的输出偏移量
    std::vector<uint64_t> offsets(numthread, 0);
    uint64_t totalCompressedBits = 0;
    for (size_t i = 0; i < numthread; i++) {
        offsets[i] = totalCompressedBits;
        totalCompressedBits += bitSizes[i];
        // std::cerr << " In" << i <<" block, bitSize is : "<< bitSizes[i] <<"\n";
    }
    //    std::cout<<"end4\n";

    uint64_t totalCompressedBytes = (totalCompressedBits + 7) / 8; // 按字节对齐
    // printf("outputSize(bits):%d\n",totalCompressedBits);

    // // 分配输出缓冲区
    output.resize(totalCompressedBytes, 0);
    // printf("outputSize(byte):%d\n",totalCompressedBytes);
    // 复制 d_output 到主机的临时缓冲区
    std::vector<unsigned char> tempOutput(totalCompressedBytes);
    cudaCheckError(cudaMemcpy(tempOutput.data(), d_output,totalCompressedBytes * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    output = std::move(tempOutput);
        //把output中totalCompressedBits到totalCompressedBytes*8部分的bit位数置0
        // int remainingBits = totalCompressedBits % 8; // 计算最后一个字节的有效位数
        // if (remainingBits > 0) {
        //     unsigned char mask = (1 << remainingBits) - 1; // 创建掩码，例如如果有效位数为 5，则掩码为 0b00011111
        //     output[totalCompressedBytes - 1] &= mask;     // 将无效位清零
        // }
    //std::cout<<"end5\n";
    // 释放设备内存
    freeDeviceMemory(d_input, d_output, d_bitSizes);
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}


__global__ void GDFC_compress_kernel(
    const double* input,
    unsigned char* output,
    volatile unsigned int* const __restrict__ cmpOffset, // 压缩数据偏移量数组（输出）
    volatile unsigned int* const __restrict__ locOffset, // 局部偏移量数组（输出）
    volatile int* const __restrict__ flag,             // 标志数组，用于同步不同warp的状态（输出
    int totalSize
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
    int endIdx = min(startIdx + DATA_PER_THREAD, totalSize);
    int numDatas = endIdx - startIdx;
    uint64_t deltas[DATA_PER_THREAD];
    //int numDeltas = numDatas - 1;
    // if(numDatas<=0)
    // {
    //     return;
    // }
    // 局部变量
    int maxDecimalPlaces = 0;
    int maxBeta =0;
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
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);

        double alpha = getDecimalPlaces(value, sp);// 得到小数位数
        double beta =  alpha + sp + 1;
        maxBeta = device_max(maxBeta,beta);
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    // 2. FOR + zigzag（用4向量化进行实现）
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); // 量化第一个
    prevQuant = firstValue;// 初始化第一个量化值
    for(int j = 0; j < (numDatas+31) / 32; j++) { // 每个线程的数量 / 一个数据批次（32）
        base_block_start_idx = startIdx + j * 32;           //每一组32个数据的起始位置
        base_block_end_idx = base_block_start_idx + 32;     //每一组32个数据的结束位置

        if(base_block_end_idx < totalSize) {
                int i = base_block_start_idx;
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
                    currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); // 量化当前数据点
                    lorenQuant = currQuant - prevQuant; // 计算差分

                    deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
                    prevQuant = currQuant; // 更新前一个量化值
                    maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
                }

                // 处理y分量
                currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

                // 处理z分量
                currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

                // 处理w分量
                currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+3],quant_chunk_idx+3,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);
                i+=4;
            #pragma unroll 7 //循环展开8次，就是4*8=32个数据,修改为7次，把第一次提取出来
            for(; i < base_block_end_idx; i += 4) {

                tmp_buffer = reinterpret_cast<const double4*>(input)[i / 4];
                quant_chunk_idx = j * 32 + (i % 32); //处理的每一组的第几个数据

                // 处理x分量
                // if(i == startIdx) { // 针对第一个数据
                //     // firstValue = double2long(tmp_buffer.x, maxDecimalPlaces); // 量化当前数据点
                //     // prevQuant = firstValue;
                //     // printf("firstValue:%d\n",firstValue);
                //     deltas[quant_chunk_idx] = 0;//填充第一个数据为0保证1024个数据
                // }
                // else {
                currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); // 量化当前数据点
                lorenQuant = currQuant - prevQuant; // 计算差分

                deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant; // 更新前一个量化值
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
                // }

                // 处理y分量
                currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

                // 处理z分量
                currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

                // 处理w分量
                currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
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
                int remainbEle = totalSize - base_block_start_idx;  // 剩余有效数据元素数
                int zeronbEle = base_block_end_idx - totalSize;     // 超出范围的数据元素数

                // 处理剩余有效数据元素
                for(int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++) {
                    if(i==startIdx)
                    {
                        deltas[0]=0;
                        continue;
                    }
                    quant_chunk_idx = j * 32 + (i % 32);
                    currQuant = double2long(input[i], maxDecimalPlaces,maxBeta);

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

                quant_chunk_idx = j * 32 + (totalSize % 32);
                for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
                    deltas[i] = 0;
            }
        }
    }

    // 3. 得到编码过后的最大delta值，并且得到最大bit位
    // bitCount = 0;

    // while (maxDelta > 0) {
    //     maxDelta >>= 1;
    //     bitCount++;
    // }

    // // 防止 bitCount 为 0（所有 deltas 都为 0）
    // if (bitCount == 0) {
    //     bitCount = 1;
    // }
    // // 限制 bitCount 不超过 MAX_BITCOUNT
    // bitCount = device_min(bitCount, MAX_BITCOUNT);
    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//用内置函数 替代处理循环
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

    // // 使用寄存器存储局部变量
    // register int local_bitCount = bitCount;

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
            int j=0;
                // while(j+32<numDatas)
                // {
                //     int byteIndex = j / 8;  // 当前bit属于第几个字节
                //     result[i][byteIndex] = result[i][byteIndex] |
                //                             (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << (0));

                //     result[i][byteIndex+1] = result[i][byteIndex+1] |
                //                             (((deltas[j+8] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+9] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+10] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+11] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+12] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+13] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+14] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+15] >> (bitCount - 1 - i)) & 1) << (0));

                //     result[i][byteIndex+2] = result[i][byteIndex+2] |
                //                             (((deltas[j+16] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+17] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+18] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+19] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+20] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+21] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+22] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+23] >> (bitCount - 1 - i)) & 1) << (0));

                //     result[i][byteIndex+3] = result[i][byteIndex+3] |
                //                             (((deltas[j+24] >> (bitCount - 1 - i)) & 1) << (7))|
                //                             (((deltas[j+25] >> (bitCount - 1 - i)) & 1) << (6))|
                //                             (((deltas[j+26] >> (bitCount - 1 - i)) & 1) << (5))|
                //                             (((deltas[j+27] >> (bitCount - 1 - i)) & 1) << (4))|
                //                             (((deltas[j+28] >> (bitCount - 1 - i)) & 1) << (3))|
                //                             (((deltas[j+29] >> (bitCount - 1 - i)) & 1) << (2))|
                //                             (((deltas[j+30] >> (bitCount - 1 - i)) & 1) << (1))|
                //                             (((deltas[j+31] >> (bitCount - 1 - i)) & 1) << (0));
                //     j+=32;
                // }
            while(j+8<numDatas)//有效 0.0027->0.0023
            {
                int byteIndex = j / 8;  // 当前bit属于第几个字节
                result[i][byteIndex] = result[i][byteIndex] |
                                        (((deltas[j] >> (bitCount - 1 - i)) & 1) << (7))|
                                        (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << (6))|
                                        (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << (5))|
                                        (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << (4))|
                                        (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << (3))|
                                        (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << (2))|
                                        (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << (1))|
                                        (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << (0));
                j+=8;
            }
            for (; j <numDatas ; ++j) {//列numBytes
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
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;              // 用于记录每一列是否为稀疏列
        uint8_t flag2[64][16];          // 对于稀疏列统计稀疏位置,最多1024个数据，所以最多1024bit，即128byte,
        memset(flag2, 0, sizeof(flag2));
                                        //每一个byte用1bit标识，所以最多16byte
        //很重要的一点：flag2是byte单位
        // for( int i=0;i<bitCount;++i)
        // {
        //     int b0=0;
        //     int b1=0;
        //     // int s=0;
        //     for(int j=0;j<numByte;++j)
        //     {
        //         int m_byte =j/8; //flag2的第几个byte位
        //         int m_bit =j%8;  //flag2的这个byte位的第几个bit
        //         if(result[i][j]==0)//如果是0可以稀疏化
        //         {
        //             b0++;   //全0byte 只用1个标识位
        //             flag2[i][m_byte] &= ~(1ULL << m_bit);//设置flag2的第i行（和flag对应）的对应字节的bit位为0
        //         }
        //         else
        //         {
        //             b1++;//有效的byte 多用一位进行标识
        //             flag2[i][m_byte] |= (1ULL << m_bit); //设置flag2的第i行（和flag对应）的对应字节的bit位为1
        //         }
        //     }
        //     if (((numByte+7)/8+b1)>= numByte) {//存储flag2的单位也是BYTE,有numByte位
        //         // 浪费了，设置 flag1 的第 i 位为 0
        //         flag1 &= ~(1ULL << i); // 使用按位与和按位取反清除第 i 位

        //         // 4.3 非稀疏列处理
        //         bitSize += 8 * numByte;

        //     } else {
        //         // 设置 flag1 的第 i 位为 1
        //         flag1 |= (1ULL << i); // 使用按位或设置第 i 位为 1

        //         // 4.4 稀疏列处理
        //         bitSize += ((numByte+7)/8+b1)*8;    //用byte为单位*8得到bit
        //     }
        // }
        int BITS_PER_THREAD=4;
        for(int i = 0; i < bitCount; i += BITS_PER_THREAD) { // 每次处理4个比特位
            for(int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b) {
                int bit = i + b;
                int b0 = 0;
                int b1 = 0;
                for(int j = 0; j < numByte; j++) {
                    int m_byte = j / 8;
                    int m_bit = j % 8;
                    uint8_t current_result = result[bit][j];
                    b0 += (current_result == 0);
                    b1 += (current_result != 0);
                    flag2[bit][m_byte] |= (current_result != 0) << m_bit;//设置1
                    flag2[bit][m_byte] &= ~((current_result == 0) << m_bit);//清零
                }
                // 使用掩码和算术操作代替分支(有效0.0023->0.0021)
                uint64_t is_sparse = ((numByte + 7) / 8 + b1) < numByte;
                flag1 |= (is_sparse << bit);
                flag1 &= ~((!is_sparse) << bit);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
                //flag2的长度+b1或者numByte*8
            }
        }

        // for (int i = 0; i < bitCount; ++i) {
        //     for (int j = 0; j < numBytes; ++j) {
        //         printf("0x %02x ", result[i][j]); // 打印为十六进制，且确保每个字节以2位输出
        //     }
        //     printf("\n");
        // }
        if(numDatas<=0)
        {
            bitSize=0;
        }
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
        // printf("thread_ofs[%d]:%d",lane,thread_ofs);

        // 5.2 Warp(块)内最后一个线程更新locOffset和flag数组
        if(lane == 31||numDatas<=0)//或者最后一个线程出现，但是不是第32个线程
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
                    // for(int i=0;i<=warp+1;i++)
                    // {
                    //     printf("cmpOffset[%d] :%d\n",i,cmpOffset[i]);
                    // }
                }
                flag[warp] = 2;             // 标记当前warp完成
                //printf("flag[%d] over2\n",warp);
                __threadfence();
            }
        }
        __syncthreads(); // 同步线程，确保cmpOffset更新完成
        if(numDatas<=0)
        {
            if(cmpOffset[warp + 1]<=0)
            {
                cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1];
            }
            return;
        }
        // 5.4 得到写入位置
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap偏移+wrap内偏移 得到当前压缩后数据应该写入的起始位置
        int outputIdx = (outputIdxBit+7)/8;
        // if((idx+1)%256==0)
        // {
        //     printf("idx:%d outputIdx:%d bitSize :%d \n",idx,outputIdx,bitSize );
        // }
        
    // 6 开始写入
        // printf("excl_sum:%d\n",excl_sum);
        // printf("thread_ofs:%d\n",thread_ofs);
        // printf("bitSize:%d\n",bitSize);
        // printf("firstValue:%d\n",firstValue);
        // printf("maxDecimalPlaces:%d\n",maxDecimalPlaces);
        // printf("bitCount:%d\n",bitCount);
        // printf("flag1:%016llx\n",flag1);


        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        // if (outputIdx % 8 != 0) {
        // 6.1 写入 bitSize (8 字节)
            for(int i = 0; i < 8; i++) {
                output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            }


            // 6.2. 写入 firstValue (8 字节)
            for(int i = 0; i < 8; i++) {
                output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            }
        // }
        // else
        // {
        //     // 6.1+6.2优化 示例：一次性写入 bitSize 和 firstValueBits
        //     unsigned long long* output_ptr = reinterpret_cast<unsigned long long*>(&output[outputIdx]);
        //     output_ptr[0] = bitSize;
        //     output_ptr[1] = firstValueBits;

        // }
        // 6.3. 写入 maxDecimalPlaces 和 bitCount (各1字节)
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        // 6.4 写入flag1(8字节 标识稀疏)
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        }
        // printf("In %d  flag1 is : %llx\n",idx,flag1);
        // 6.5 写入每一列
        int flag2Byte=(numByte+7)/8;
        int ofs=outputIdx + 27;
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

    // if(idx==49)
    // {
    //     __syncthreads();
    //     printf("\nfinish idx:%d outputIdx:%d\n",idx,outputIdx);
    //     for(int i = outputIdx; i < outputIdx+(bitSize+7)/8; i++) {
    //         printf("0x%02x ", output[i]);
    //         // 每 16 个字节换行
    //         if ((i + 1) % 16 == 0) {
    //             printf("\n");
    //         }
    //     }
    // }
    //printf("\nofs:%d\n",ofs);
    // if(idx==1)
    // {
    //     for(int i = 0; i < 10; i++) {
    //         printf("0x%02x  ", output[i]);
            
    //         // 每 16 个字节换行
    //         if ((i + 1) % 16 == 0) {
    //             printf("\n");
    //         }
    //     }
    // }
}

void GDFCompressor::GDFC_compress(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream)
{

    // if(nbEle>MAX_NUMS_PER_CHUNK)
    // {
    //     printf("Failed to get max output size\n");
    //     return ;
    // }
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    unsigned int glob_sync;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    printf("run\n");
    GDFC_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(&glob_sync, d_cmpOffset+cmpOffSize-1, sizeof(unsigned int), cudaMemcpyDeviceToHost,stream);
    
    *cmpSize = ((size_t)glob_sync+7)/8;//+ (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;
    
    // printf("g_offset:%d\n",cmpOffSize-1);
    // *cmpSize = (totalCompressedBits + 7) / 8; // 按字节对齐
    // Free memory that is used.
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}


void GDFCompressor::GDFC_compress_stream(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
{
    // Data blocking.
    int bsize = cmp_tblock_size;
    int gsize = (nbEle + bsize * DATA_PER_THREAD - 1) / (bsize * DATA_PER_THREAD);
    // size_t numthread = (nbEle + cmp_chunk - 1) / (cmp_chunk);
    int cmpOffSize = gsize + 1;

    // Initializing global memory for GPU compression.
    unsigned int* d_cmpOffset;
    unsigned int* d_locOffset;
    int* d_flag;
    cudaMallocAsync((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMemsetAsync(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize,stream);
    cudaMallocAsync((void**)&d_flag, sizeof(int)*cmpOffSize,stream);
    cudaMemsetAsync(d_flag, 0, sizeof(int)*cmpOffSize,stream);
    // cudaMalloc((void**)&d_cmpOffset, sizeof(unsigned int)*cmpOffSize);
    // cudaMemset(d_cmpOffset, 0, sizeof(unsigned int)*cmpOffSize);
    // cudaMalloc((void**)&d_locOffset, sizeof(unsigned int)*cmpOffSize);
    // cudaMemset(d_locOffset, 0, sizeof(unsigned int)*cmpOffSize);
    // cudaMalloc((void**)&d_flag, sizeof(int)*cmpOffSize);
    // cudaMemset(d_flag, 0, sizeof(int)*cmpOffSize);

    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    GDFC_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);


    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);

    // printf("g_offset:%d\n",cmpOffSize-1);
    // *cmpSize = (totalCompressedBits + 7) / 8; // 按字节对齐
    // Free memory that is used.
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);

    // cudaFree(d_cmpOffset);
    // cudaFree(d_locOffset);
    // cudaFree(d_flag);
}