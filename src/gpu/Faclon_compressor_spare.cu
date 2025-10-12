//
// Created by lz on 24-9-26.
//
#include "Faclon_compressor.cuh"
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
    // double deltaBound = pow(2,ilogb(temp)-52);
    while (abs(temp - trac) >= deltaBound * td && digits < 16 - sp - 1)
    {
        digits++;
        td = pow10_table[digits];
        temp = value * td;
        // double deltaBound = pow(2,ilogb(temp)-52);
        trac = temp + POW_NUM_G - POW_NUM_G;
    }

    return digits;
}

__device__ static int getDecimalPlaces_br(double v,int sp) {
//     value = value < 0 ? -value : value;
//     double trac = value + POW_NUM_G - POW_NUM_G;
//     double temp = value;

//     int digits = 0;
//     double td = 1;
//     double deltaBound = abs(value) * pow(2, -52);
//     while (abs(temp - trac) == 0 && digits < 16 - sp - 1)
//     {
//         digits++;
//         td = pow10_table[digits];
//         temp = value * td;
//         trac = temp + POW_NUM_G - POW_NUM_G;
//     }

//     return digits;
// }
    v = v < 0 ? -v : v;
    
    int i = 0;
    double scale = 1.0;
    
    // 找到能精确表示为整数的最小倍数
    while (i < 17) {
        double temp = v * scale;
        
        if (round(temp) == temp) {

            return i;
        }
        i++;
        scale *= 10.0;
    }
    return 17; // 达到双精度极限
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

    return (maxBeta > 15 ||maxDecimalPlaces>15)
    ? (encodeDoubleWithSignLast(data))
    : static_cast<long>(round(data * pow10_table[maxDecimalPlaces]));
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
    int numDatas = max(0,endIdx - startIdx);
    
    uint64_t deltas[DATA_PER_THREAD]={0};

    if((idx-1)*DATA_PER_THREAD>totalSize){
        return;
    }
    int maxDecimalPlaces = 0;
    long firstValue = 0;
    volatile int bitCount = 0;
    volatile int maxBeta = 0;

    int base_block_start_idx=0, base_block_end_idx=0;
    // int quant_chunk_idx;
    // int block_idx; // 如果不使用，可以移除

    long currQuant=0;
    long lorenQuant=0;
    long prevQuant=0;

    unsigned int thread_ofs = 0;

    int maxSp = -99;
    // 1. 采样
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);
        maxSp = device_max(maxSp, sp);
        double alpha = getDecimalPlaces(value, sp);// 得到小数位数
        // double beta =  alpha + sp + 1;
        // maxBeta = device_max(maxBeta,beta);
        // if(alpha>maxDecimalPlaces){
        //     idwrong = i;
        // }
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }

    maxBeta = maxSp + maxDecimalPlaces+1;
    
    // for (int i = 0; i < numDatas; i++) {
    //     double value =input[startIdx + i];
    //     if(value == -4.0023584){
    //     printf("maxBeta:%d, maxAlpha: %d, maxSp: %d, wrongAlpha:%.16f \n", maxBeta, maxDecimalPlaces, maxSp,input[startIdx + idwrong]);
    // }
    // }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    // 2. FOR + zigzag（用4向量化进行实现）
    volatile uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); // 量化第一个
    prevQuant = firstValue;// 初始化第一个量化值
    base_block_start_idx = startIdx + 1;

    for(int i=0;i<numDatas-1;i++){
        currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); // 量化当前数据点
        lorenQuant = currQuant - prevQuant; // 计算差分
        deltas[i] = zigzag_encode_cuda(lorenQuant);
    
        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;

    }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//用内置函数 替代处理循环
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

        const int numByte = (numDatas - 1 + 7) / 8;
        uint8_t result_flat[8192] = {};

        for (int i = 0; i < bitCount; ++i) {//行
            int j=0;
            while((j+8+1)<numDatas)
            {
                int byteIndex = j / 8;  // 当前bit属于第几个字节
                uint8_t currentByte = 0;
                currentByte |= (((deltas[j] >> (bitCount - 1 - i)) & 1) << 7);
                currentByte |= (((deltas[j+1] >> (bitCount - 1 - i)) & 1) << 6);
                currentByte |= (((deltas[j+2] >> (bitCount - 1 - i)) & 1) << 5);
                currentByte |= (((deltas[j+3] >> (bitCount - 1 - i)) & 1) << 4);
                currentByte |= (((deltas[j+4] >> (bitCount - 1 - i)) & 1) << 3);
                currentByte |= (((deltas[j+5] >> (bitCount - 1 - i)) & 1) << 2);
                currentByte |= (((deltas[j+6] >> (bitCount - 1 - i)) & 1) << 1);
                currentByte |= (((deltas[j+7] >> (bitCount - 1 - i)) & 1) << 0);
                
                result_flat[i * numByte + byteIndex] = currentByte;
                j+=8;
            }
            for (; j <(numDatas -1); ++j) {//列numBytes

                int byteIndex = j / 8;  // 当前bit属于第几个字节
                int bitIndex = j % 8;   // 当前bit在字节中的位置

                uint8_t bitVal = ((deltas[j] >> (bitCount - 1 - i)) & 1);
                // 提取当前bit位并存入结果数组


                result_flat[i * numByte + byteIndex] |= bitVal << (7 - bitIndex);


            }

        }


        // 4.2 设置稀疏列，并且进行标记，同时计算bitsize
        uint64_t bitSize =  64ULL +                 // bitsize
                            64ULL +                 // firstValue
                            8ULL +                  // maxDecimalPlaces
                            8ULL +                  // maxBeta
                            8ULL +                  // bitCount
                            64ULL;                  // flag1

        uint64_t flag1 = 0;
        uint8_t flag2[(DATA_PER_THREAD-1)];

        memset(flag2,0,sizeof(flag2));

        for(int i = 0;i<bitCount;i++){
            int b0 = 0;
            int b1 = 0;

            // 当前行在扁平化 result_flat 和 flag2_flat 中的起始偏移
            size_t result_row_start_offset = i * numByte;
            size_t flag2_row_start_offset = i * ((numByte + 7) / 8); // flag2 每一行的字节数
        
            for (int j = 0; j < numByte; j++) {
                uint8_t current_result_byte = result_flat[result_row_start_offset + j];

                b0 += (current_result_byte == 0);
                b1 += (current_result_byte != 0);

                int flag2_byte_idx = j / 8;
                int flag2_bit_idx = j % 8;

                uint8_t mask = (1 << flag2_bit_idx);
                // 原子操作更新 flag2 字节，避免写冲突
                if (current_result_byte != 0) {
                    flag2[flag2_row_start_offset + flag2_byte_idx]|= (current_result_byte != 0) << flag2_bit_idx;
                } else {
                    flag2[flag2_row_start_offset + flag2_byte_idx]&= ~((current_result_byte == 0) << flag2_bit_idx);
                }
            }
            uint64_t is_sparse = (uint64_t)(((numByte + 7) / 8 + b1) < numByte);
                flag1|=(1<< i);
                // flag1 |= (is_sparse << i);
                // flag1 &= ~((!is_sparse) << i);
                bitSize += is_sparse ? ((numByte + 7) / 8 + b1) * 8 : 8 * numByte;
        }


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
        else {
            // warp==0：显式把排他前缀和置 0（由一个线程写，block 内可见）
            if (!lane) { excl_sum = 0; }
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

    // 6 开始写入

    


        // unsigned long long firstValueBits = 0;
        // memcpy(&firstValueBits, &firstValue, sizeof(long));
        // if (outputIdx % 8 != 0) {
        // 6.1 写入 bitSize (8 字节)
            // for(int i = 0; i < 8; i++) {
            //     output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            // }


            // // 6.2. 写入 firstValue (8 字节)
            // for(int i = 0; i < 8; i++) {
            //     output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            // }
        memcpy(output + outputIdx, &bitSize, sizeof(unsigned long long));

        // 6.2. 写入 firstValue (8 字节)
        // 直接将 firstValue (double) 的字节内容复制到 output 数组
        memcpy(output + outputIdx + 8, &firstValue, sizeof(double)); // 注意这里用的是 sizeof(double)
            

        // }
        // 6.3. 写入 maxDecimalPlaces 和 bitCount (各1字节)
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        // 6.4 写入flag1(8字节 标识稀疏)
        // for(int i = 0; i < 8; i++) {
        //     output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        // }

        memcpy(output + outputIdx + 19, &flag1, sizeof(unsigned long long));
        // printf("In %d  flag1 is : %llx\n",idx,flag1);
        // 6.5 写入每一列
        int flag2Byte = (numByte+7)/8;
        int ofs=outputIdx + 27;
        //int res=0;              //byte中剩余的bit位
        for(int i=0;i<bitCount;i++)
        {
            size_t flag2_row_start_offset = i * flag2Byte;
            size_t result_row_start_offset = i * numByte;
            if((flag1 & (1ULL << i)) != 0){
                memcpy(output + ofs, flag2 + flag2_row_start_offset, flag2Byte);
                ofs += flag2Byte;
                for (int j = 0; j < numByte; j++) {
                    if (result_flat[result_row_start_offset + j]) {
                        output[ofs++] = result_flat[result_row_start_offset + j];
                    }
                }
            } else { // 非稀疏
                memcpy(output + ofs, result_flat + result_row_start_offset, numByte);
                ofs += numByte;
            }

                
        }

}


//返回的cmpSize是BYTE
void GDFCompressor_opt::GDFC_compress(double* d_oriData, unsigned char* d_cmpBytes, size_t nbEle, size_t* cmpSize, cudaStream_t stream)
{

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
    cudaCheckError(cudaGetLastError());
    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    // printf("run\n");
    GDFC_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);
    // 在主函数中调用
    cudaStreamSynchronize(stream);
    cudaCheckError(cudaGetLastError());

    
    // Obtain compression ratio and move data back to CPU.  
    
        cudaMemcpyAsync(&glob_sync, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        cudaCheckError(cudaGetLastError());
        *cmpSize = ((size_t)glob_sync+7)/8;//+ (nbEle+cmp_tblock_size*cmp_chunk-1)/(cmp_tblock_size*cmp_chunk)*(cmp_tblock_size*cmp_chunk)/32;

    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
    
}

//返回bits
void GDFCompressor_opt::GDFC_compress_stream(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
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


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    GDFC_compress_kernel<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // printf("beta %.d ",d_cmpBytes[16]);
    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}


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
    
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); // 量化第一个
    prevQuant = firstValue;                                              // 初始化第一个量化值
    for (int j = 0; j < (numDatas + 30) / 32; j++)
    {                                                       // 每个线程的数量 / 一个数据批次（32）
        base_block_start_idx = startIdx + j * 32 + 1;       // 每一组32个数据的起始位置
        base_block_end_idx = base_block_start_idx + 32 + 1; // 每一组32个数据的结束位置

        if (base_block_end_idx < totalSize)
        {
            int i = base_block_start_idx;
            // tmp_buffer = reinterpret_cast<const double4*>(input)[i / 4];
            // quant_chunk_idx = j * 32 + (i % 32); //处理的每一组的第几个数据

            // // 处理x分量
            // if(i == startIdx) { // 针对第一个数据

            //     deltas[quant_chunk_idx] = 0;//填充第一个数据为0保证1024个数据
            // }
            // else {
            //     currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); // 量化当前数据点
            //     lorenQuant = currQuant - prevQuant; // 计算差分

            //     deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
            //     prevQuant = currQuant; // 更新前一个量化值
            //     maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
            // }

            // // 处理y分量
            // currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
            // lorenQuant = currQuant - prevQuant;

            // deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
            // prevQuant = currQuant;
            // maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

            // // 处理z分量
            // currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
            // lorenQuant = currQuant - prevQuant;

            // deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
            // prevQuant = currQuant;
            // maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

            // // 处理w分量
            // currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
            // lorenQuant = currQuant - prevQuant;

            // deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
            // prevQuant = currQuant;
            // maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);
            // i+=4;
            #pragma unroll 8 //循环展开8次，就是4*8=32个数据,修改为7次，把第一次提取出来
            for(; i < base_block_end_idx; i += 4) {

                tmp_buffer = reinterpret_cast<const double4*>(input+1)[(i-1) / 4];
                quant_chunk_idx = j * 32 + ((i-1) % 32); //处理的每一组的第几个数据

                currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); // 量化当前数据点
                lorenQuant = currQuant - prevQuant; // 计算差分

                deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant; // 更新前一个量化值
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
                // }

                // 处理y分量
                currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

                // 处理z分量
                currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
                prevQuant = currQuant;
                maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

                // 处理w分量
                currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
                lorenQuant = currQuant - prevQuant;

                deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
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
                for (int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++)
                {
                    if (i == startIdx)
                    {
                        deltas[0]=0;
                        continue;
                    }
                    quant_chunk_idx = j * 32 + (i % 32);
                    currQuant = double2long(input[i], maxDecimalPlaces,maxBeta);

                    lorenQuant = currQuant - prevQuant;

                    deltas[ quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
                    prevQuant = currQuant;
                    maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]);
                }

                quant_chunk_idx = j * 32 + (totalSize % 32);
                for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
                    deltas[i] = 0;
            }
        }
    }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//用内置函数 替代处理循环
    bitCount = min(bitCount, (int)MAX_BITCOUNT);

        int numByte = (numDatas-1 + 7) / 8;
        // uint8_t result[64][128];
        uint8_t result[64][128] = {}; //0初始化
        // 初始化二维数组

        // 遍历每个uint64_t的数据
        for (int i = 0; i < bitCount; ++i) {//行
            int j=0;
            while(j+8<numDatas-1)//有效 0.0027->0.0023
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
            for (; j <numDatas-1 ; ++j) {//列numBytes
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
        int BITS_PER_THREAD=4;
        for (int i = 0; i < bitCount; i += BITS_PER_THREAD)
        { // 每次处理4个比特位
            for (int b = 0; b < BITS_PER_THREAD && (i + b) < bitCount; ++b)
            {
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
                }
                excl_sum = loc_excl_sum; // 存储排他性前缀和
                // 2.3 更新cmpOffset数组
                cmpOffset[warp] = excl_sum; // 更新当前warp的cmpOffset
                __threadfence();           // 确保写操作完成

                if(warp == gridDim.x - 1)
                {
                    cmpOffset[warp + 1] = cmpOffset[warp] + locOffset[warp + 1]; // 更新最后一个warp的cmpOffset
                    __threadfence();
                }
                flag[warp] = 2;             // 标记当前warp完成
                __threadfence();
            }
        }
        else {
            // warp==0：显式把排他前缀和置 0（由一个线程写，block 内可见）
            if (!lane) { excl_sum = 0; }
        }
        __syncthreads(); // 同步线程，确保cmpOffset更新完成

        // 5.4 得到写入位置
        int outputIdxBit = excl_sum + thread_ofs - bitSize; //bit wrap偏移+wrap内偏移 得到当前压缩后数据应该写入的起始位置
        int outputIdx = (outputIdxBit+7)/8;

        bitSizes[idx] = bitSize;

        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));
        // 6.1 写入 bitSize (8 字节)
            for(int i = 0; i < 8; i++) {
                output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

            }


            // 6.2. 写入 firstValue (8 字节)
            for(int i = 0; i < 8; i++) {
                output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

            }
        // 6.3. 写入 maxDecimalPlaces 和 bitCount (各1字节)
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        // 6.4 写入flag1(8字节 标识稀疏)
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 19 + i] = (flag1 >> (i * 8)) & 0xFF;
        }
        // 6.5 写入每一列
        int flag2Byte=(numByte+7)/8;
        int ofs=outputIdx + 27;
        for(int i=0;i<bitCount;i++)
        {
            if((flag1 & (1ULL << i)) != 0)//flag第i个bit不为0:稀疏
            {
                // 6.5.1 稀疏列写入flag2+data
                for(int j=0;j<flag2Byte;j++)
                {
                    output[ofs++] = static_cast<unsigned char>(flag2[i][j]);
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


// 初始化设备内存
void GDFCompressor_opt::setupDeviceMemory(
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
void GDFCompressor_opt::freeDeviceMemory(
    double* d_input,
    unsigned char* d_output,
    uint64_t* d_bitSizes
) {
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_bitSizes);
}


// 主压缩函数
void GDFCompressor_opt::compress(const std::vector<double>& input, std::vector<unsigned char>& output) {
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

    }

    uint64_t totalCompressedBytes = (totalCompressedBits + 7) / 8; // 按字节对齐

    // // 分配输出缓冲区
    output.resize(totalCompressedBytes, 0);
    // 复制 d_output 到主机的临时缓冲区
    std::vector<unsigned char> tempOutput(totalCompressedBytes);
    cudaCheckError(cudaMemcpy(tempOutput.data(), d_output,totalCompressedBytes * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    output = std::move(tempOutput);

    freeDeviceMemory(d_input, d_output, d_bitSizes);
    cudaFree(d_cmpOffset);
    cudaFree(d_locOffset);
    cudaFree(d_flag);
}


__global__ void GDFC_compress_kernel_no_pack(
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
    // zhege difang youdian wenti 
    //zheyang yinggai jiu ke yi le 
    int maxSp = -99;
    // 1. 采样
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);
        maxSp = device_max(maxSp, sp);
        double alpha = getDecimalPlaces(value, sp);// 得到小数位数
        // double beta =  alpha + sp + 1;
        // maxBeta = device_max(maxBeta,beta);
        // if(alpha>maxDecimalPlaces){
        //     idwrong = i;
        // }
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }

    maxBeta = maxSp + maxDecimalPlaces+1;
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    // 2. FOR + zigzag（用4向量化进行实现）
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); // 量化第一个
    prevQuant = firstValue;// 初始化第一个量化值
    base_block_start_idx = startIdx + 1;

    for(int i=0;i<numDatas-1;i++){
        currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); // 量化当前数据点
        lorenQuant = currQuant - prevQuant; // 计算差分
        deltas[i] = zigzag_encode_cuda(lorenQuant);
    
        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;
    }

    // 3. 得到编码过后的最大delta值，并且得到最大bit位
    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//用内置函数 替代处理循环
    bitCount = min(bitCount, (int)MAX_BITCOUNT);


    // 4. 并且进行标记，同时计算bitsize
    uint64_t bitSize =  64ULL +                 // bitsize
                        64ULL +                 // firstValue
                        8ULL +                  // maxDecimalPlaces
                        8ULL +                  // maxBeta
                        8ULL +                  // bitCount
                        ((uint64_t)numDatas-1) * ((uint64_t)bitCount);

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
        else {
            // warp==0：显式把排他前缀和置 0（由一个线程写，block 内可见）
            if (!lane) { excl_sum = 0; }
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
        
    // 6 开始写入
        unsigned long long firstValueBits = 0;
        memcpy(&firstValueBits, &firstValue, sizeof(long));

        // 6.1 写入 bitSize (8 字节)
        for(int i = 0; i < 8; i++) {
            output[outputIdx + i] = (bitSize >> (i * 8)) & 0xFF;

        }


        // 6.2. 写入 firstValue (8 字节)
        for(int i = 0; i < 8; i++) {
            output[outputIdx + 8 + i] = (firstValueBits >> (i * 8)) & 0xFF;

        }

        // 6.3. 写入 maxDecimalPlaces 和 bitCount (各1字节)
        output[outputIdx + 16] = static_cast<unsigned char>(maxDecimalPlaces);
        output[outputIdx + 17] = static_cast<unsigned char>(maxBeta);
        output[outputIdx + 18] = static_cast<unsigned char>(bitCount);

        // 写入 Delta 差分值序列
        int deltaStartIdx = outputIdx + 19; // Delta 数据起始位置
        int bitOffset = 0; // 当前字节已使用的位数
        unsigned char currentByte = 0; // 当前字节缓冲区

        for(int i = 0; i < numDatas-1; i++) {
            uint64_t deltaValue = deltas[i] & ((1ULL << bitCount) -1);

            // 将 Delta 填充到当前字节的剩余空间
            currentByte |= (deltaValue << bitOffset);
            bitOffset += bitCount;

            // 如果当前字节已满（8 位），写入 output，并开始下一个字节
            while(bitOffset >=8) {
                output[deltaStartIdx++] = currentByte;
                bitOffset -=8;
                // 将剩余的 Delta 位数放入新字节
                currentByte = (deltaValue >> (bitCount - bitOffset)) &0xFF; // 运算优先级修正
            }
        }
            // 如果有未写入的字节（即 bitOffset > 0），写入最后一个字节
        if(bitOffset >0) {
            output[deltaStartIdx++] = currentByte;
        }

        // 填充剩余的输出缓冲区
        // while(deltaStartIdx < MAX_BYTES_PER_BLOCK){
        //     output[deltaStartIdx++] = 0x00;
        // }

}

// 主压缩函数
void GDFCompressor_opt::GDFC_compress_no_pack(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
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


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    GDFC_compress_kernel_no_pack<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // printf("beta %.d ",d_cmpBytes[16]);
    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}

__global__ void GDFC_compress_kernel_br(
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
    uint64_t deltas[DATA_PER_THREAD]={};

    int maxDecimalPlaces = 0;
    int maxBeta =0;
    long firstValue = 0;
    int bitCount = 0;

    int base_block_start_idx, base_block_end_idx;
    int quant_chunk_idx;
    // int block_idx; // 如果不使用，可以移除

    long currQuant=0;
    long lorenQuant=0;
    long prevQuant=0;

    unsigned int thread_ofs = 0;
    // 1. 采样
    for (int i = 0; i < numDatas; i++) {
        double value =input[startIdx + i];
        double log10v = log10(std::abs(value));
        int sp = floor(log10v);
        
        double alpha = getDecimalPlaces_br(value, sp);// 得到小数位数
        double beta =  alpha + sp + 1;
        maxBeta = device_max(maxBeta,beta);
        maxDecimalPlaces = device_max(maxDecimalPlaces, alpha);
    }
    //printf("maxDecimalPlaces:%d\n", maxDecimalPlaces);
    // 2. FOR + zigzag（用4向量化进行实现）
    uint64_t maxDelta = 0;
    firstValue = double2long(input[startIdx], maxDecimalPlaces,maxBeta); // 量化第一个
    prevQuant = firstValue;// 初始化第一个量化值
    base_block_start_idx = startIdx + 1;

    for(int i=0;i<numDatas-1;i++){
        currQuant = double2long(input[base_block_start_idx+i], maxDecimalPlaces,maxBeta); // 量化当前数据点
        lorenQuant = currQuant - prevQuant; // 计算差分
        deltas[i] = zigzag_encode_cuda(lorenQuant);
    
        maxDelta = device_max_uint64(maxDelta, deltas[i]);
        prevQuant = currQuant;
        // if(startIdx<4){
        //     printf("input %d, ",deltas[i]);
        // }
    }
    // for(int j = 0; j < (numDatas+31) / 32; j++) { // 每个线程的数量 / 一个数据批次（32）
    //     base_block_start_idx = startIdx + j * 32;           //每一组32个数据的起始位置
    //     base_block_end_idx = base_block_start_idx + 32;     //每一组32个数据的结束位置

    //     if(base_block_end_idx < totalSize) {
    //             int i = base_block_start_idx;
                
            
    //         #pragma unroll 8 //循环展开8次，就是4*8=32个数据,修改为7次，把第一次提取出来
    //         for(; i < base_block_end_idx; i += 4) {

    //             tmp_buffer = reinterpret_cast<const double4*>(input)[(i) / 4];
    //             quant_chunk_idx = j * 32 + ((i) % 32); //处理的每一组的第几个数据

    //             currQuant = double2long(tmp_buffer.x, maxDecimalPlaces,maxBeta); // 量化当前数据点
    //             lorenQuant = currQuant - prevQuant; // 计算差分

    //             deltas[quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);
    //             //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx],quant_chunk_idx,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant; // 更新前一个量化值
    //             maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]); // 存储差分绝对值
    //             // }

    //             // 处理y分量
    //             currQuant = double2long(tmp_buffer.y, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 1] = zigzag_encode_cuda(lorenQuant);
    //             //printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+1],quant_chunk_idx+1,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[ quant_chunk_idx + 1]);

    //             // 处理z分量
    //             currQuant = double2long(tmp_buffer.z, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 2] = zigzag_encode_cuda(lorenQuant);
    //             // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+2],quant_chunk_idx+2,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 2]);

    //             // 处理w分量
    //             currQuant = double2long(tmp_buffer.w, maxDecimalPlaces,maxBeta);
    //             lorenQuant = currQuant - prevQuant;

    //             deltas[quant_chunk_idx + 3] = zigzag_encode_cuda(lorenQuant);
    //             // printf("zigzag:%02x delta[%d]:%ld currQuant:%ld  prevQuant:%ld \n",deltas[quant_chunk_idx+3],quant_chunk_idx+3,lorenQuant,currQuant,prevQuant);
    //             prevQuant = currQuant;
    //             maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx + 3]);
    //         }
    //     }
    //     else {
    //         // 处理当前数据块超出数据范围的情况
    //         if(base_block_start_idx >= endIdx) {
    //             // 如果整个数据块都超出范围，将absQuant设置为0
    //             quant_chunk_idx = j * 32 + (base_block_start_idx % 32);
    //             for(int i = quant_chunk_idx; i < quant_chunk_idx + 32; i++)
    //                 deltas[i] = 0;
    //         }
    //         else {
    //             // 部分数据块在范围内，部分超出范围
    //             int remainbEle = totalSize - base_block_start_idx;  // 剩余有效数据元素数
    //             int zeronbEle = base_block_end_idx - totalSize;     // 超出范围的数据元素数

    //             // 处理剩余有效数据元素
    //             for(int i = base_block_start_idx; i < base_block_start_idx + remainbEle; i++) {
    //                 if(i==startIdx)
    //                 {
    //                     deltas[0]=0;
    //                     continue;
    //                 }
    //                 quant_chunk_idx = j * 32 + (i % 32);
    //                 currQuant = double2long(input[i], maxDecimalPlaces,maxBeta);

    //                 lorenQuant = currQuant - prevQuant;

    //                 deltas[ quant_chunk_idx] = zigzag_encode_cuda(lorenQuant);

    //                 prevQuant = currQuant;
    //                 maxDelta = device_max_uint64(maxDelta, deltas[quant_chunk_idx]);
    //             }

    //             quant_chunk_idx = j * 32 + (totalSize % 32);
    //             for(int i = quant_chunk_idx; i < quant_chunk_idx + zeronbEle; i++)
    //                 deltas[i] = 0;
    //         }
    //     }
    // }

    bitCount = maxDelta > 0 ? 64 - __clzll(maxDelta) : 1;//用内置函数 替代处理循环
    bitCount = min(bitCount, (int)MAX_BITCOUNT);


        int numByte = (numDatas-1 + 7) / 8;
        // uint8_t result[64][128];
        uint8_t result[64][128] = {}; 
        // 初始化二维数组

        // 遍历每个uint64_t的数据
        for (int i = 0; i < bitCount; ++i) {//行
            int j=0;

            while(j+8<numDatas-1)//有效 0.0027->0.0023
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
            for (; j <numDatas -1; ++j) {//列numBytes
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
        else {
            // warp==0：显式把排他前缀和置 0（由一个线程写，block 内可见）
            if (!lane) { excl_sum = 0; }
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

    // 6 开始写入


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


}

//返回bits
void GDFCompressor_opt::GDFC_compress_br(double* d_oriData, unsigned char* d_cmpBytes, unsigned int* d2h_async_totalBits_ptr, size_t nbEle, cudaStream_t stream)
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


    // cuSZp GPU compression.
    dim3 blockSize(bsize);
    dim3 gridSize(gsize);

    GDFC_compress_kernel_br<<<gridSize, blockSize, sizeof(unsigned int)*2, stream>>>(d_oriData, d_cmpBytes, d_cmpOffset, d_locOffset, d_flag, nbEle);

    // Obtain compression ratio and move data back to CPU.  
    cudaMemcpyAsync(d2h_async_totalBits_ptr, (d_cmpOffset + cmpOffSize-1), sizeof(unsigned int), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_cmpOffset,stream);
    cudaFreeAsync(d_locOffset,stream);
    cudaFreeAsync(d_flag,stream);
}
