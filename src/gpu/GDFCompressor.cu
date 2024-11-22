#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
__device__ long zigzagEncode(long delta)
{
    return (delta << 1) ^ (delta >> 63);
}

__device__ int getDecimalPlaces(double value)
{
    const double POW_NUM = pow(2, 52) + pow(2, 51); // 根据精度需求调整
    double trac = value + POW_NUM - POW_NUM;
    double temp = value;
    int digits = 0;
    int64_t int_temp;
    int64_t trac_temp;
    memcpy(&int_temp, &temp, sizeof(double));
    memcpy(&trac_temp, &trac, sizeof(double));

    while (abs(trac_temp - int_temp) >= 1 && digits < 15)
    {
        temp *= 10;
        memcpy(&int_temp, &temp, sizeof(double));
        trac = temp + POW_NUM - POW_NUM;
        memcpy(&trac_temp, &trac, sizeof(double));
        digits++;
    }
    return digits;
}

__device__ int getByteCount(long value)
{
    // 计算存储给定long值所需的字节数
    return (value == 0) ? 1 : (int)(log2(value) / 8) + 1;
}

__global__ void compressKernel(const double* data, long* deltaData, int* effectiveDigits, int n, unsigned char* byteArray, int* byteArraySize)
{
    extern __shared__ int sharedEffectiveDigits[];
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int totalThreads = gridDim.x * blockDim.x;

    // 计算每个线程应处理的数据量
    int dataPerThread = (n + totalThreads - 1) / totalThreads;
    int startIdx = idx * dataPerThread;
    int endIdx = min(startIdx + dataPerThread, n);

    // 步骤1：计算每个线程的最大有效位数
    int threadMaxDigits = 0;

    for (int i = startIdx; i < endIdx; ++i)
    {
        int digits = getDecimalPlaces(data[i]);
        if (digits > threadMaxDigits)
            threadMaxDigits = digits;
    }


    // 加载有效位数以计算缩放
    int scale = pow(10, *effectiveDigits);
    long currentValue;

    // 计算currentValue和deltaValue，避免使用if
    currentValue = *effectiveDigits <= 14 ? static_cast<long>(data[idx] * scale) : reinterpret_cast<long>(&data[idx]);

    long deltaValue = (idx == 0) ? currentValue : currentValue - *effectiveDigits <= 14 ? static_cast<long>(data[idx - 1] * scale) : reinterpret_cast<long>(&data[idx-1]);


    // 第四步：Zigzag变换
    deltaData[idx] = zigzagEncode(deltaValue);


    __syncthreads();

    // 第五步：查找最大delta值
    long long maxDelta = 0;
    if (idx < n)
    {
        long encodedValue = deltaData[idx];
        atomicMax(&maxDelta, encodedValue);
    }
    __syncthreads();

    // 将最大delta所需字节数存储在第一个线程中
    if (idx == 0)
    {
        int maxByteCount = getByteCount(maxDelta);
        *byteArraySize = maxByteCount * n; // 字节数组的总大小
    }
    __syncthreads();

    // 第六步：将编码后的deltas存储到字节数组中
    if (idx < n)
    {
        long encodedValue = deltaData[idx];
        int byteCount = getByteCount(encodedValue);

        // 将此编码delta的字节存储到字节数组中
        for (int b = 0; b < byteCount; b++)
        {
            byteArray[idx * (*byteArraySize / n) + b] = (encodedValue >> (b * 8)) & 0xFF;
        }
    }

}

void compressDoubleData(const double* h_data, int n)
{
    double* d_data;
    long* d_longData;
    long* d_deltaData;
    int* d_effectiveDigits;
    int h_effectiveDigits = 0;

    cudaMalloc(&d_data, n * sizeof(double));
    cudaMalloc(&d_longData, n * sizeof(long));
    cudaMalloc(&d_deltaData, n * sizeof(long));
    cudaMalloc(&d_effectiveDigits, sizeof(int));
    cudaMemcpy(d_data, h_data, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_effectiveDigits, &h_effectiveDigits, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Step 1: Calculate effective digits
    calculateEffectiveDigits<<<numBlocks, blockSize>>>(d_data, d_effectiveDigits, n);
    cudaMemcpy(&h_effectiveDigits, d_effectiveDigits, sizeof(int), cudaMemcpyDeviceToHost);

    int scale = pow(10, h_effectiveDigits);

    // Step 2: Convert to long
    convertToLong<<<numBlocks, blockSize>>>(d_data, d_longData, n, scale);

    // Step 3: Delta encode
    deltaEncode<<<numBlocks, blockSize>>>(d_longData, d_deltaData, n);

    // Step 4: Zigzag transform
    zigzagTransform<<<numBlocks, blockSize>>>(d_deltaData, n);

    // Copy result back to host if needed
    long* h_deltaData = new long[n];
    cudaMemcpy(h_deltaData, d_deltaData, n * sizeof(long), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_longData);
    cudaFree(d_deltaData);
    cudaFree(d_effectiveDigits);

    delete[] h_deltaData;
}

int main()
{
    const int n = 1024;
    double h_data[n];
    // Initialize h_data with your double values

    compressDoubleData(h_data, n);

    return 0;
}
