//
// Created by lizhzz on 25-7-8.
// Fixed version - 修复了工具函数的实现错误
//
// 添加跨平台 ssize_t 定义
#ifndef _SSIZE_T_DEFINED
    #define _SSIZE_T_DEFINED

    #ifdef _WIN32
        #include <basetsd.h> // 包含 SSIZE_T 定义
        typedef SSIZE_T ssize_t;
    #else
        #include <sys/types.h> // Linux/macOS 环境
    #endif
#endif

#ifndef DEFS_CUH
#define DEFS_CUH
#include <cstdint>

union DOUBLE {
    double d;
    uint64_t i;
};

union FLOAT {
    float f;
    uint32_t i;
};

#define LOG_2_10 3.32192809489

#define F_TABLE_SIZE 21
#define MAP_SP_GREATER_1_SIZE 10
#define MAP_SP_LESS_1_SIZE 11
#define MAP_10_I_P_SIZE 21
#define MAP_10_I_N_SIZE 21

static __device__ __constant__ int f[] = {0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50, 54, 57, 60, 64, 67};

static __device__ __constant__ double map10iP[] = {1.0, 1.0E1, 1.0E2, 1.0E3, 1.0E4, 1.0E5, 1.0E6, 1.0E7, 1.0E8, 1.0E9, 1.0E10, 1.0E11,
                                 1.0E12, 1.0E13, 1.0E14, 1.0E15, 1.0E16, 1.0E17, 1.0E18, 1.0E19, 1.0E20};

static __device__ __constant__ long mapSPGreater1[] = {
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
};

static __device__ __constant__ double mapSPLess1[] = {
    1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001
};

static __device__ __constant__ double map10iN[] = {1.0, 1.0E-1, 1.0E-2, 1.0E-3, 1.0E-4, 1.0E-5, 1.0E-6, 1.0E-7, 1.0E-8, 1.0E-9,
                                 1.0E-10, 1.0E-11, 1.0E-12, 1.0E-13, 1.0E-14, 1.0E-15, 1.0E-16, 1.0E-17, 1.0E-18,
                                 1.0E-19, 1.0E-20};

static __device__ void getSPAnd10iNFlag(double v, int result_sp_flag[2]) {
    result_sp_flag[1] = 0;
    if (v >= 1.0) {
        int i = 0;
        while (i < MAP_SP_GREATER_1_SIZE - 1) {
            if (v < mapSPGreater1[i + 1]) {
                result_sp_flag[0] = i;
                return;
            }
            i++;
        }
        // 修复：处理超出范围的情况
        result_sp_flag[0] = MAP_SP_GREATER_1_SIZE - 1;
    } else {
        int i = 1;
        while (i < MAP_SP_LESS_1_SIZE) {
            if (v >= mapSPLess1[i]) {
                result_sp_flag[0] = -i;
                result_sp_flag[1] = (v == mapSPLess1[i]) ? 1 : 0;
                return;
            }
            i++;
        }
        // 修复：处理超出范围的情况
        result_sp_flag[0] = -(MAP_SP_LESS_1_SIZE - 1);
    }
    
    // 修复：如果没有在表中找到，使用log10计算
    double log10v = log10(v);
    result_sp_flag[0] = (int) floor(log10v);
    result_sp_flag[1] = (fabs(log10v - floor(log10v)) < 1e-10) ? 1 : 0;
}

static __device__ double get10iP(int i) {
    if (i < 0) return 1.0; // 修复：处理负数情况
    if (i >= MAP_10_I_P_SIZE) {
        return pow(10.0, i);
    } else {
        return map10iP[i];
    }
}

static __device__ int getSignificantCount(double v, int sp, int lastBetaStar) {
    int i;
    if (lastBetaStar != 0x7FFFFFFF && lastBetaStar != 0) {
        i = lastBetaStar - sp - 1;
        if (i <= 1) {
            i = 1;
        }
    } else if (lastBetaStar == 0x7FFFFFFF) {
        i = 17 - sp - 1;
    } else if (sp >= 0) {
        i = 1;
    } else {
        i = -sp;
    }

    // 修复：添加边界检查
    if (i < 0) i = 1;
    if (i > 20) i = 20; // 限制在合理范围内

    double temp = v * get10iP(i);
    long tempLong = (long) temp;
    
    // 修复：添加迭代次数限制，防止无限循环
    int max_iterations = 10;
    int iterations = 0;
    
    while (tempLong != temp && iterations < max_iterations) {
        i++;
        if (i > 20) break; // 防止超出合理范围
        temp = v * get10iP(i);
        tempLong = (long) temp;
        iterations++;
    }
    
    // 修复：更精确的浮点数比较
    if (fabs(temp / get10iP(i) - v) > 1e-15) {
        return 17;
    } else {
        while (i > 0 && tempLong % 10 == 0) {
            i--;
            tempLong = tempLong / 10;
        }
        return sp + i + 1;
    }
}

static __device__ void getAlphaAndBetaStar(double v, int lastBetaStar, int alphaAndBetaStar[2]) {
    // 使用GPU内置函数fabs()来求绝对值，效率更高
    v = fabs(v);
    int spAnd10iNFlag[2];
    getSPAnd10iNFlag(v, spAnd10iNFlag);
    int beta = getSignificantCount(v, spAnd10iNFlag[0], lastBetaStar);
    alphaAndBetaStar[0] = beta - spAnd10iNFlag[0] - 1;
    alphaAndBetaStar[1] = (spAnd10iNFlag[1] == 1) ? 0 : beta;
}

static __device__ int getFAlpha(int alpha) {
    if (alpha < 0) alpha = 0;
    if (alpha >= F_TABLE_SIZE) {
        return (int) ceil(alpha * LOG_2_10);
    } else {
        return f[alpha];
    }
}

static __device__ int getSP(double v) {
    if (v >= 1) {
        int i = 0;
        while (i < MAP_SP_GREATER_1_SIZE - 1) {
            if (v < mapSPGreater1[i + 1]) {
                return i;
            }
            i++;
        }
        return MAP_SP_GREATER_1_SIZE - 1; // 修复：返回最大索引
    } else {
        int i = 1;
        while (i < MAP_SP_LESS_1_SIZE) {
            if (v >= mapSPLess1[i]) {
                return -i;
            }
            i++;
        }
        return -(MAP_SP_LESS_1_SIZE - 1); // 修复：返回最小索引
    }
    
    // 备用计算
    return (int) floor(log10(v));
}

static __device__ double get10iN(int i) {
    if (i < 0) return 1.0; // 修复：处理负数情况
    if (i >= MAP_10_I_N_SIZE) {
        return pow(10.0, -i);
    } else {
        return map10iN[i];
    }
}

static __device__ double roundUp(double v, int alpha) {
    double scale = get10iP(alpha);
    if (v < 0) {
        return floor(v * scale) / scale;
    } else {
        return ceil(v * scale) / scale;
    }
}

#endif //DEFS_CUH