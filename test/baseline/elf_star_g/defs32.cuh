
#ifndef _SSIZE_T_DEFINED
    #define _SSIZE_T_DEFINED
    #ifdef _WIN32
        #include <basetsd.h>
        typedef SSIZE_T ssize_t;
    #else
        #include <sys/types.h>
    #endif
#endif

#ifndef DEFS32_CUH
#define DEFS32_CUH
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

#define FLOAT_MANTISSA_BITS 23
#define FLOAT_EXPONENT_BITS 8
#define FLOAT_EXPONENT_BIAS 127

static __device__ __constant__ int f_32[] = {
    0, 4, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40, 44, 47, 50, 54, 57, 60, 64, 67
};

static __device__ __constant__ float map10iP_32[] = {
    1.0f, 1.0E1f, 1.0E2f, 1.0E3f, 1.0E4f, 1.0E5f, 1.0E6f, 1.0E7f, 1.0E8f, 1.0E9f, 
    1.0E10f, 1.0E11f, 1.0E12f, 1.0E13f, 1.0E14f, 1.0E15f, 1.0E16f, 1.0E17f, 1.0E18f, 1.0E19f, 1.0E20f
};

static __device__ __constant__ int mapSPGreater1_32[] = {
    1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000
};

static __device__ __constant__ float mapSPLess1_32[] = {
    1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f, 
    0.00000001f, 0.000000001f, 0.0000000001f
};

static __device__ __constant__ float map10iN_32[] = {
    1.0f, 1.0E-1f, 1.0E-2f, 1.0E-3f, 1.0E-4f, 1.0E-5f, 1.0E-6f, 1.0E-7f, 1.0E-8f, 1.0E-9f,
    1.0E-10f, 1.0E-11f, 1.0E-12f, 1.0E-13f, 1.0E-14f, 1.0E-15f, 1.0E-16f, 1.0E-17f, 1.0E-18f,
    1.0E-19f, 1.0E-20f
};

static __device__ void getSPAnd10iNFlag_32(float v, int result_sp_flag[2]) {
    result_sp_flag[1] = 0;
    if (v >= 1.0f) {
        int i = 0;
        while (i < MAP_SP_GREATER_1_SIZE - 1) {
            if (v < mapSPGreater1_32[i + 1]) {
                result_sp_flag[0] = i;
                return;
            }
            i++;
        }
        result_sp_flag[0] = MAP_SP_GREATER_1_SIZE - 1;
    } else {
        int i = 1;
        while (i < MAP_SP_LESS_1_SIZE) {
            if (v >= mapSPLess1_32[i]) {
                result_sp_flag[0] = -i;
                result_sp_flag[1] = (v == mapSPLess1_32[i]) ? 1 : 0;
                return;
            }
            i++;
        }
        result_sp_flag[0] = -(MAP_SP_LESS_1_SIZE - 1);
    }
    
    float log10v = log10f(v);
    result_sp_flag[0] = (int) floorf(log10v);
    result_sp_flag[1] = (fabsf(log10v - floorf(log10v)) < 1e-6f) ? 1 : 0;
}

static __device__ float get10iP_32(int i) {
    if (i <= 0) return 1.0f;  // 🔥 Java throws exception, but we return 1.0f
    if (i >= MAP_10_I_P_SIZE) {
        return powf(10.0f, (float)i);
    } else {
        return map10iP_32[i];
    }
}

// 🔥 完全按照Java逻辑重写
static __device__ int getSignificantCount_32(float v, int sp, int lastBetaStar) {
    int i;
    
    // 🔥 完全按照Java的条件分支
    if (lastBetaStar != 0x7FFFFFFF && lastBetaStar != 0) {
        // Java: i = Math.max(lastBetaStar - sp - 1, 1);
        i = lastBetaStar - sp - 1;
        if (i < 1) i = 1;
    } else if (sp >= 0) {
        i = 1;
    } else {
        i = -sp;
    }

    float temp = v * get10iP_32(i);
    long tempLong = (long) temp;
    
    // Java: while ((int) temp != temp)
    while (tempLong != temp) {
        i++;
        temp = v * get10iP_32(i);
        tempLong = (long) temp;
        
        // 安全检查防止无限循环
        if (i > 20) break;
    }
    
    // Java: if (temp / get10iP(i) != v) return 8;
    if (temp / get10iP_32(i) != v) {
        return 8;
    } else {
        // 🔥 Java的关键逻辑：去除尾部的0
        while (i > 0 && tempLong % 10 == 0) {
            i--;
            tempLong = tempLong / 10;
        }
        return sp + i + 1;
    }
}

static __device__ void getAlphaAndBetaStar_32(float v, int lastBetaStar, int alphaAndBetaStar[2]) {
    v = fabsf(v);
    int spAnd10iNFlag[2];
    getSPAnd10iNFlag_32(v, spAnd10iNFlag);
    int beta = getSignificantCount_32(v, spAnd10iNFlag[0], lastBetaStar);
    alphaAndBetaStar[0] = beta - spAnd10iNFlag[0] - 1;
    alphaAndBetaStar[1] = (spAnd10iNFlag[1] == 1) ? 0 : beta;
}

static __device__ int getFAlpha_32(int alpha) {
    if (alpha < 0) alpha = 0;
    if (alpha >= F_TABLE_SIZE) {
        return (int) ceilf(alpha * LOG_2_10);
    } else {
        return f_32[alpha];
    }
}

static __device__ int getSP_32(float v) {
    if (v >= 1.0f) {
        int i = 0;
        while (i < MAP_SP_GREATER_1_SIZE - 1) {
            if (v < mapSPGreater1_32[i + 1]) {
                return i;
            }
            i++;
        }
        return MAP_SP_GREATER_1_SIZE - 1;
    } else {
        int i = 1;
        while (i < MAP_SP_LESS_1_SIZE) {
            if (v >= mapSPLess1_32[i]) {
                return -i;
            }
            i++;
        }
        return -(MAP_SP_LESS_1_SIZE - 1);
    }
    
    return (int) floorf(log10f(v));
}

static __device__ float get10iN_32(int i) {
    if (i <= 0) return 1.0f;  // 🔥 Java throws exception
    if (i >= MAP_10_I_N_SIZE) {
        return powf(10.0f, -(float)i);
    } else {
        return map10iN_32[i];
    }
}

static __device__ float roundUp_32(float v, int alpha) {
    float scale = get10iP_32(alpha);
    if (v < 0.0f) {
        return floorf(v * scale) / scale;
    } else {
        return ceilf(v * scale) / scale;
    }
}

#endif //DEFS32_CUH