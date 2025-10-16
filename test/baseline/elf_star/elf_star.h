#pragma once

#include <cstdint>
#include <cstddef>

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


#ifdef __cplusplus
extern "C" {
#endif
// #include <sys/types.h>

ssize_t elf_star_encode(double *in, ssize_t len, uint8_t **out);
ssize_t elf_star_decode(uint8_t *in, ssize_t len, double *out);

ssize_t elf_star_encode_32(float *in, ssize_t len, uint8_t **out);
ssize_t elf_star_decode_32(uint8_t *in, ssize_t len, float *out);

#ifdef __cplusplus
}
#endif 
