#pragma once

#include <cstdint>
#include <cstring>

// 核心类型定义
typedef uint64_t idx_t;

// DuckDB API 宏定义
#ifndef DUCKDB_API
#ifdef _WIN32
#define DUCKDB_API __declspec(dllimport)
#else
#define DUCKDB_API
#endif
#endif

// Debug断言宏
#ifdef DEBUG
#define D_ASSERT(condition) assert(condition)
#else
#define D_ASSERT(condition)
#endif

namespace alp_bench {

// Load函数模板 - 用于快速内存读取
template <class T>
inline T Load(const uint8_t* ptr) {
	T ret;
	memcpy(&ret, ptr, sizeof(T));
	return ret;
}

} // namespace alp_bench
