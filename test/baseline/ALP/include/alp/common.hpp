#ifndef ALP_COMMON_HPP
#define ALP_COMMON_HPP

#include <cstdint>
#include <vector>
#include <utility>

namespace alp {
//! bitwidth type
using bw_t = uint8_t;
//! exception counter type
using exp_c_t = uint16_t;
//! exception position type
using exp_p_t = uint16_t;
//! factor idx  type
using factor_idx_t = uint8_t;
//! exponent idx type
using exponent_idx_t = uint8_t;

// ALP记录系统 - 全局变量和函数
inline std::vector<std::pair<uint8_t, uint8_t>>& get_vector_ef_records() {
    static std::vector<std::pair<uint8_t, uint8_t>> records;
    return records;
}

inline std::vector<std::vector<std::pair<std::pair<int, int>, int>>>& get_rowgroup_k_combinations() {
    static std::vector<std::vector<std::pair<std::pair<int, int>, int>>> records;
    return records;
}

inline bool& get_recording_enabled() {
    static bool enabled = false;
    return enabled;
}

// 记录函数
inline void record_vector_ef(uint8_t e, uint8_t f) {
    if (get_recording_enabled()) {
        get_vector_ef_records().emplace_back(e, f);
    }
}

inline void record_rowgroup_combinations(const std::vector<std::pair<std::pair<int, int>, int>>& combinations) {
    if (get_recording_enabled()) {
        get_rowgroup_k_combinations().push_back(combinations);
    }
}

// 控制函数
inline void enable_alp_recording(bool enable = true) {
    get_recording_enabled() = enable;
}

inline void clear_alp_records() {
    get_vector_ef_records().clear();
    get_rowgroup_k_combinations().clear();
}

} // namespace alp

#endif // ALP_COMMON_HPP