//
// 完整GPU版本的PostOffice算法 - 严格遵循CPU版本逻辑
// 

#include "post_office_solver32.cuh"
#include "BitStream/BitWriter.cuh"
#include <climits>

// 辅助函数：计算总数和非零位置信息
__device__ void calTotalCountAndNonZerosCounts_GPU32(
    const int *distribution,
    int *out_pre_non_zeros_count,
    int *out_post_non_zeros_count,
    int *out_total_count,
    int *out_non_zeros_count
) {
    int non_zeros_count = 32; // 初始假设所有位置都非零
    int total_count = distribution[0];
    out_pre_non_zeros_count[0] = 1; // 第一个位置视为非零
    
    for (int i = 1; i < 32; ++i) {
        total_count += distribution[i];
        // 使用CPU版本的"magic code"技术避免分支
        int magic_code = (distribution[i] == 0) ? 1 : 0;
        non_zeros_count -= magic_code;
        out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1] + (1 - magic_code);
    }
    
    // 计算后缀非零计数
    for (int i = 0; i < 32; ++i) {
        out_post_non_zeros_count[i] = non_zeros_count - out_pre_non_zeros_count[i];
    }
    
    *out_total_count = total_count;
    *out_non_zeros_count = non_zeros_count;
}

// 核心动态规划算法 - BuildPostOffice的GPU实现
__device__ int buildPostOffice_GPU32(
    const int *distribution,
    int num,
    int non_zeros_count,
    const int *pre_non_zeros_count,
    const int *post_non_zeros_count,
    int *out_positions
) {
    int original_num = num;
    num = min(num, non_zeros_count);
    
    if (num <= 0) {
        return INT_MAX;
    }
    
    // 动态规划状态数组 - 使用栈上分配避免动态内存
    int dp[32][16]; // dp[i][j] = 前i个位置放j个邮局的最小代价
    int pre[32][16]; // pre[i][j] = dp[i][j]对应的前一个邮局位置
    
    // 初始化DP数组
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 16; j++) {
            dp[i][j] = INT_MAX;
            pre[i][j] = -1;
        }
    }
    
    // 第0个位置是第0个邮局，此时状态为0
    dp[0][0] = 0;
    pre[0][0] = -1;
    
    // 动态规划填表
    for (int i = 1; i < 32; ++i) {
        if (distribution[i] == 0) {
            continue; // 跳过零位置
        }
        
        for (int j = max(1, num + i - 32); j <= i && j < num; ++j) {
            if (j == 1 && i > 1) {
                // 特殊情况：只有一个邮局
                dp[i][j] = 0;
                for (int k = 1; k < i; k++) {
                    dp[i][j] += distribution[k] * k;
                }
                pre[i][j] = 0;
            } else {
                // 一般情况：多个邮局
                if (pre_non_zeros_count[i] < j + 1 || 
                    post_non_zeros_count[i] < num - 1 - j) {
                    continue; // 剪枝：不满足邮局数量约束
                }
                
                int app_cost = INT_MAX;
                int pre_k = 0;
                
                for (int k = j - 1; k <= i - 1; ++k) {
                    // 检查位置k的有效性
                    if ((distribution[k] == 0 && k > 0) || 
                        pre_non_zeros_count[k] < j || 
                        post_non_zeros_count[k] < num - j ||
                        dp[k][j-1] == INT_MAX) {
                        continue;
                    }
                    
                    int sum = dp[k][j - 1];
                    for (int p = k + 1; p <= i - 1; ++p) {
                        sum += distribution[p] * (p - k);
                    }
                    
                    if (app_cost > sum) {
                        app_cost = sum;
                        pre_k = k;
                        if (sum == 0) {
                            break; // 提前终止优化
                        }
                    }
                }
                
                if (app_cost != INT_MAX) {
                    dp[i][j] = app_cost;
                    pre[i][j] = pre_k;
                }
            }
        }
    }
    
    // 寻找最优解
    int temp_total_app_cost = INT_MAX;
    int temp_best_last = INT_MAX;
    
    for (int i = num - 1; i < 32; ++i) {
        if (num - 1 == 0 && i > 0) {
            break;
        }
        if ((distribution[i] == 0 && i > 0) || 
            pre_non_zeros_count[i] < num ||
            dp[i][num-1] == INT_MAX) {
            continue;
        }
        
        int sum = dp[i][num - 1];
        for (int j = i + 1; j < 32; ++j) {
            sum += distribution[j] * (j - i);
        }
        
        if (temp_total_app_cost > sum) {
            temp_total_app_cost = sum;
            temp_best_last = i;
        }
    }
    
    if (temp_best_last == INT_MAX) {
        return INT_MAX; // 无解
    }
    
    // 回溯构建邮局位置
    int temp_positions[16];
    int pos_count = 0;
    int current = temp_best_last;
    int j = num - 1;
    
    while (current != -1 && pos_count < 16) {
        temp_positions[pos_count++] = current;
        if (j == 0) break;
        current = pre[current][j];
        j--;
    }
    
    // 反转位置数组（回溯是逆序的）
    for (int i = 0; i < pos_count; i++) {
        out_positions[i] = temp_positions[pos_count - 1 - i];
    }
    
    // 处理邮局数量扩展
    if (original_num > non_zeros_count) {
        int modifying_positions[16];
        int mod_count = 0;
        int orig_idx = 0, pos_idx = 0;
        
        while (mod_count < original_num && mod_count < 16) {
            if (orig_idx - pos_idx < original_num - pos_count && 
                pos_idx < pos_count && 
                orig_idx < out_positions[pos_idx]) {
                modifying_positions[mod_count++] = orig_idx;
                orig_idx++;
            } else if (pos_idx < pos_count) {
                modifying_positions[mod_count++] = out_positions[pos_idx];
                orig_idx++;
                pos_idx++;
            } else {
                modifying_positions[mod_count++] = orig_idx;
                orig_idx++;
            }
        }
        
        // 复制扩展后的结果
        for (int i = 0; i < mod_count; i++) {
            out_positions[i] = modifying_positions[i];
        }
        pos_count = mod_count;
    }
    
    return pos_count;
}

// 主要的初始化函数 - 完整实现CPU版本逻辑
__device__ int initRoundAndRepresentation32(
    const int *distribution,
    int *representation,
    int *round,
    int *out_positions
) {
    // 1. 计算总数和非零位置信息
    int pre_non_zeros_count[32];
    int post_non_zeros_count[32];
    int total_count, non_zeros_count;
    
    calTotalCountAndNonZerosCounts_GPU32(
        distribution, pre_non_zeros_count, post_non_zeros_count,
        &total_count, &non_zeros_count
    );
    
    // 2. 遍历不同的z值，寻找最优解
    int max_z = min(kPositionLength2Bits[non_zeros_count], 4); // 最多4个bit
    int total_cost = INT_MAX;
    int best_positions[16];
    int best_positions_count = 0;
    
    for (int z = 0; z <= max_z; ++z) {
        int present_cost = total_count * z;
        if (present_cost >= total_cost) break; // 提前终止
        
        int num = kPow2z[z]; // 邮局数量 = 2^z
        int temp_positions[16];
        
        int pos_count = buildPostOffice_GPU32(
            distribution, num, non_zeros_count,
            pre_non_zeros_count, post_non_zeros_count,
            temp_positions
        );
        
        if (pos_count > 0 && pos_count <= 16) {
            // 计算总代价（需要重新计算应用代价）
            int app_cost = 0;
            
            // 简化的代价计算
            for (int i = 0; i < 32; i++) {
                if (distribution[i] > 0) {
                    int min_dist = INT_MAX;
                    for (int j = 0; j < pos_count; j++) {
                        int dist = abs(i - temp_positions[j]);
                        if (dist < min_dist) {
                            min_dist = dist;
                        }
                    }
                    app_cost += distribution[i] * min_dist;
                }
            }
            
            int temp_total_cost = app_cost + present_cost;
            if (temp_total_cost < total_cost) {
                total_cost = temp_total_cost;
                best_positions_count = pos_count;
                for (int i = 0; i < pos_count; i++) {
                    best_positions[i] = temp_positions[i];
                }
            }
        }
    }
    
    // 3. 构建representation和round数组
    for (int i = 0; i < 32; i++) {
        representation[i] = 0;
        round[i] = 0;
    }
    
    if (best_positions_count > 0) {
        representation[0] = 0;
        round[0] = 0;
        int pos_idx = 1;
        
        for (int j = 1; j < 32; ++j) {
            // 使用CPU版本的"magic code"技术
            int magic_code = (pos_idx < best_positions_count && j == best_positions[pos_idx]) ? 1 : 0;
            representation[j] = representation[j - 1] + magic_code;
            round[j] = magic_code ? j : round[j - 1];
            pos_idx += magic_code;
        }
        
        // 输出最终位置
        for (int i = 0; i < best_positions_count; i++) {
            out_positions[i] = best_positions[i];
        }
    } else {
        // 默认情况
        out_positions[0] = 0;
        best_positions_count = 1;
    }
    
    return best_positions_count;
}

__device__ int write_positions_device32(
    BitWriter *writer,
    const int *positions,
    int positions_len
) {
    // 限制范围
    if (positions_len < 0) positions_len = 0;
    if (positions_len > 16) positions_len = 16;
    
    // 写入位置数量（4位）
    write(writer, positions_len, 4);
    int total_bits = 4;
    
    // 写入每个位置（5位each）
    for (int i = 0; i < positions_len; i++) {
        int pos = positions[i];
        if (pos < 0) pos = 0;
        if (pos > 31) pos = 31;
        
        write(writer, pos, 5);
        total_bits += 5;
    }
    
    return total_bits;
}

// 验证函数
__device__ bool validate_distribution32(const int *distribution) {
    int total = 0;
    int non_zero_count = 0;
    
    for (int i = 0; i < 32; i++) {
        if (distribution[i] < 0) return false;
        if (distribution[i] > 1000000) return false;
        
        total += distribution[i];
        if (distribution[i] > 0) {
            non_zero_count++;
        }
    }
    
    return (total <= 10000000 && non_zero_count <= 32);
}

// 调试函数
__device__ void debug_print_post_office_result32(
    const int *positions, int count, int thread_id
) {
    if (thread_id == 0 && count > 0) {
        printf("PostOffice结果: %d个邮局, 位置: ", count);
        for (int i = 0; i < min(count, 10); i++) {
            printf("%d ", positions[i]);
        }
        printf("\n");
    }
}