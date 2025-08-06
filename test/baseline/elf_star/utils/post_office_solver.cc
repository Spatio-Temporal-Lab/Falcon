#include <algorithm>

#include "post_office_solver.h"

Array<int>
PostOfficeSolver::InitRoundAndRepresentation(Array<int> &distribution, Array<int> &representation, Array<int> &round) {
  // 当前及前面的非零个数（包括当前）
  Array<int> pre_non_zeros_count(distribution.length());
  // 当前后面的非零个数（不包括当前）
  Array<int> post_non_zeros_count(distribution.length());
  Array<int> total_count_and_non_zeros_counts = CalTotalCountAndNonZerosCounts(distribution, pre_non_zeros_count,
                                                                               post_non_zeros_count);

  int max_z = std::min(kPositionLength2Bits[total_count_and_non_zeros_counts[1]], 5);  // 最多用5个bit来表示
  int total_cost = std::numeric_limits<int>::max();
  Array<int> positions = {};

  int present_cost;
  for (int z = 0; z <= max_z && (present_cost = total_count_and_non_zeros_counts[0] * z) < total_cost; ++z) {
    // 邮局的总数量
    int num = PostOfficeSolver::kPow2z[z];
    PostOfficeResult por = PostOfficeSolver::BuildPostOffice(distribution, num,
                                                             total_count_and_non_zeros_counts[1],
                                                             pre_non_zeros_count, post_non_zeros_count);
    int temp_total_cost = por.total_app_cost() + present_cost;
    if (temp_total_cost < total_cost) {
      total_cost = temp_total_cost;
      positions = por.office_positions();
    }
  }

  representation[0] = 0;
  round[0] = 0;
  int i = 1;
  for (int j = 1; j < distribution.length(); ++j) {
    // Bad but useful code
    int magic_code = (i < positions.length() && j == positions[i]);
    representation[j] = representation[j - 1] + magic_code;
    round[j] = magic_code ? j : round[j - 1];
    i += magic_code;
//    if (i < positions.length() && j == positions[i]) {
//      representation[j] = representation[j - 1] + 1;
//      round[j] = j;
//      ++i;
//    } else {
//      representation[j] = representation[j - 1];
//      round[j] = round[j - 1];
//    }
  }

  return positions;
}

Array<int>
PostOfficeSolver::CalTotalCountAndNonZerosCounts(Array<int> &arr, Array<int> &out_pre_non_zeros_count,
                                                 Array<int> &out_post_non_zeros_count) {
  int non_zeros_count = arr.length();
  int total_count = arr[0];
  out_pre_non_zeros_count[0] = 1;            // 第一个视为非零
  for (int i = 1; i < arr.length(); ++i) {
    total_count += arr[i];
    // Bad but useful code
    int magic_code = (arr[i] == 0);
    non_zeros_count -= magic_code;
    out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1] + !magic_code;
    // Equivalent logic below
    //if (arr[i] == 0) {
    //  --non_zeros_count;
    //  out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1];
    //} else {
    //  out_pre_non_zeros_count[i] = out_pre_non_zeros_count[i - 1] + 1;
    //}
  }
  for (int i = 0; i < arr.length(); ++i) {
    out_post_non_zeros_count[i] = non_zeros_count - out_pre_non_zeros_count[i];
  }
  return Array<int>{total_count, non_zeros_count};
}

PostOfficeResult PostOfficeSolver::BuildPostOffice(Array<int> &arr, int num, int non_zeros_count,
                                                   Array<int> &pre_non_zeros_count, Array<int> &post_non_zeros_count) {
  int original_num = num;
  num = std::min(num, non_zeros_count);

  /**
   * 状态矩阵。d[i][j]表示，只考虑前i个居民点，且第i个位置是第j个邮局的总距离，i >= j
   * 下标从0开始。注意，并非是所有居民点的总距离，因为没有考虑第j个邮局之后的居民点的距离
   */
  int dp[arr.length()][num];
  // 对应于dp[i][j]，表示让dp[i][j]最小时，第j-1个邮局所在的位置信息
  int pre[arr.length()][num];

  // 第0个位置是第0个邮局，此时状态为0
  dp[0][0] = 0;
  // 让dp[0][0]最小时，第-1个邮局所在的位置信息为-1
  pre[0][0] = -1;

  for (int i = 1; i < arr.length(); ++i) {
    if (arr[i] == 0) {
      continue;
    }
    for (int j = std::max(1, num + i - arr.length()); j <= i && j < num; ++j) {
      // arr.length - i < num - j，
      // 表示i后面的居民数（arr.length - i）不足以构建剩下的num - j个邮局
      if (i > 1 && j == 1) {
        dp[i][j] = 0;
        for (int k = 1; k < i; k++) {
          dp[i][j] += arr[k] * k;
        }
        pre[i][j] = 0;
      } else {
        if (pre_non_zeros_count[i] < j + 1 || post_non_zeros_count[i] < num - 1 - j) {
          continue;
        }
        int app_cost = std::numeric_limits<int>::max();
        int pre_k = 0;
        for (int k = j - 1; k <= i - 1; ++k) {
          if (arr[k] == 0 && k > 0 || pre_non_zeros_count[k] < j || post_non_zeros_count[k] < num - j) {
            continue;
          }
          int sum = dp[k][j - 1];
          for (int p = k + 1; p <= i - 1; ++p) {
            sum += arr[p] * (p - k);
          }
          if (app_cost > sum) {
            app_cost = sum;
            pre_k = k;
            if (sum == 0) {
              // 找到其中一个0，提前终止
              break;
            }
          }
        }
        if (app_cost != std::numeric_limits<int>::max()) {
          dp[i][j] = app_cost;
          pre[i][j] = pre_k;
        }
      }
    }
  }
  int temp_total_app_cost = std::numeric_limits<int>::max();
  int temp_best_last = std::numeric_limits<int>::max();
  for (int i = num - 1; i < arr.length(); ++i) {
    if (num - 1 == 0 && i > 0) {
      break;
    }
    if (arr[i] == 0 && i > 0 || pre_non_zeros_count[i] < num) {
      continue;
    }
    int sum = dp[i][num - 1];
    for (int j = i + 1; j < arr.length(); ++j) {
      sum += arr[j] * (j - i);
    }
    if (temp_total_app_cost > sum) {
      temp_total_app_cost = sum;
      temp_best_last = i;
    }
  }

  Array<int> office_positions(num);
  int i = 1;

  while (temp_best_last != -1) {
    office_positions[num - i] = temp_best_last;
    temp_best_last = pre[temp_best_last][num - i];
    ++i;
  }

  if (original_num > non_zeros_count) {
    Array<int> modifying_office_positions(original_num);
    int j = 0, k = 0;
    while (j < original_num && k < num) {
      if (j - k < original_num - num && j < office_positions[k]) {
        modifying_office_positions[j] = j;
        ++j;
      } else {
        modifying_office_positions[j] = office_positions[k];
        ++j;
        ++k;
      }
    }
    office_positions = modifying_office_positions;
  }

  return {office_positions, temp_total_app_cost};
}

int PostOfficeSolver::WritePositions(Array<int> &positions, BitWriter *writer) {
  write(writer, positions.length(), 5);
  int this_size = 5;
  for (const auto &position : positions) {
    write(writer, position, 6);
    this_size += 6;
  }
  return this_size;
}
