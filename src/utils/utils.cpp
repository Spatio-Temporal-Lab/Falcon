//
// Created by lz on 24-9-27.
//

#include "utils.h"

int count_significant_digits(double num) {
  // 处理0的情况
    if (num == 0.0) {
        return 1; // 0只有一个有效位
    }
    int count = 0;
    int started = 0; // 标记是否开始计数

    // 处理负数
    if (num < 0) {
        num = -num;
    }
    
    // 处理小数部分
    while (num!=(int)num) {
        num *=10;
        count++;
    }
    return count;
}