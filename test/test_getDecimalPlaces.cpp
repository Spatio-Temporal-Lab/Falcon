#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <cstdint>
#include <vector>
#include <string>
#include <filesystem>
#include <vector>
#include <string>
#include <filesystem>
#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
namespace fs = std::filesystem;
// 你的精确方法实现（简化版，需要定义相关常量）
#define POW_NUM_G ((1L << 51) + (1L << 52))
static double pow10_table[16] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 
                                100000000, 1000000000, 10000000000, 100000000000, 
                                1000000000000, 10000000000000, 100000000000000, 1000000000000000};
                                
                                // 常量定义
#define MAX_DECIMAL_PLACES 17      // 双精度最大有效小数位
#define POW_NUM_G 6755399441055744.0  // 2^52 + 2^51，用于浮点数对齐

// 预计算的10的幂次表
static const double pow10_table_64[] = {
    1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
    1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20
};


// 函数声明
double getPow10_64(int exp) {
    if (exp < 0) return 1.0 / getPow10_64(-exp);
    if (exp < sizeof(pow10_table_64) / sizeof(pow10_table_64[0])) {
        return pow10_table_64[exp];
    }
    return pow(10.0, exp);
}

// 方法1: 暴力误差计算方法
int getDecimalPlacesBruteForce(double v) {
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


// 方法3: 改进的字符串方法（处理科学计数法）
int getDecimalPlacesStringAdvanced(double value) {
    if (value == 0.0) return 0;
    
    char buffer[32];
    sprintf(buffer, "%.15g", value);
    
    // 检查是否为科学计数法
    char* e_pos = strchr(buffer, 'e');
    if (e_pos == NULL) {
        e_pos = strchr(buffer, 'E');
    }
    
    if (e_pos != NULL) {
        // 科学计数法格式
        int exponent = atoi(e_pos + 1);
        
        // 找到小数点
        char* decimal_point = strchr(buffer, '.');
        if (decimal_point == NULL) {
            return (exponent < 0) ? -exponent : 0;
        }
        
        // 计算尾数的小数位数
        int mantissa_decimals = (int)(e_pos - decimal_point - 1);
        
        // 最终小数位数 = 尾数小数位数 - 指数
        int result = mantissa_decimals - exponent;
        return (result > 0) ? result : 0;
    } else {
        // 普通格式，使用之前的方法
        char* decimal_point = strchr(buffer, '.');
        if (decimal_point == NULL) {
            return 0;
        }
        
        int decimal_places = strlen(decimal_point + 1);
        
        // 去除末尾的0
        char* end = decimal_point + strlen(decimal_point) - 1;
        while (end > decimal_point && *end == '0') {
            decimal_places--;
            end--;
        }
        
        return decimal_places;
    }
}


static int getDecimalPlacesAccurate(double value,int sp0) {
    double log10v = log10(std::abs(value));
    int sp = floor(log10v);
    double trac = value + POW_NUM_G - POW_NUM_G;
    double temp = value;

    int digits = 0;
    double td = 1;
    double deltaBound = abs(value) * pow(2, -52);
    while (abs(temp - trac) >= deltaBound * td && digits < 16 - sp - 1)
    {
        digits++;
        td = pow10_table[digits];
        temp = value * td;
        trac = temp + POW_NUM_G - POW_NUM_G;
    }

    return digits;
}

// 准确率测试结构
typedef struct {
    double value;
    int expected_places;  // 期望的小数位数
    const char* description;
} TestCase;

// 生成测试用例
void generateTestCases(TestCase** test_cases, int* num_cases) {
    static TestCase cases[] = {
        // 基础测试用例
        {0.0, 0, "零"},
        {1.0, 0, "整数1"},
        {10.0, 0, "整数10"},
        {100.0, 0, "整数100"},
        
        // 简单小数
        {0.1, 1, "0.1"},
        {0.5, 1, "0.5"},
        {1.5, 1, "1.5"},
        {3.14, 2, "3.14"},
        {3.141, 3, "3.141"},
        {3.1415, 4, "3.1415"},
        
        // 末尾零的情况
        {1.10, 1, "1.10 (末尾零)"},
        {1.100, 1, "1.100 (多个末尾零)"},
        {10.20, 1, "10.20"},
        
        // 很小的数
        {0.001, 3, "0.001"},
        {0.0001, 4, "0.0001"},
        {0.00001, 5, "0.00001"},
        {1e-6, 6, "1e-6"},
        {1e-10, 10, "1e-10"},
        
        // 精度边界测试
        {0.123456789012345, 15, "15位小数"},
        {1.23456789012345e-5, 20, "科学计数法小数"},
        
        // 浮点数精度问题
        {0.1 + 0.2, 1, "0.1+0.2 (浮点精度问题)"},
        {1.0/3.0, 16, "1/3 (无限循环小数)"},
        {1.0/7.0, 16, "1/7 (无限循环小数)"},
        
        // 负数
        {-3.14, 2, "-3.14"},
        {-0.001, 3, "-0.001"},
        
        // 科学计数法
        {1.23e5, 0, "1.23e5 (大数)"},
        {1.23e-5, 7, "1.23e-5 (小数)"},
        {1.5e-10, 11, "1.5e-10"},
        
        // 边界情况
        {1.23456789012345e-15, 27, "极小数"},
        {9.87654321098765, 14, "多位小数"}
    };
    
    *test_cases = cases;
    *num_cases = sizeof(cases) / sizeof(cases[0]);
}

// 计算准确率
void calculateAccuracy() {
    TestCase* test_cases;
    int num_cases;
    generateTestCases(&test_cases, &num_cases);
    
    int accurate_correct = 0, brute_correct = 0, advanced_correct = 0;
    int accurate_total = 0, brute_total = 0, advanced_total = 0;
    
    printf("\n=== 准确率测试报告 ===\n");
    printf("测试用例描述                   | 期望 | 精确 | 暴力 高级字符串 | 测试结果\n");
    printf("-------------------------+------+------+------+----------+----------\n");
    
    for (int i = 0; i < num_cases; i++) {
        TestCase tc = test_cases[i];
        
        int result_accurate = getDecimalPlacesAccurate(tc.value, 0);
        int result_brute = getDecimalPlacesBruteForce(tc.value);
        int result_advanced = getDecimalPlacesStringAdvanced(tc.value);
        
        // 统计正确性（允许合理的误差范围）
        int tolerance = 0; // 允许1位误差
        
        bool accurate_ok = abs(result_accurate - tc.expected_places) <= tolerance;
        bool brute_ok = abs(result_brute - tc.expected_places) <= tolerance;
        bool advanced_ok = abs(result_advanced - tc.expected_places) <= tolerance;
        
        if (accurate_ok) accurate_correct++;
        if (brute_ok) brute_correct++;
        if (advanced_ok) advanced_correct++;
        
        accurate_total++;
        brute_total++;
        advanced_total++;
        
        // 显示结果
        printf("%-25s | %4d | %4d | %4d | %8d | ", 
               tc.description, tc.expected_places, 
               result_accurate, result_brute, result_advanced);
        
        if (accurate_ok && brute_ok && advanced_ok) {
            printf("全部正确\n");
        } else {
            printf("存在差异: ");
            if (!accurate_ok) printf("精确 ");
            if (!brute_ok) printf("暴力 ");
            if (!advanced_ok) printf("高级字符串 ");
            printf("\n");
        }
    }
    
    printf("\n=== 准确率统计 ===\n");
    printf("精确方法:     %d/%d = %.2f%%\n", accurate_correct, accurate_total, 
           (double)accurate_correct / accurate_total * 100);
    printf("暴力方法:     %d/%d = %.2f%%\n", brute_correct, brute_total, 
           (double)brute_correct / brute_total * 100);
    printf("高级字符串:   %d/%d = %.2f%%\n", advanced_correct, advanced_total, 
           (double)advanced_correct / advanced_total * 100);
}

// 性能测试
void performanceTest() {
    printf("\n=== 性能测试 ===\n");
    
    // 生成大量测试数据
    const int test_count = 100000;
    double* test_values = (double*)malloc(test_count * sizeof(double));
    
    srand(42); // 固定随机种子以确保可重复性
    for (int i = 0; i < test_count; i++) {
        // 生成各种范围的随机数
        double base = (double)rand() / RAND_MAX;
        int exp = (rand() % 20) - 10; // -10 到 9 的指数
        test_values[i] = base * pow(10, exp);
    }
    
    clock_t start, end;
    
    // 测试精确方法
    start = clock();
    for (int i = 0; i < test_count; i++) {
        getDecimalPlacesAccurate(test_values[i], 0);
    }
    end = clock();
    double time_accurate = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // 测试暴力方法
    start = clock();
    for (int i = 0; i < test_count; i++) {
        getDecimalPlacesBruteForce(test_values[i]);
    }
    end = clock();
    double time_brute = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // 测试高级字符串方法
    start = clock();
    for (int i = 0; i < test_count; i++) {
        getDecimalPlacesStringAdvanced(test_values[i]);
    }
    end = clock();
    double time_advanced = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("测试数量: %d\n", test_count);
    printf("精确方法:     %.4f 秒 (%.2f us/次)\n", time_accurate, time_accurate * 1000000 / test_count);
    printf("暴力方法:     %.4f 秒 (%.2f us/次)\n", time_brute, time_brute * 1000000 / test_count);
    printf("高级字符串:   %.4f 秒 (%.2f us/次)\n", time_advanced, time_advanced * 1000000 / test_count);
    
    free(test_values);
}

// 综合测试函数

void comprehensiveTest() {
    printf("=== 小数位数计算方法综合测试 ===\n");
    calculateAccuracy();
    performanceTest();

}


std::vector<double> read_data(const std::string& file_path, bool show_progress=0, char delimiter=',') {
    std::vector<double> result;
    
    // 第一次读取：获取元数据
    size_t total_lines = 0;
    size_t num_columns = 0;
    {
        std::ifstream meta_file(file_path);
        if (!meta_file) {
            std::cerr << "无法打开文件: " << file_path << std::endl;
            return {};
        }

        // 获取列数和总行数
        std::string first_line;
        if (std::getline(meta_file, first_line)) {
            std::stringstream ss(first_line);
            std::string cell;
            while (std::getline(ss, cell, delimiter)) ++num_columns;
            ++total_lines;
            while (std::getline(meta_file, first_line)) ++total_lines;
        }
        
        meta_file.seekg(0);
    }

    // 初始化列存储
    std::vector<std::vector<double>> columns(num_columns);
    for (auto& col : columns) {
        col.reserve(total_lines + 1000);  // 预分配+缓冲
    }

    // 第二次读取：处理数据
    std::ifstream file(file_path);
    size_t current_line = 0;
    
    // 进度条初始化（与原始实现相同）
    if (show_progress) {
        std::cout << "读取进度:\n0%   10%  20%  30%  40%  50%  60%  70%  80%  90%  100%\n|----|----|----|----|----|----|----|----|----|----|\n";
    }

    std::string line;
    while (std::getline(file, line)) {
        ++current_line;
        
        // 更新进度条（保持原始逻辑）
        if (show_progress && total_lines > 0 && (current_line % (total_lines/20) == 0 || current_line == total_lines)) {
            float progress = static_cast<float>(current_line) / total_lines;
            int bar_width = static_cast<int>(progress * 50);
            
            std::cout << "\r[";
            for(int i = 0; i < 50; ++i) {
                if(i < bar_width) std::cout << "=";
                else if(i == bar_width) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " 
                    << std::fixed << std::setw(5) << std::setprecision(1) 
                    << progress * 100 << "%";
            std::cout.flush();
        }

        // 高效数据解析
        std::stringstream ss(line);
        std::string cell;
        size_t col_idx = 0;
        
        while (std::getline(ss, cell, delimiter) && col_idx < num_columns) {
            char* endptr = nullptr;
            double val = std::strtod(cell.c_str(), &endptr);
            
            // 高效数值校验
            if (endptr != cell.c_str() && !std::isnan(val)) {
                columns[col_idx].emplace_back(val);
            } else if (!cell.empty()) {
                // std::cerr << "\n无效数值: " << cell << "，跳过（行：" << current_line << "）" << std::endl;
            }
            ++col_idx;
        }
    }

    // 合并列数据（内存预分配）
    const size_t total_elements = num_columns * total_lines;
    result.reserve(total_elements + 1000);
    
    for (size_t row = 0; row < total_lines; ++row) {
        for (size_t col = 0; col < num_columns; ++col) {
            if (row < columns[col].size()) {
                result.emplace_back(columns[col][row]);
            }
        }
    }

    if (show_progress) {
        std::cout << "\n\n实际读取数据量: " << result.size() << "/" << total_elements 
                  << " (" << std::round(result.size()*100.0/total_elements) << "%)\n";
    }

    return result;
}

class ANS
{
public:
    ANS(float my,float br,double time_br,double time_ac,double time_ad,int cou):my_w(my),br_w(br),time_brute(time_br),time_accurate(time_ac),time_advanced(time_advanced),count(cou)
    {

    }

    float my_w;
    float br_w;
    double time_brute;
    double time_accurate;
    double time_advanced;
    int count;
};
ANS a(std::vector<double> &values)
{
    float wrong=0;
    int my=0;
    float count=values.size();
    // 创建一个新向量存储需要保留的值
    std::vector<double> new_values;
    for (int i = 0; i < count; i++) {
        double value = values[i];
        
        int result_advanced = getDecimalPlacesStringAdvanced(value);
        int result_accurate = getDecimalPlacesAccurate(value, 0);
        int result_brute = getDecimalPlacesBruteForce(value);
        if(result_advanced>16||result_accurate>16)
        {
            continue;
        }
        else{
            new_values.push_back(values[i]);
        }
        if(result_accurate==result_brute&&result_brute==result_advanced)//全部相等
        {
            continue;
        }
        else if(result_accurate==result_advanced)//暴力不相等
        {
            wrong++;
            continue;
        }
        else if(result_accurate==result_brute)//我们的不相等
        {
            my++;
        }
        else
        {
            continue;
        }
        printf("%.15g,%d,%d,%d\n", 
                value, result_accurate, result_brute,  result_advanced);
    }
    printf("误差暴力准确率 : %0.4f \n",(count-wrong)/count);
    printf("精确方法准确率 : %0.4f \n",(count-my)/count);
    values=new_values;
    return ANS(my,wrong,0,0,0,values.size());
}

// 新添加的CSV文件测试函数
ANS testFromCSVFile(std::string filename, const char* output_filename) {
    
    std::vector<double> values = read_data(filename);
    
    ANS C=a(values);

    int count = values.size();
    printf("成功从 %s 读取 %d 个值\n", filename, count);
    clock_t start, end;
    
    // 测试精确方法
    start = clock();
    for (int i = 0; i < count; i++) {
        getDecimalPlacesAccurate(values[i], 0);
    }
    end = clock();
    double time_accurate = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // 测试暴力方法
    start = clock();
    for (int i = 0; i < count; i++) {
        getDecimalPlacesBruteForce(values[i]);
    }
    end = clock();
    double time_brute = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // 测试高级字符串方法
    start = clock();
    for (int i = 0; i < count; i++) {
        getDecimalPlacesStringAdvanced(values[i]);
    }
    end = clock();
    double time_advanced = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("测试数量: %d\n", count);
    printf("精确方法:     %.4f 秒 (%.2f us/次)\n", time_accurate, time_accurate * 1000000 / count);
    printf("暴力方法:     %.4f 秒 (%.2f us/次)\n", time_brute, time_brute * 1000000 / count);
    printf("高级字符串:   %.4f 秒 (%.2f us/次)\n", time_advanced, time_advanced * 1000000 / count);

    return ANS(C.my_w,C.br_w,time_brute,time_accurate,time_advanced,C.count);

}
// 修改main函数以支持文件测试
int main(int argc, char* argv[]) {
    if (argc >= 2) {
        // 命令行参数格式: 程序名 输入文件名 [输出文件名]
        ANS A();
        const char * output_file = (argc >= 3) ? argv[2] : "NULL.csv";
        // testFromCSVFile(argv[1], output_file);
        std::string dir_path = argv[1]; // 数据文件目录
        for (const auto &entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file()) {
                std::string file_path = entry.path().string();
                std::cout << "正在处理文件: " << file_path << "\n";
                testFromCSVFile(file_path, output_file);
                std::cout << "---------------------------------------------\n";
            }
        }
    }
    else
    comprehensiveTest();
    return 0;

}
