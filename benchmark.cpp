// easy coding benchmark
#include <cstdio>
#include <iostream>
#include <ostream>

#include "Serf/test/baselines/gorilla/gorilla_compressor.h"
#include "Serf/test/baselines/gorilla/gorilla_decompressor.h"

#include <bitset>
#include <cstring>  // 用于 memcpy
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>
#include <sstream>

void printBinary(double number)
{
    // 将 double 类型的 number 转换为 64 位的二进制表示
    uint64_t binaryRepresentation;
    memcpy(&binaryRepresentation, &number, sizeof(double));

    // 手动逐位输出
    for (int i = 63; i >= 0; i--)
    {
        std::cout << ((binaryRepresentation >> i) & 1);
        // 每隔 4 位输出一个空格，方便阅读
        if (i % 4 == 0)
        {
            std::cout << ' ';
        }
    }
    std::cout << std::endl;
}

void printBinary(int64_t number)
{
    for (int i = 63; i >= 0; i--)
    {
        std::cout << ((number >> i) & 1);
        // 每隔 4 位输出一个空格，方便阅读
        if (i % 4 == 0)
        {
            std::cout << ' ';
        }
    }
    std::cout << std::endl;
}

// 函数用于找到整数的位数
double findDigitCount(double number)
{
    return number + pow(2, 52) + pow(2, 51) - pow(2, 52) - pow(2, 51);
}

// 函数用于找到小数的有效位数
int findFractionalDigitCount(double number)
{
    const double pow5 = pow(2, 51) + pow(2, 52);
    double trac = number + pow5 - pow5;
    double temp = number;
    int digits = 0;
    int64_t int_temp;
    int64_t trac_temp;
    std::memcpy(&int_temp, &temp, sizeof(double));
    std::memcpy(&trac_temp, &trac, sizeof(double));
    while (std::abs(trac_temp - int_temp) >= 3 && digits < 15)
    {
        temp *= 10;
        std::memcpy(&int_temp, &temp, sizeof(double));
        trac = temp + pow5 - pow5;
        std::memcpy(&trac_temp, &trac, sizeof(double));
        digits++;
    }
    return digits;
}

// 函数用于找到小数的有效位数
int findFractionalDigitCountOptimized(double value)
{
    const double pow5 = pow(2, 51) + pow(2, 52);
    double trac = value + pow(2, 52) + pow(2, 51) - pow(2, 52) - pow(2, 51);
    double temp = value;
    int digits = 0;
    int64_t int_temp;
    int64_t trac_temp;
    std::memcpy(&int_temp, &temp, sizeof(double));
    std::memcpy(&trac_temp, &trac, sizeof(double));
    while (std::abs(trac_temp - int_temp) >= 3 && digits < 15)
    {
        temp *= 10;
        std::memcpy(&int_temp, &temp, sizeof(double));
        trac = temp + pow(2, 52) + pow(2, 51) - pow(2, 52) - pow(2, 51);
        std::memcpy(&trac_temp, &trac, sizeof(double));
        digits++;
    }
    return digits;
    // std::ostringstream out;
    // out.precision(15); // 设置精度，确保足够的位数
    // out << value;
    // std::string numStr = out.str();
    //
    // size_t pos = numStr.find('.');
    // if (pos == std::string::npos)
    // {
    //     return 0; // 没有小数部分
    // }
    //
    // // 去掉末尾的零
    // size_t lastNonZero = numStr.find_last_not_of('0');
    // if (lastNonZero != std::string::npos && lastNonZero > pos)
    // {
    //     return lastNonZero - pos;
    // }
    // return numStr.size() - pos - 1;
}


// 测试函数
void testFindDigitCount()
{
    double testNumbers[] = {64.165, 5.123, 123.456, -456.7890, 7890.001, -100000.00001};
    // double testNumbers[] = {123.456};
    int numTests = sizeof(testNumbers) / sizeof(testNumbers[0]);

    for (int i = 0; i < numTests; ++i)
    {
        double number = testNumbers[i];
        double digitCount = findDigitCount(number);
        std::cout << "Number: " << number << " has " << digitCount << " digits." << std::endl;
        // printBinary(number);
        // printBinary(digitCount);
    }
}


// 测试函数
void testFindFractionalDigitCount()
{
    // double testNumbers[] = {0.0, 5.123, 123.456, -456.7890, 7890.001, -100000.00001};
    double testNumbers[] = {64.165};
    int numTests = sizeof(testNumbers) / sizeof(testNumbers[0]);

    for (int i = 0; i < numTests; ++i)
    {
        double number = testNumbers[i];
        int fractionalDigitCount = findFractionalDigitCount(number);
        std::cout << "Number: " << number << " has " << fractionalDigitCount << " fractional digits." << std::endl;
    }
}

// int main()
// {
//     testFindDigitCount();
//     testFindFractionalDigitCount();
//     // double testNumber = 0.1234867459;
//     // int iterations = 100000000;
//     //
//     // // Measure time for original function
//     // auto start = std::chrono::high_resolution_clock::now();
//     // for (int i = 0; i < iterations; ++i)
//     // {
//     //     findFractionalDigitCount(testNumber);
//     // }
//     // auto end = std::chrono::high_resolution_clock::now();
//     // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     // std::cout << "Original function time: " << duration << " ms" << std::endl;
//     //
//     // // Measure time for optimized function
//     // start = std::chrono::high_resolution_clock::now();
//     // for (int i = 0; i < iterations; ++i)
//     // {
//     //     findFractionalDigitCountOptimized(testNumber);
//     // }
//     // end = std::chrono::high_resolution_clock::now();
//     // duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//     // std::cout << "Optimized function time: " << duration << " ms" << std::endl;
//
//     return 0;
//     return 0;
// }
