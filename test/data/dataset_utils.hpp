#ifndef DATASET_UTILS_HPP
#define DATASET_UTILS_HPP

#include <vector>
#include <string>
#include <filesystem>
#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cctype>
struct Column {
    uint64_t id;
    std::string name;
    std::string csv_file_path;
    std::string binary_file_path;
    uint8_t factor = 14;
    uint16_t exponent = 10;
    uint16_t exceptions_count = 5;
    uint8_t bit_width = 16;
    bool suitable_for_cutting = false;
};

std::vector<Column> get_dynamic_dataset(const std::string& directory_path, bool show_progress = true, char delimiter = ',');

// 读取浮点数数据文件
std::vector<double> read_data(const std::string& file_path, bool a=1, char delimiter = ','); 
#endif // DATASET_UTILS_HPP
