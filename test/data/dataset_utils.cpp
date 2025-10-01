#include "dataset_utils.hpp"

namespace fs = std::filesystem;


// 辅助函数：分析浮点数数据特征
void analyze_column_data(Column& column, const std::vector<double>& data) {
    if (data.empty()) return;
    
    // 计算数据范围
    double min_val = *std::min_element(data.begin(), data.end());
    double max_val = *std::max_element(data.begin(), data.end());
    double range = max_val - min_val;
    
    // 分析数据精度需求
    std::vector<int> decimal_places;
    for (const auto& val : data) {
        std::string str_val = std::to_string(val);
        size_t decimal_pos = str_val.find('.');
        if (decimal_pos != std::string::npos) {
            // 移除尾随零
            str_val.erase(str_val.find_last_not_of('0') + 1, std::string::npos);
            if (str_val.back() == '.') str_val.pop_back();
            
            int places = str_val.length() - decimal_pos - 1;
            if (places > 0) decimal_places.push_back(places);
        }
    }
    
    // 设置压缩参数
    if (!decimal_places.empty()) {
        int avg_decimal_places = std::accumulate(decimal_places.begin(), decimal_places.end(), 0) / decimal_places.size();
        column.factor = std::min(20, std::max(8, avg_decimal_places + 2));
    }
    
    // 根据数据范围调整指数位宽
    if (range > 1000000) {
        column.exponent = 12;
        column.bit_width = 32;
    } else if (range > 1000) {
        column.exponent = 11;
        column.bit_width = 24;
    } else {
        column.exponent = 10;
        column.bit_width = 16;
    }
    
    // 计算异常值数量
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();
    double variance = 0;
    for (const auto& val : data) {
        variance += (val - mean) * (val - mean);
    }
    variance /= data.size();
    double std_dev = std::sqrt(variance);
    
    int outliers = 0;
    for (const auto& val : data) {
        if (std::abs(val - mean) > 2 * std_dev) {
            outliers++;
        }
    }
    column.exceptions_count = std::min(255, std::max(1, outliers));
    
    // 判断是否适合切割（基于数据分布）
    column.suitable_for_cutting = (range > mean * 0.1) && (data.size() > 1000);
}

// 辅助函数：验证字符串是否为有效的数值
bool is_valid_number(const std::string& str) {
    if (str.empty()) return false;
    
    // 检查是否包含无效字符
    bool has_dot = false;
    bool has_e = false;
    bool has_sign = false;
    
    for (size_t i = 0; i < str.length(); ++i) {
        char c = str[i];
        
        // 允许的字符：数字、小数点、科学记数法(e/E)、正负号
        if (std::isdigit(c)) {
            continue;
        } else if (c == '.' && !has_dot && !has_e) {
            has_dot = true;
        } else if ((c == 'e' || c == 'E') && !has_e && i > 0) {
            has_e = true;
            has_sign = false; // 重置符号标志，因为e后面可以有符号
        } else if ((c == '+' || c == '-') && (!has_sign || (has_e && i > 0 && (str[i-1] == 'e' || str[i-1] == 'E')))) {
            has_sign = true;
        } else {
            return false; // 无效字符
        }
    }
    
    return true;
}

// 辅助函数：安全地将字符串转换为double
bool safe_stod(const std::string& str, double& result) {
    if (!is_valid_number(str)) {
        return false;
    }
    
    try {
        result = std::stod(str);
        // 检查转换结果是否为有限数值
        if (!std::isfinite(result)) {
            return false;
        }
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

// 辅助函数：从CSV文件读取数据
std::vector<double> read_csv_column(const std::string& file_path, const std::string& column_name, char delimiter) {
    std::vector<double> data;
    std::ifstream file(file_path);
    
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return data;
    }
    
    std::string line;
    std::vector<std::string> headers;
    int target_column_index = -1;
    
    // 读取表头
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string header;
        int index = 0;
        
        while (std::getline(ss, header, delimiter)) {
            // 去除可能的空白字符和引号
            header.erase(0, header.find_first_not_of(" \t\r\n\""));
            header.erase(header.find_last_not_of(" \t\r\n\"") + 1);
            
            if (header == column_name) {
                target_column_index = index;
                break;
            }
            index++;
        }
    }
    
    if (target_column_index == -1) {
        std::cerr << "找不到列: " << column_name << " 在文件 " << file_path << std::endl;
        return data;
    }
    
    // 读取数据行
    int line_number = 1; // 从1开始，因为第0行是表头
    while (std::getline(file, line)) {
        line_number++;
        std::stringstream ss(line);
        std::string cell;
        int current_index = 0;
        
        while (std::getline(ss, cell, delimiter)) {
            if (current_index == target_column_index) {
                // 去除空白字符和引号
                cell.erase(0, cell.find_first_not_of(" \t\r\n\""));
                cell.erase(cell.find_last_not_of(" \t\r\n\"") + 1);
                
                // 检查常见的非数值表示
                if (cell.empty() || 
                    cell == "NULL" || cell == "null" || 
                    cell == "NaN" || cell == "nan" ||
                    cell == "NA" || cell == "na" ||
                    cell == "inf" || cell == "INF" ||
                    cell == "-inf" || cell == "-INF" ||
                    cell == "#N/A" || cell == "#ERROR" ||
                    cell == "?" || cell == "-") {
                    break; // 跳过这个值
                }
                
                double value;
                if (safe_stod(cell, value)) {
                    data.push_back(value);
                } else {
                    // 输出调试信息（可选）
                    // std::cerr << "警告: 无法解析数值 '" << cell << "' 在文件 " 
                    //           << file_path << " 第 " << line_number << " 行" << std::endl;
                }
                break;
            }
            current_index++;
        }
    }
    
    return data;
}

// 主函数实现
std::vector<Column> get_dynamic_dataset(const std::string& directory_path, bool show_progress, char delimiter) {
    std::vector<Column> columns;
    
    if (!std::filesystem::exists(directory_path)) {
        std::cerr << "目录不存在: " << directory_path << std::endl;
        return columns;
    }
    
    uint64_t column_id = 0;
    
    // 遍历目录中的所有CSV文件
    for (const auto& entry : std::filesystem::recursive_directory_iterator(directory_path)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".csv") {
            continue;
        }
        
        std::ifstream file(entry.path());
        if (!file.is_open()) continue;
        
        std::string header_line;
        if (!std::getline(file, header_line)) continue;
        
        // 解析表头获取列名
        std::stringstream ss(header_line);
        std::string column_name;
        
        while (std::getline(ss, column_name, ',')) {
            // 清理列名
            column_name.erase(0, column_name.find_first_not_of(" \t\r\n\""));
            column_name.erase(column_name.find_last_not_of(" \t\r\n\"") + 1);
            
            if (!column_name.empty()) {
                Column column;
                column.id = column_id++;
                column.name = column_name;
                column.csv_file_path = entry.path().string();
                
                // 设置默认参数
                column.factor = 10;
                column.exponent = 11;
                column.bit_width = 24;
                column.exceptions_count = 10;
                column.suitable_for_cutting = true;
                
                columns.push_back(column);
            }
        }
        
        file.close();
    }
    
    std::cout << "找到 " << columns.size() << " 个列" << std::endl;
    return columns;
}
// // 统一的数据验证函数
// bool is_valid_number(const std::string& str) {
//     if (str.empty()) return false;
    
//     // 去除前后空白字符
//     std::string trimmed = str;
//     trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r\f\v"));
//     trimmed.erase(trimmed.find_last_not_of(" \t\n\r\f\v") + 1);
    
//     if (trimmed.empty()) return false;
    
//     // 使用 strtod 进行验证（更鲁棒）
//     char* endptr = nullptr;
//     double val = std::strtod(trimmed.c_str(), &endptr);
    
//     // 检查是否完全转换并且是有效数值
//     return (endptr == trimmed.c_str() + trimmed.length()) && 
//            !std::isnan(val) && std::isfinite(val);
// }

// // 安全的字符串转双精度浮点数函数
// double safe_string_to_double(const std::string& str) {
//     // 去除前后空白字符
//     std::string trimmed = str;
//     trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r\f\v"));
//     trimmed.erase(trimmed.find_last_not_of(" \t\n\r\f\v") + 1);
    
//     char* endptr = nullptr;
//     double val = std::strtod(trimmed.c_str(), &endptr);
    
//     // 如果转换失败，返回NaN
//     if (endptr != trimmed.c_str() + trimmed.length()) {
//         return std::numeric_limits<double>::quiet_NaN();
//     }
    
//     return val;
// }

// // 新增的验证和预处理函数
// bool validate_and_preprocess_csv(const std::string& file_path, bool show_progress, char delimiter) {
//     // 第一次读取：获取元数据和验证文件结构
//     size_t total_lines = 0;
//     size_t num_columns = 0;
//     size_t valid_data_count = 0;
    
//     {
//         std::ifstream meta_file(file_path);
//         if (!meta_file) {
//             std::cerr << "无法打开文件: " << file_path << std::endl;
//             return false;
//         }

//         // 获取列数和总行数
//         std::string first_line;
//         if (std::getline(meta_file, first_line)) {
//             std::stringstream ss(first_line);
//             std::string cell;
//             while (std::getline(ss, cell, delimiter)) ++num_columns;
//             ++total_lines;
//             while (std::getline(meta_file, first_line)) ++total_lines;
//         }
        
//         if (num_columns == 0 || total_lines == 0) {
//             std::cerr << "文件为空或格式错误: " << file_path << std::endl;
//             return false;
//         }
//     }

//     // 第二次读取：验证数据质量
//     std::ifstream file(file_path);
//     size_t current_line = 0;
//     size_t invalid_count = 0;
    
//     if (show_progress) {
//         std::cout << "验证文件: " << fs::path(file_path).filename() << std::endl;
//         std::cout << "验证进度:\n0%   10%  20%  30%  40%  50%  60%  70%  80%  90%  100%\n|----|----|----|----|----|----|----|----|----|----|\n";
//     }

//     std::string line;
//     while (std::getline(file, line)) {
//         ++current_line;
        
//         // 更新进度条
//         if (show_progress && total_lines > 0 && (current_line % std::max(1UL, total_lines/50) == 0 || current_line == total_lines)) {
//             float progress = static_cast<float>(current_line) / total_lines;
//             int bar_width = static_cast<int>(progress * 50);
            
//             std::cout << "\r[";
//             for(int i = 0; i < 50; ++i) {
//                 if(i < bar_width) std::cout << "=";
//                 else if(i == bar_width) std::cout << ">";
//                 else std::cout << " ";
//             }
//             std::cout << "] " 
//                     << std::fixed << std::setw(5) << std::setprecision(1) 
//                     << progress * 100 << "%";
//             std::cout.flush();
//         }

//         // 验证当前行的数据
//         std::stringstream ss(line);
//         std::string cell;
//         size_t col_idx = 0;
//         size_t valid_cells_in_row = 0;
        
//         while (std::getline(ss, cell, delimiter) && col_idx < num_columns) {
//             // 统一的数据验证逻辑
//             if (is_valid_number(cell)) {
//                 ++valid_cells_in_row;
//             } else if (!cell.empty()) {
//                 ++invalid_count;
//                 if (show_progress && invalid_count % 100000000 == 0) {
//                     std::cerr << "\n已发现 " << invalid_count << " 个无效数值: '" << cell << "'..." << std::endl;
//                 }
//             }
//             ++col_idx;
//         }
        
//         // 统计有效数据
//         valid_data_count += valid_cells_in_row;
//     }

//     if (show_progress) {
//         std::cout << std::endl;
//     }

//     // 计算数据质量指标
//     size_t total_expected_cells = total_lines * num_columns;
//     double data_quality_ratio = static_cast<double>(valid_data_count) / total_expected_cells;
    
//     // 输出验证结果
//     std::cout << "文件验证完成: " << fs::path(file_path).filename() << std::endl;
//     std::cout << "总行数: " << total_lines << ", 列数: " << num_columns << std::endl;
//     std::cout << "有效数据: " << valid_data_count << "/" << total_expected_cells 
//               << " (" << std::fixed << std::setprecision(1) << data_quality_ratio * 100 << "%)" << std::endl;
//     std::cout << "无效数据: " << invalid_count << std::endl;
    
//     // 设定数据质量阈值（可根据需要调整）
//     const double MIN_DATA_QUALITY_THRESHOLD = 0.5; // 至少50%的数据有效
    
//     if (data_quality_ratio >= MIN_DATA_QUALITY_THRESHOLD) {
//         std::cout << "✓ 数据质量合格，包含在数据集中" << std::endl << std::endl;
//         return true;
//     } else {
//         std::cout << "✗ 数据质量不合格（低于" << MIN_DATA_QUALITY_THRESHOLD * 100 << "%），跳过此文件" << std::endl << std::endl;
//         return false;
//     }
// }


// std::vector<Column> get_dynamic_dataset(const std::string& directory_path, bool show_progress, char delimiter) {
//     std::vector<Column> dataset;
//     uint64_t id = 1;

//     for (const auto& entry : fs::directory_iterator(directory_path)) {
//         if (entry.is_regular_file() && entry.path().extension() == ".csv") {
//             Column column;
//             column.id = id++;
//             column.name = entry.path().stem().string();
//             column.csv_file_path = entry.path().string();
//             column.binary_file_path = entry.path().parent_path().string() + "/" + entry.path().stem().string() + ".bin";
            
//             // 验证并预处理CSV文件数据
//             if (validate_and_preprocess_csv(column.csv_file_path, show_progress, delimiter)) {
//                 dataset.push_back(column);
//             } else {
//                 std::cerr << "跳过文件（数据验证失败）: " << column.csv_file_path << std::endl;
//             }
//         }
//     }
//     return dataset;
// }


std::vector<double> read_data(const std::string& file_path, bool show_progress, char delimiter) {

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
        
        if (num_columns == 0 || total_lines == 0) {
            std::cerr << "文件为空或格式错误: " << file_path << std::endl;
            return {};
        }
    }

    // 初始化列存储
    std::vector<std::vector<double>> columns(num_columns);
    for (auto& col : columns) {
        col.reserve(total_lines + 1000);  // 预分配+缓冲
    }

    // 第二次读取：处理数据
    std::ifstream file(file_path);
    size_t current_line = 0;
    size_t skipped_values = 0;
    
    // 进度条初始化
    if (show_progress) {
        std::cout << "读取数据: " << fs::path(file_path).filename() << std::endl;
        std::cout << "读取进度:\n0%   10%  20%  30%  40%  50%  60%  70%  80%  90%  100%\n|----|----|----|----|----|----|----|----|----|----|\n";
    }

    std::string line;
    while (std::getline(file, line)) {
        ++current_line;
        
        // 更新进度条
        if (show_progress && total_lines > 0 && (current_line % std::max(1UL, total_lines/50) == 0 || current_line == total_lines)) {
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
            if (!cell.empty()) {
                char* endptr = nullptr;
                double val = std::strtod(cell.c_str(), &endptr);
                
                // 增强的数值校验（包括无穷大检查）
                if (endptr != cell.c_str() && !std::isnan(val) && std::isfinite(val)) {
                    columns[col_idx].emplace_back(val);
                } else {
                    ++skipped_values;
                    // 只在调试模式或跳过数量较多时输出警告
                    // if (skipped_values % 10000 == 0) {
                        // std::cerr << "\n已跳过 " << skipped_values << " 个无效数值..." << std::endl;
                    // }
                }
            }
            ++col_idx;
        }
    }

    // 找到最大行数（处理不规则数据）
    size_t max_rows = 0;
    for (const auto& col : columns) {
        max_rows = std::max(max_rows, col.size());
    }

    // 合并列数据（改进的内存管理）
    const size_t estimated_elements = max_rows * num_columns;
    result.reserve(estimated_elements);
    
    for (size_t col = 0; col < num_columns; ++col) {
        for (size_t row = 0; row < max_rows; ++row) {
            if (row < columns[col].size()) {
                result.emplace_back(columns[col][row]);
            }
        }
    }

    if (show_progress) {
        std::cout << "\n\n数据读取完成!" << std::endl;
        std::cout << "实际读取数据量: " << result.size() << "/" << estimated_elements;
        if (estimated_elements > 0) {
            std::cout << " (" << std::round(result.size()*100.0/estimated_elements) << "%)";
        }
        std::cout << std::endl;
        std::cout << "跳过的无效数值: " << skipped_values << std::endl << std::endl;
    }
    // printf("数据大小: %0.6f MB",result.size()*sizeof(double)/1024.0/1024.0);
    return result;
}



std::vector<float> read_data_float(const std::string& file_path, bool show_progress, char delimiter) {

    std::vector<float> result;
    
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
        
        if (num_columns == 0 || total_lines == 0) {
            std::cerr << "文件为空或格式错误: " << file_path << std::endl;
            return {};
        }
    }

    // 初始化列存储
    std::vector<std::vector<float>> columns(num_columns);
    for (auto& col : columns) {
        col.reserve(total_lines + 1000);  // 预分配+缓冲
    }

    // 第二次读取：处理数据
    std::ifstream file(file_path);
    size_t current_line = 0;
    size_t skipped_values = 0;
    
    // 进度条初始化
    if (show_progress) {
        std::cout << "读取数据: " << fs::path(file_path).filename() << std::endl;
        std::cout << "读取进度:\n0%   10%  20%  30%  40%  50%  60%  70%  80%  90%  100%\n|----|----|----|----|----|----|----|----|----|----|\n";
    }

    std::string line;
    while (std::getline(file, line)) {
        ++current_line;
        
        // 更新进度条
        if (show_progress && total_lines > 0 && (current_line % std::max(1UL, total_lines/50) == 0 || current_line == total_lines)) {
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
            if (!cell.empty()) {
                char* endptr = nullptr;
                float val = std::strtof(cell.c_str(), &endptr);
                
                // 增强的数值校验（包括无穷大检查）
                if (endptr != cell.c_str() && !std::isnan(val) && std::isfinite(val)) {
                    columns[col_idx].emplace_back(val);
                } else {
                    ++skipped_values;
                    // 只在调试模式或跳过数量较多时输出警告
                    // if (skipped_values % 10000 == 0) {
                        // std::cerr << "\n已跳过 " << skipped_values << " 个无效数值..." << std::endl;
                    // }
                }
            }
            ++col_idx;
        }
    }

    // 找到最大行数（处理不规则数据）
    size_t max_rows = 0;
    for (const auto& col : columns) {
        max_rows = std::max(max_rows, col.size());
    }

    // 合并列数据（改进的内存管理）
    const size_t estimated_elements = max_rows * num_columns;
    result.reserve(estimated_elements);
    
    for (size_t col = 0; col < num_columns; ++col) {
        for (size_t row = 0; row < max_rows; ++row) {
            if (row < columns[col].size()) {
                result.emplace_back(columns[col][row]);
            }
        }
    }

    if (show_progress) {
        std::cout << "\n\n数据读取完成!" << std::endl;
        std::cout << "实际读取数据量: " << result.size() << "/" << estimated_elements;
        if (estimated_elements > 0) {
            std::cout << " (" << std::round(result.size()*100.0/estimated_elements) << "%)";
        }
        std::cout << std::endl;
        std::cout << "跳过的无效数值: " << skipped_values << std::endl << std::endl;
    }
    // printf("数据大小: %0.6f MB",result.size()*sizeof(double)/1024.0/1024.0);
    return result;
}
