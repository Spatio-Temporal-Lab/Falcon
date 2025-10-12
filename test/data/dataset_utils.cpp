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
/*
    std::vector<double> read_data(const std::string& file_path, bool show_progress, char delimiter, int decimal_places) {

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

        // 计算精度调整系数（如果需要）
        double precision_multiplier = 1.0;
        if (decimal_places >= 0) {
            precision_multiplier = std::pow(10.0, decimal_places);
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
                        // 根据decimal_places参数调整精度
                        if (decimal_places >= 0) {
                            val = std::round(val * precision_multiplier) / precision_multiplier;
                        }
                        columns[col_idx].emplace_back(val);
                    } else {
                        ++skipped_values;
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
            std::cout << "跳过的无效数值: " << skipped_values << std::endl;
            if (decimal_places >= 0) {
                std::cout << "小数精度: " << decimal_places << " 位" << std::endl;
            }
            std::cout << std::endl;
        }
        return result;
    }
*/

std::vector<double> read_data(const std::string& file_path, int significant_figures, bool show_progress, char delimiter) {

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

    // 纯字符串保留有效位数的函数
    auto truncate_significant = [](std::string cell, int sig_figs) -> std::string {
        if (sig_figs < 0) return cell;
        
        // 去除首尾空格
        size_t start = cell.find_first_not_of(" \t\r\n");
        size_t end = cell.find_last_not_of(" \t\r\n");
        if (start == std::string::npos) return cell;
        cell = cell.substr(start, end - start + 1);
        
        // 处理负号
        bool is_negative = false;
        if (!cell.empty() && cell[0] == '-') {
            is_negative = true;
            cell = cell.substr(1);
        }
        
        // 检查是否为零
        bool is_zero = true;
        for (char c : cell) {
            if (c != '0' && c != '.') {
                is_zero = false;
                break;
            }
        }
        if (is_zero) return "0";
        
        // 找到小数点位置
        size_t dot_pos = cell.find('.');
        
        // 统计有效数字（从第一个非零数字开始）
        std::string digits_only;
        int first_nonzero_pos = -1;  // 相对于小数点的位置
        bool found_nonzero = false;
        int pos_relative_to_dot = (dot_pos == std::string::npos) ? 
                                   static_cast<int>(cell.length()) - 1 : 
                                   static_cast<int>(dot_pos) - 1;
        
        for (size_t i = 0; i < cell.length(); ++i) {
            if (cell[i] == '.') {
                pos_relative_to_dot = -1;
                continue;
            }
            
            if (cell[i] != '0' || found_nonzero) {
                if (!found_nonzero && cell[i] != '0') {
                    first_nonzero_pos = pos_relative_to_dot;
                    found_nonzero = true;
                }
                digits_only += cell[i];
            }
            
            if (cell[i] != '.') {
                pos_relative_to_dot--;
            }
        }
        
        if (digits_only.empty()) return "0";
        
        // 只保留前sig_figs位有效数字
        if (digits_only.length() > static_cast<size_t>(sig_figs)) {
            // 检查是否需要进位
            bool need_round_up = (digits_only[sig_figs] >= '5');
            digits_only = digits_only.substr(0, sig_figs);
            
            if (need_round_up) {
                // 进位处理
                int carry = 1;
                for (int i = digits_only.length() - 1; i >= 0 && carry; --i) {
                    int digit = digits_only[i] - '0' + carry;
                    if (digit == 10) {
                        digits_only[i] = '0';
                    } else {
                        digits_only[i] = '0' + digit;
                        carry = 0;
                    }
                }
                
                if (carry) {
                    // 溢出，需要在前面加1
                    digits_only = "1" + digits_only;
                    first_nonzero_pos++;
                }
            }
        }
        
        // 构建结果字符串
        std::string result;
        
        // 计算小数点应该在哪里
        int dot_position = first_nonzero_pos - sig_figs + 1;
        
        if (first_nonzero_pos >= 0) {
            // 数字 >= 1
            if (first_nonzero_pos - sig_figs + 1 >= 0) {
                // 需要补零（大整数）
                result = digits_only;
                for (int i = 0; i < first_nonzero_pos - sig_figs + 1; ++i) {
                    result += '0';
                }
            } else {
                // 需要添加小数点
                int integer_digits = first_nonzero_pos + 1;
                result = digits_only.substr(0, integer_digits);
                if (digits_only.length() > static_cast<size_t>(integer_digits)) {
                    result += '.';
                    result += digits_only.substr(integer_digits);
                }
            }
        } else {
            // 数字 < 1 (0.00xxx形式)
            result = "0.";
            for (int i = 0; i < -first_nonzero_pos - 1; ++i) {
                result += '0';
            }
            result += digits_only;
        }
        
        if (is_negative) {
            result = "-" + result;
        }
        
        return result;
    };

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
                // 进行有效位数处理（纯字符串）
                std::string processed_cell = truncate_significant(cell, significant_figures);
                
                char* endptr = nullptr;
                double val = std::strtod(processed_cell.c_str(), &endptr);
                
                // 增强的数值校验（包括无穷大检查）
                if (endptr != processed_cell.c_str() && !std::isnan(val) && std::isfinite(val)) {
                    columns[col_idx].emplace_back(val);
                } else {
                    ++skipped_values;
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
        std::cout << "跳过的无效数值: " << skipped_values << std::endl;
        if (significant_figures >= 0) {
            std::cout << "有效位数: " << significant_figures << " 位" << std::endl;
        }
        std::cout << std::endl;
    }
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
