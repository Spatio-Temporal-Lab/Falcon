#include "dataset_utils.hpp"

namespace fs = std::filesystem;

std::vector<Column> get_dynamic_dataset(const std::string& directory_path) {
    std::vector<Column> dataset;
    uint64_t id = 1;

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".csv") {
            Column column;
            column.id = id++;
            column.name = entry.path().stem().string();
            column.csv_file_path = entry.path().string();
            column.binary_file_path = entry.path().parent_path().string() + "/" + entry.path().stem().string() + ".bin";
            dataset.push_back(column);
        }
    }
    return dataset;
}
// 读取浮点数数据文件
// std::vector<double> read_data(const std::string& file_path,bool a) {
//     if(a)
//     {
//         std::cout<<file_path<<" 01\n";
//     }
//     std::vector<double> data;
//     std::ifstream file(file_path);
//     if (!file) {
//         std::cerr << "无法打开文件: " << file_path << std::endl;
//         return data;
//     }
//     std::string line;
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         double value;
//         if (ss >> value) { // 从字符串流读取 double
//             data.push_back(value);
//         }
//     }
//     return data;
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
                std::cerr << "\n无效数值: " << cell << "，跳过（行：" << current_line << "）" << std::endl;
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