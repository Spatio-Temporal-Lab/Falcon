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

std::vector<double> read_data(const std::string& file_path, bool a, char delimiter) {
    if (a) {
        std::cout << file_path << " 01\n";
    }
    std::vector<double> result;
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "无法打开文件: " << file_path << std::endl;
        return {};
    }

    std::vector<size_t> column_sizes;
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value_str;
        size_t col_index = 0;

        while (std::getline(ss, value_str, delimiter)) {
            try {
                double value = std::stod(value_str);
                if (col_index >= column_sizes.size()) {
                    column_sizes.push_back(0);
                    result.resize(result.size() + 1); // 预分配空间
                }
                // 计算插入位置：列优先
                size_t insert_pos = 0;
                for (size_t i = 0; i < col_index; ++i) {
                    insert_pos += column_sizes[i];
                }
                insert_pos += column_sizes[col_index];
                result.insert(result.begin() + insert_pos, value);
                column_sizes[col_index]++;
                col_index++;
            } catch (const std::exception& e) {
                std::cerr << "无效数值: " << value_str << "，跳过" << std::endl;
            }
        }
    }
    return result;
}