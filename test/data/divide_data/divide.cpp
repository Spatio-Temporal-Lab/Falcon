#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <algorithm>

namespace fs = std::filesystem;

class EnhancedCSVSplitter {
private:
    std::string inputFilename;
    std::vector<double> targetSizesInMB;
    std::string headerLine;
    std::string outputDirectory;
    char delimiter;
    size_t totalColumns;
    size_t totalRows;
    
public:
    /**
     * 构造函数支持自定义范围和步长
     * @param filename 输入文件名
     * @param startSizeMB 起始大小（MB）
     * @param endSizeMB 结束大小（MB）
     * @param stepSizeMB 步长（MB）
     * @param delim 分隔符，默认为逗号
     */
    EnhancedCSVSplitter(const std::string& filename, 
                       double startSizeMB = 30.0, 
                       double endSizeMB = 300.0, 
                       double stepSizeMB = 30.0,
                       char delim = ',') 
        : inputFilename(filename), delimiter(delim), totalColumns(0), totalRows(0) {
        
        // 生成目标大小列表
        generateTargetSizes(startSizeMB, endSizeMB, stepSizeMB);
        
        // 创建输出目录名
        createOutputDirectory();
    }
    
    /**
     * 构造函数支持自定义大小列表
     * @param filename 输入文件名
     * @param customSizes 自定义大小列表（MB）
     * @param delim 分隔符，默认为逗号
     */
    EnhancedCSVSplitter(const std::string& filename, 
                       const std::vector<double>& customSizes,
                       char delim = ',') 
        : inputFilename(filename), targetSizesInMB(customSizes), delimiter(delim), totalColumns(0), totalRows(0) {
        
        // 创建输出目录名
        createOutputDirectory();
    }
    
private:
    /**
     * 生成目标大小列表
     */
    void generateTargetSizes(double startSizeMB, double endSizeMB, double stepSizeMB) {
        if (stepSizeMB <= 0) {
            std::cerr << "错误: 步长必须大于0" << std::endl;
            return;
        }
        
        if (startSizeMB > endSizeMB) {
            std::cerr << "错误: 起始大小不能大于结束大小" << std::endl;
            return;
        }
        
        targetSizesInMB.clear();
        for (double size = startSizeMB; size <= endSizeMB + 1e-9; size += stepSizeMB) {
            targetSizesInMB.push_back(size);
        }
        
        std::cout << "生成目标大小列表: ";
        for (size_t i = 0; i < targetSizesInMB.size(); ++i) {
            std::cout << std::fixed << std::setprecision(1) << targetSizesInMB[i] << "MB";
            if (i < targetSizesInMB.size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    
    /**
     * 创建输出目录
     */
    void createOutputDirectory() {
        // 提取原文件名（不含扩展名和路径）
        size_t lastDot = inputFilename.find_last_of('.');
        size_t lastSlash = inputFilename.find_last_of("/\\");
        
        std::string baseName;
        if (lastSlash != std::string::npos) {
            baseName = inputFilename.substr(lastSlash + 1);
        } else {
            baseName = inputFilename;
        }
        
        if (lastDot != std::string::npos && lastDot > lastSlash) {
            baseName = baseName.substr(0, lastDot - (lastSlash != std::string::npos ? lastSlash + 1 : 0));
        }
        
        outputDirectory = baseName + "_split_output";
        
        // 创建目录
        try {
            if (!fs::exists(outputDirectory)) {
                fs::create_directory(outputDirectory);
                std::cout << "创建输出目录: " << outputDirectory << std::endl;
            } else {
                std::cout << "输出目录已存在: " << outputDirectory << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "创建目录失败: " << e.what() << std::endl;
            outputDirectory = "."; // 如果创建失败，使用当前目录
        }
    }
    
    /**
     * 分析CSV文件结构，获取精确的数据量信息
     */
    bool analyzeCSVStructure() {
        std::ifstream file(inputFilename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << inputFilename << std::endl;
            return false;
        }
        
        std::cout << "正在分析CSV文件结构..." << std::endl;
        
        // 读取头部
        if (!std::getline(file, headerLine)) {
            std::cerr << "无法读取文件头部" << std::endl;
            file.close();
            return false;
        }
        
        // 计算列数
        std::stringstream ss(headerLine);
        std::string cell;
        totalColumns = 0;
        while (std::getline(ss, cell, delimiter)) {
            totalColumns++;
        }
        
        // 计算行数（不包括头部）
        totalRows = 0;
        std::string line;
        while (std::getline(file, line)) {
            totalRows++;
        }
        
        file.close();
        
        std::cout << "文件分析完成:" << std::endl;
        std::cout << "  - 列数: " << totalColumns << std::endl;
        std::cout << "  - 数据行数: " << totalRows << std::endl;
        std::cout << "  - 总数据点数: " << totalColumns * totalRows << std::endl;
        std::cout << "  - 估计总数据大小: " << std::fixed << std::setprecision(2) 
                  << (double)(totalColumns * totalRows * sizeof(double)) / (1024.0 * 1024.0) << " MB" << std::endl;
        
        return totalColumns > 0 && totalRows > 0;
    }
    
    /**
     * 计算需要多少行数据来达到目标大小
     */
    size_t calculateRowsForTargetSize(double targetSizeMB) {
        double targetSizeBytes = targetSizeMB * 1024.0 * 1024.0;
        double bytesPerRow = totalColumns * sizeof(double);
        size_t targetRows = static_cast<size_t>(std::round(targetSizeBytes / bytesPerRow));
        
        // 确保不超过总行数
        return std::min(targetRows, totalRows);
    }
    
    /**
     * 生成输出文件名
     */
    std::string generateOutputFilename(double sizeInMB) {
        // 提取原文件名（不含扩展名和路径）
        size_t lastDot = inputFilename.find_last_of('.');
        size_t lastSlash = inputFilename.find_last_of("/\\");
        
        std::string baseName;
        if (lastSlash != std::string::npos) {
            baseName = inputFilename.substr(lastSlash + 1);
        } else {
            baseName = inputFilename;
        }
        
        if (lastDot != std::string::npos && lastDot > lastSlash) {
            baseName = baseName.substr(0, lastDot - (lastSlash != std::string::npos ? lastSlash + 1 : 0));
        }
        
        std::ostringstream oss;
        oss << outputDirectory << "/" << baseName << "_" 
            << std::fixed << std::setprecision(1) << sizeInMB << "MB.csv";
        return oss.str();
    }
    
    /**
     * 创建指定大小的分割文件
     */
    bool createSplitFile(double targetSizeMB) {
        size_t targetRows = calculateRowsForTargetSize(targetSizeMB);
        
        if (targetRows == 0) {
            std::cout << "跳过 " << targetSizeMB << "MB - 目标大小太小" << std::endl;
            return true;
        }
        
        if (targetRows > totalRows) {
            std::cout << "跳过 " << targetSizeMB << "MB - 目标大小超过文件总大小" << std::endl;
            return true;
        }
        
        std::ifstream inputFile(inputFilename);
        if (!inputFile.is_open()) {
            return false;
        }
        
        std::string outputFilename = generateOutputFilename(targetSizeMB);
        std::ofstream outputFile(outputFilename);
        if (!outputFile.is_open()) {
            inputFile.close();
            return false;
        }
        
        // 写入头部
        outputFile << headerLine << "\n";
        
        // 跳过输入文件的头部
        std::string line;
        std::getline(inputFile, line);
        
        size_t linesWritten = 0;
        size_t validDataPoints = 0;
        
        std::cout << "正在创建 " << targetSizeMB << "MB 文件..." << std::endl;
        
        // 读取并写入数据行，直到达到目标行数
        while (std::getline(inputFile, line) && linesWritten < targetRows) {
            // 验证行中的有效数据点数
            size_t validPointsInLine = countValidDataPoints(line);
            
            outputFile << line << "\n";
            linesWritten++;
            validDataPoints += validPointsInLine;
            
            // 每10000行显示一次进度
            if (linesWritten % 10000 == 0 || linesWritten == targetRows) {
                double currentSizeMB = (double)(validDataPoints * sizeof(double)) / (1024.0 * 1024.0);
                double progress = (double)linesWritten / targetRows * 100.0;
                
                std::cout << "  进度: " << std::fixed << std::setprecision(1) << progress << "% "
                          << "(" << linesWritten << "/" << targetRows << " 行) "
                          << "当前数据大小: " << std::setprecision(3) << currentSizeMB << " MB" << std::endl;
            }
        }
        
        inputFile.close();
        outputFile.close();
        
        // 计算实际大小
        double actualSizeMB = (double)(validDataPoints * sizeof(double)) / (1024.0 * 1024.0);
        double accuracy = (actualSizeMB / targetSizeMB) * 100.0;
        
        std::cout << "✓ 创建文件: " << fs::path(outputFilename).filename().string() << std::endl;
        std::cout << "  - 目标大小: " << std::fixed << std::setprecision(3) << targetSizeMB << " MB" << std::endl;
        std::cout << "  - 实际大小: " << std::setprecision(3) << actualSizeMB << " MB" << std::endl;
        std::cout << "  - 精度: " << std::setprecision(2) << accuracy << "%" << std::endl;
        std::cout << "  - 行数: " << linesWritten + 1 << " (含头部)" << std::endl;
        std::cout << "  - 有效数据点: " << validDataPoints << std::endl;
        std::cout << std::endl;
        
        return true;
    }
    
    /**
     * 计算一行中的有效数据点数量
     */
    size_t countValidDataPoints(const std::string& line) {
        std::stringstream ss(line);
        std::string cell;
        size_t validCount = 0;
        
        while (std::getline(ss, cell, delimiter)) {
            if (!cell.empty()) {
                char* endptr = nullptr;
                double val = std::strtod(cell.c_str(), &endptr);
                
                // 增强的数值校验（包括无穷大检查）
                if (endptr != cell.c_str() && !std::isnan(val) && std::isfinite(val)) {
                    validCount++;
                }
            }
        }
        
        return validCount;
    }
    
public:
    /**
     * 分割CSV文件
     */
    bool splitCSV() {
        if (!analyzeCSVStructure()) {
            return false;
        }
        
        if (targetSizesInMB.empty()) {
            std::cerr << "没有指定目标大小" << std::endl;
            return false;
        }
        
        std::cout << "========================================" << std::endl;
        std::cout << "开始分割文件..." << std::endl;
        std::cout << "========================================" << std::endl;
        
        // 为每个目标大小生成文件
        bool success = true;
        for (double targetSize : targetSizesInMB) {
            if (!createSplitFile(targetSize)) {
                std::cerr << "创建 " << targetSize << "MB 文件失败" << std::endl;
                success = false;
                continue;
            }
        }
        
        return success;
    }
    
    /**
     * 显示当前配置信息
     */
    void showConfiguration() {
        std::cout << "CSV分割器配置:" << std::endl;
        std::cout << "  - 输入文件: " << inputFilename << std::endl;
        std::cout << "  - 分隔符: '" << delimiter << "'" << std::endl;
        std::cout << "  - 输出目录: " << outputDirectory << std::endl;
        std::cout << "  - 目标大小数量: " << targetSizesInMB.size() << std::endl;
        std::cout << "  - 大小范围: " << std::fixed << std::setprecision(1) 
                  << *std::min_element(targetSizesInMB.begin(), targetSizesInMB.end()) << "MB - " 
                  << *std::max_element(targetSizesInMB.begin(), targetSizesInMB.end()) << "MB" << std::endl;
    }
};

/**
 * 显示使用帮助
 */
void showUsage(const char* programName) {
    std::cout << "增强版CSV分割器 - 基于数据大小精确分割" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "用法 1 (自定义范围): " << programName << " <CSV文件> <起始大小MB> <结束大小MB> <步长MB> [分隔符]" << std::endl;
    std::cout << "用法 2 (默认范围): " << programName << " <CSV文件>" << std::endl;
    std::cout << std::endl;
    std::cout << "示例:" << std::endl;
    std::cout << "  " << programName << " data.csv                    # 默认: 30-300MB, 步长30MB" << std::endl;
    std::cout << "  " << programName << " data.csv 10 100 10          # 10-100MB, 步长10MB" << std::endl;
    std::cout << "  " << programName << " data.csv 5 50 5 ';'         # 5-50MB, 步长5MB, 分号分隔" << std::endl;
    std::cout << std::endl;
    std::cout << "特性:" << std::endl;
    std::cout << "  - 基于实际double数据量进行精确分割" << std::endl;
    std::cout << "  - 支持自定义大小范围和步长" << std::endl;
    std::cout << "  - 高精度大小控制（通常误差<1%）" << std::endl;
    std::cout << "  - 自动验证数据有效性" << std::endl;
    std::cout << "  - 详细的进度和统计信息" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2 || argc > 6) {
        showUsage(argv[0]);
        return 1;
    }
    
    std::string inputFile = argv[1];
    
    // 检查文件是否存在
    std::ifstream testFile(inputFile);
    if (!testFile.is_open()) {
        std::cerr << "错误: 无法打开文件 " << inputFile << std::endl;
        return 1;
    }
    testFile.close();
    
    EnhancedCSVSplitter* splitter = nullptr;
    
    if (argc == 2) {
        // 默认配置
        splitter = new EnhancedCSVSplitter(inputFile);
    } else if (argc >= 5) {
        // 自定义范围
        double startSize = std::stod(argv[2]);
        double endSize = std::stod(argv[3]);
        double stepSize = std::stod(argv[4]);
        char delimiter = (argc == 6) ? argv[5][0] : ',';
        
        if (startSize <= 0 || endSize <= 0 || stepSize <= 0) {
            std::cerr << "错误: 所有大小参数必须大于0" << std::endl;
            return 1;
        }
        
        splitter = new EnhancedCSVSplitter(inputFile, startSize, endSize, stepSize, delimiter);
    } else {
        showUsage(argv[0]);
        return 1;
    }
    
    std::cout << "增强版CSV分割器启动" << std::endl;
    std::cout << "========================================" << std::endl;
    
    splitter->showConfiguration();
    std::cout << "========================================" << std::endl;
    
    if (splitter->splitCSV()) {
        std::cout << "========================================" << std::endl;
        std::cout << "所有文件处理完成!" << std::endl;
        std::cout << "输出文件保存在: " << std::endl;
        std::cout << "  " << fs::current_path() << "/" << fs::path(inputFile).stem().string() << "_split_output/" << std::endl;
    } else {
        std::cerr << "处理过程中出现错误!" << std::endl;
        delete splitter;
        return 1;
    }
    
    delete splitter;
    return 0;
}