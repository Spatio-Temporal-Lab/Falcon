#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace fs = std::filesystem;

class CSVSplitter {
private:
    std::string inputFilename;
    std::vector<size_t> targetSizes;
    std::string headerLine;
    std::string outputDirectory;
    
public:
    CSVSplitter(const std::string& filename) : inputFilename(filename) {
        // 预定义的目标大小（MB）
        std::vector<int> sizesInMB = {500};//{30, 60, 90, 120, 150, 180, 210, 240, 270, 300};
        for (int size : sizesInMB) {
            targetSizes.push_back(size * 1024 * 1024); // 转换为字节
        }
        
        // 创建输出目录名
        createOutputDirectory();
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
        
        outputDirectory = baseName + "_output";
        
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
     * 读取CSV文件头部
     */
    bool readHeader() {
        std::ifstream file(inputFilename);
        if (!file.is_open()) {
            std::cerr << "无法打开文件: " << inputFilename << std::endl;
            return false;
        }
        
        if (std::getline(file, headerLine)) {
            std::cout << "读取到头部: " << headerLine.substr(0, 100) << "..." << std::endl;
            file.close();
            return true;
        }
        
        file.close();
        return false;
    }
    
    /**
     * 估算每行的平均大小
     */
    size_t estimateLineSize() {
        std::ifstream file(inputFilename);
        if (!file.is_open()) return 0;
        
        std::string line;
        size_t totalSize = 0;
        int lineCount = 0;
        const int sampleLines = 1000; // 采样前1000行来估算
        
        // 跳过头部
        std::getline(file, line);
        
        while (std::getline(file, line) && lineCount < sampleLines) {
            totalSize += line.length() + 1; // +1 for newline
            lineCount++;
        }
        
        file.close();
        
        if (lineCount == 0) return 0;
        return totalSize / lineCount;
    }
    
    /**
     * 生成输出文件名
     */
    std::string generateOutputFilename(size_t sizeInBytes) {
        size_t sizeInMB = sizeInBytes / (1024 * 1024);
        
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
        oss << outputDirectory << "/" << baseName << "_" << sizeInMB << "MB.csv";
        return oss.str();
    }
    
    /**
     * 分割CSV文件
     */
    bool splitCSV() {
        if (!readHeader()) {
            return false;
        }
        
        size_t avgLineSize = estimateLineSize();
        if (avgLineSize == 0) {
            std::cerr << "无法估算行大小" << std::endl;
            return false;
        }
        
        std::cout << "估算的平均行大小: " << avgLineSize << " 字节" << std::endl;
        
        // 为每个目标大小生成文件
        for (size_t targetSize : targetSizes) {
            if (!createSplitFile(targetSize, avgLineSize)) {
                std::cerr << "创建 " << targetSize / (1024 * 1024) << "MB 文件失败" << std::endl;
                continue;
            }
        }
        
        return true;
    }
    
private:
    /**
     * 创建指定大小的分割文件
     */
    bool createSplitFile(size_t targetSize, size_t avgLineSize) {
        std::ifstream inputFile(inputFilename);
        if (!inputFile.is_open()) {
            return false;
        }
        
        std::string outputFilename = generateOutputFilename(targetSize);
        std::ofstream outputFile(outputFilename);
        if (!outputFile.is_open()) {
            inputFile.close();
            return false;
        }
        
        // 写入头部
        outputFile << headerLine << "\n";
        size_t currentSize = headerLine.length() + 1;
        
        // 跳过输入文件的头部
        std::string line;
        std::getline(inputFile, line);
        
        size_t linesWritten = 0;
        
        // 读取并写入数据行，直到达到目标大小
        while (std::getline(inputFile, line) && currentSize < targetSize) {
            size_t lineSize = line.length() + 1; // +1 for newline
            
            // 检查是否会超过目标大小
            if (currentSize + lineSize > targetSize && linesWritten > 0) {
                break;
            }
            
            outputFile << line << "\n";
            currentSize += lineSize;
            linesWritten++;
            
            // 每10000行显示一次进度
            if (linesWritten % 10000 == 0) {
                std::cout << "正在处理 " << targetSize / (1024 * 1024) << "MB 文件... "
                          << "已写入 " << linesWritten << " 行, "
                          << "当前大小: " << std::fixed << std::setprecision(2) 
                          << (double)currentSize / (1024 * 1024) << " MB" << std::endl;
            }
        }
        
        inputFile.close();
        outputFile.close();
        
        std::cout << "✓ 创建文件: " << outputFilename 
                  << " (大小: " << std::fixed << std::setprecision(2) 
                  << (double)currentSize / (1024 * 1024) << " MB, "
                  << "行数: " << linesWritten + 1 << ")" << std::endl;
        
        return true;
    }
};

/**
 * 显示使用帮助
 */
void showUsage(const char* programName) {
    std::cout << "用法: " << programName << " <CSV文件路径>" << std::endl;
    std::cout << "示例: " << programName << " large_data.csv" << std::endl;
    std::cout << std::endl;
    std::cout << "程序将生成以下大小的文件:" << std::endl;
    std::cout << "30MB, 60MB, 90MB, 120MB, 150MB, 180MB, 210MB, 240MB, 270MB, 300MB" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
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
    
    std::cout << "开始处理文件: " << inputFile << std::endl;
    std::cout << "========================================" << std::endl;
    
    CSVSplitter splitter(inputFile);
    
    if (splitter.splitCSV()) {
        std::cout << "========================================" << std::endl;
        std::cout << "所有文件处理完成!" << std::endl;
    } else {
        std::cerr << "处理过程中出现错误!" << std::endl;
        return 1;
    }
    
    return 0;
}