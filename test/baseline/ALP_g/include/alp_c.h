#ifndef ALP_COMPRESSION_H
#define ALP_COMPRESSION_H

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <iostream>
#include <memory>
#include <limits>
#include <unordered_set>

// Forward declarations
class OutputBitStream;
class InputBitStream;
template<typename T> class ALPrdCompression;
template<typename T> class ALPrdDecompression;

// Constants
namespace ALPConstants {
    static int ALP_VECTOR_SIZE = 1000;
    static int RG_SAMPLES = 8;
    static short SAMPLES_PER_VECTOR = 32;
    static const uint8_t EXCEPTION_POSITION_SIZE = sizeof(short) * 8;
    static const uint8_t SAMPLING_EARLY_EXIT_THRESHOLD = 2;
    static const uint8_t MAX_COMBINATIONS = 5;
    
    // 采样配置参数 - 按照论文设置
    static const int ROWGROUP_SAMPLE_VECTORS = 8;    // m: 行组级采样向量数
    static const int VALUES_PER_VECTOR_SAMPLE = 32;  // n: 每向量采样值数
    static const int MAX_KEPT_COMBINATIONS = 5;      // k: 保留的最佳组合数
    static const int VECTOR_SAMPLE_VALUES = 32;      // s: 向量级采样值数
    static const int ROWGROUP_SIZE = 100;            // w: 行组大小(向量数)
    
    static const uint64_t FACT_ARR[] = {
        1, 10, 100, 1000, 10000, 100000, 1000000, 10000000,
        100000000, 1000000000, 10000000000ULL, 100000000000ULL, 1000000000000ULL,
        10000000000000ULL, 100000000000000ULL, 1000000000000000ULL,
        10000000000000000ULL, 100000000000000000ULL, 1000000000000000000ULL
    };
    
    static const uint64_t U_FACT_ARR[] = {
        1, 10, 100, 1000, 10000, 100000, 1000000, 10000000,
        100000000, 1000000000, 10000000000ULL, 100000000000ULL, 1000000000000ULL,
        10000000000000ULL, 100000000000000ULL, 1000000000000000ULL,
        10000000000000000ULL, 100000000000000000ULL, 1000000000000000000ULL
    };
    
    void selfAdaption(int vectorSize) {
        ALP_VECTOR_SIZE = vectorSize;
        SAMPLES_PER_VECTOR = static_cast<short>(std::max(32.0 * (vectorSize / 1000.0), 32.0));
    }
}

// 压缩模式枚举
enum class CompressionMode {
    ALP,     // 标准ALP用于decimal-like数据
    ALPrd    // ALPrd用于真正的双精度数据
};

// 保持原有的bit stream实现
class OutputBitStream {
private:
    std::vector<uint8_t> buffer;
    size_t bitPos;

public:
    OutputBitStream(size_t initialSize = 7000000) : buffer(initialSize), bitPos(0) {}
    
    size_t writeBit(bool bit) {
        size_t bytePos = bitPos / 8;
        size_t bitOffset = bitPos % 8;
        
        if (bytePos >= buffer.size()) {
            buffer.resize(bytePos + 1000);
        }
        
        if (bit) {
            buffer[bytePos] |= (1 << (7 - bitOffset));
        } else {
            buffer[bytePos] &= ~(1 << (7 - bitOffset));
        }
        
        bitPos++;
        return 1;
    }
    
    size_t writeInt(uint64_t value, int bits) {
        for (int i = bits - 1; i >= 0; i--) {
            writeBit((value >> i) & 1);
        }
        return bits;
    }
    
    size_t writeLong(uint64_t value, int bits) {
        return writeInt(value, bits);
    }
    
    const uint8_t* getBuffer() const { return buffer.data(); }
    size_t getBitPosition() const { return bitPos; }
    
    void flush() {}
    void close() {}
};

class InputBitStream {
private:
    const uint8_t* buffer;
    size_t bufferSize;
    size_t bitPos;

public:
    InputBitStream(const uint8_t* buf, size_t size) : buffer(buf), bufferSize(size), bitPos(0) {}
    
    int readBit() {
        if (bitPos / 8 >= bufferSize) return -1;
        
        size_t bytePos = bitPos / 8;
        size_t bitOffset = bitPos % 8;
        int bit = (buffer[bytePos] >> (7 - bitOffset)) & 1;
        bitPos++;
        return bit;
    }
    
    uint64_t readInt(int bits) {
        uint64_t value = 0;
        for (int i = 0; i < bits; i++) {
            int bit = readBit();
            if (bit == -1) break;
            value = (value << 1) | bit;
        }
        return value;
    }
    
    uint64_t readLong(int bits) {
        return readInt(bits);
    }
};

// 改进的ALP组合类
class ALPCombination {
public:
    uint8_t e;        // exponent
    uint8_t f;        // factor  
    uint64_t count;   // 出现次数
    double estimatedSize; // 估计的压缩大小
    
    ALPCombination(uint8_t exponent = 0, uint8_t factor = 0, uint64_t cnt = 0, double size = 0.0) 
        : e(exponent), f(factor), count(cnt), estimatedSize(size) {}
    
    bool operator<(const ALPCombination& other) const {
        if (count != other.count) return count > other.count;
        if (e != other.e) return e > other.e; // 优先高指数
        return f > other.f; // 优先高因子
    }
    
    bool operator==(const ALPCombination& other) const {
        return e == other.e && f == other.f;
    }
};

// Hash函数用于unordered_map
struct CombinationHash {
    size_t operator()(const ALPCombination& c) const {
        return std::hash<uint16_t>()(static_cast<uint16_t>(c.e) << 8 | c.f);
    }
};

// 两级采样器
template<typename T>
class TwoLevelSampler {
// private:
    // static constexpr uint8_t MAX_EXPONENT = std::is_same_v<T, double> ? 18 : 10;
    // static const std::vector<T> EXP_ARR;
    // static const std::vector<T> FRAC_ARR;

public:
    static constexpr uint8_t MAX_EXPONENT = std::is_same_v<T, double> ? 18 : 10;
    static const std::vector<T> EXP_ARR;
    static const std::vector<T> FRAC_ARR;
    // 第一级采样：行组级别
    static std::vector<ALPCombination> firstLevelSampling(const std::vector<std::vector<T>>& rowGroup) {
        std::unordered_map<ALPCombination, uint64_t, CombinationHash> combinationCounts;
        
        // 按等距采样向量
        size_t vectorStep = std::max(1UL, rowGroup.size() / ALPConstants::ROWGROUP_SAMPLE_VECTORS);
        
        for (size_t vecIdx = 0; vecIdx < rowGroup.size(); vecIdx += vectorStep) {
            if (vecIdx >= rowGroup.size()) break;
            
            const auto& vector = rowGroup[vecIdx];
            if (vector.empty()) continue;
            
            // 对当前向量进行采样
            std::vector<T> sampledValues = sampleVector(vector, ALPConstants::VALUES_PER_VECTOR_SAMPLE);
            
            // 找到此采样的最佳组合
            ALPCombination bestCombo = findBestCombinationForSample(sampledValues);
            if (bestCombo.count > 0) {
                combinationCounts[bestCombo]++;
            }
        }
        
        // 转换为向量并排序
        std::vector<ALPCombination> result;
        for (const auto& [combo, count] : combinationCounts) {
            ALPCombination c = combo;
            c.count = count;
            result.push_back(c);
        }
        
        std::sort(result.begin(), result.end());
        
        // 只保留前k个
        if (result.size() > ALPConstants::MAX_KEPT_COMBINATIONS) {
            result.resize(ALPConstants::MAX_KEPT_COMBINATIONS);
        }
        
        return result;
    }
    
    // 第二级采样：向量级别
    static ALPCombination secondLevelSampling(const std::vector<T>& vector, 
                                            const std::vector<ALPCombination>& candidates) {
        if (candidates.empty()) {
            return ALPCombination();
        }
        
        if (candidates.size() == 1) {
            return candidates[0]; // 无需第二级采样
        }
        
        // 采样当前向量
        std::vector<T> sampledValues = sampleVector(vector, ALPConstants::VECTOR_SAMPLE_VALUES);
        
        ALPCombination bestCombo;
        double bestSize = std::numeric_limits<double>::max();
        int worseCount = 0;
        
        // 在候选组合中搜索
        for (const auto& candidate : candidates) {
            double estimatedSize = evaluateCombination(sampledValues, candidate.e, candidate.f);
            
            if (estimatedSize < bestSize) {
                bestSize = estimatedSize;
                bestCombo = candidate;
                bestCombo.estimatedSize = bestSize;
                worseCount = 0;
            } else {
                worseCount++;
                if (worseCount >= ALPConstants::SAMPLING_EARLY_EXIT_THRESHOLD) {
                    break; // 早期退出
                }
            }
        }
        
        return bestCombo;
    }

// private:
    // 对向量进行等距采样
    static std::vector<T> sampleVector(const std::vector<T>& vector, int sampleCount) {
        if (vector.size() <= sampleCount) {
            return vector;
        }
        
        std::vector<T> samples;
        double step = static_cast<double>(vector.size()) / sampleCount;
        
        for (int i = 0; i < sampleCount; ++i) {
            size_t idx = static_cast<size_t>(i * step);
            if (idx < vector.size()) {
                samples.push_back(vector[idx]);
            }
        }
        
        return samples;
    }
// private:
    // 为采样数据找到最佳组合
    static ALPCombination findBestCombinationForSample(const std::vector<T>& sample) {
        ALPCombination best;
        double bestSize = std::numeric_limits<double>::max();
        
        // 搜索所有可能的e和f组合
        for (uint8_t e = 0; e <= MAX_EXPONENT; ++e) {
            for (uint8_t f = 0; f <= e; ++f) {
                double size = evaluateCombination(sample, e, f);
                if (size < bestSize) {
                    bestSize = size;
                    best = ALPCombination(e, f, 1, size);
                }
            }
        }
        
        return best;
    }
    
    // 评估组合的压缩效果
    static double evaluateCombination(const std::vector<T>& sample, uint8_t e, uint8_t f) {
        if (sample.empty()) return std::numeric_limits<double>::max();
        
        int exceptions = 0;
        int64_t maxEncoded = LLONG_MIN;
        int64_t minEncoded = LLONG_MAX;
        
        for (T value : sample) {
            // ALPenc过程
            T encodedValue = value * EXP_ARR[e] * FRAC_ARR[f];
            int64_t intValue = fastRound(encodedValue);
            
            // ALPdec验证
            T decodedValue = static_cast<T>(intValue) * ALPConstants::FACT_ARR[f] * FRAC_ARR[e];
            
            if (decodedValue == value) {
                maxEncoded = std::max(maxEncoded, intValue);
                minEncoded = std::min(minEncoded, intValue);
            } else {
                exceptions++;
            }
        }
        
        if (exceptions >= sample.size() * 0.5) {
            return std::numeric_limits<double>::max(); // 太多异常，不适用
        }
        
        // 估计压缩大小
        int64_t range = maxEncoded - minEncoded;
        if (range <= 0) range = 1;
        
        int bitsNeeded = static_cast<int>(std::ceil(std::log2(range + 1)));
        double baseSize = sample.size() * bitsNeeded;
        double exceptionSize = exceptions * (sizeof(T) * 8 + ALPConstants::EXCEPTION_POSITION_SIZE);
        
        return baseSize + exceptionSize;
    }
    
    // 快速舍入实现
    static int64_t fastRound(T value) {
        if constexpr (std::is_same_v<T, double>) {
            static constexpr T SWEET_SPOT = static_cast<T>(1ULL << 51) + static_cast<T>(1ULL << 52);
            return static_cast<int64_t>(value + SWEET_SPOT - SWEET_SPOT);
        } else {
            static constexpr T SWEET_SPOT = static_cast<T>(1U << 22) + static_cast<T>(1U << 23);
            return static_cast<int64_t>(value + SWEET_SPOT - SWEET_SPOT);
        }
    }
};

// 模板特化 - EXP_ARR 和 FRAC_ARR
template<>
const std::vector<double> TwoLevelSampler<double>::EXP_ARR = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
    100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0,
    10000000000000.0, 100000000000000.0, 1000000000000000.0,
    10000000000000000.0, 100000000000000000.0, 1000000000000000000.0
};

template<>
const std::vector<double> TwoLevelSampler<double>::FRAC_ARR = {
    1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
    0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.000000000001,
    0.0000000000001, 0.00000000000001, 0.000000000000001, 0.0000000000000001,
    0.00000000000000001, 0.000000000000000001, 0.0000000000000000001, 0.00000000000000000001
};

template<>
const std::vector<float> TwoLevelSampler<float>::EXP_ARR = {
    1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f, 100000.0f, 1000000.0f, 10000000.0f,
    100000000.0f, 1000000000.0f, 10000000000.0f
};

template<>
const std::vector<float> TwoLevelSampler<float>::FRAC_ARR = {
    1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f,
    0.00000001f, 0.000000001f, 0.0000000001f
};

// 模式决策器
template<typename T>
class ModeDecider {
    friend TwoLevelSampler<T>;
public:
    static CompressionMode decideMode(const std::vector<std::vector<T>>& rowGroup) {
        if (rowGroup.empty()) return CompressionMode::ALP;
        
        // 统计高精度值的比例
        int totalValues = 0;
        int highPrecisionValues = 0;
        
        for (const auto& vector : rowGroup) {
            for (T value : vector) {
                totalValues++;
                if (isHighPrecision(value)) {
                    highPrecisionValues++;
                }
            }
        }
        
        // 如果高精度值超过50%，使用ALPrd
        double highPrecisionRatio = static_cast<double>(highPrecisionValues) / totalValues;
        if (highPrecisionRatio > 0.5) {
            // printf("highPrecisionRatio:%.6f\n",highPrecisionRatio);
            return CompressionMode::ALPrd;
        }
        
        // 尝试ALP采样，如果异常率太高则切换到ALPrd
        auto candidates = TwoLevelSampler<T>::firstLevelSampling(rowGroup);
        if (candidates.empty()) {
            return CompressionMode::ALPrd;
        }
        
        // 检查最佳候选的异常率
        double avgExceptionRate = estimateExceptionRate(rowGroup, candidates[0]);
        // printf("avgExceptionRate:%.6f\n",avgExceptionRate);
        if (avgExceptionRate > 0.2) {
            // printf("avgExceptionRate:%.6f\n",avgExceptionRate);
            return CompressionMode::ALPrd;
        }
        
        return CompressionMode::ALP;
    }

// private:
    static bool isHighPrecision(T value) {
        // 简单的高精度检测：检查是否有超过15位有效数字
        if (value == 0) return false;
        
        std::string str = std::to_string(value);
        size_t pointPos = str.find('.');
        if (pointPos == std::string::npos) return false;
        
        // 计算小数点后的有效位数
        size_t decimalDigits = 0;
        for (size_t i = pointPos + 1; i < str.size(); ++i) {
            if (str[i] != '0' && str[i] != '\0') {
                decimalDigits = str.size() - pointPos - 1;
                break;
            }
        }
        
        return decimalDigits > 15; // 超过15位小数认为是高精度
    }
    
    static double estimateExceptionRate(const std::vector<std::vector<T>>& rowGroup, 
                                       const ALPCombination& combo) {
        int totalSamples = 0;
        int exceptions = 0;
        
        // 采样向量进行估计
        size_t step = std::min(20UL, rowGroup.size());
        for (size_t i = 0; i < rowGroup.size(); i += step) {
            auto sample = TwoLevelSampler<T>::sampleVector(rowGroup[i], 32);
            for (T value : sample) {
                totalSamples++;
                if (!canEncode(value, combo.e, combo.f)) {
                    exceptions++;
                }
            }
        }
        
        return totalSamples > 0 ? static_cast<double>(exceptions) / totalSamples : 1.0;
    }
    
    static bool canEncode(T value, uint8_t e, uint8_t f) {
        const auto& EXP_ARR = TwoLevelSampler<T>::EXP_ARR;
        const auto& FRAC_ARR = TwoLevelSampler<T>::FRAC_ARR;
        
        T encodedValue = value * EXP_ARR[e] * FRAC_ARR[f];
        int64_t intValue = TwoLevelSampler<T>::fastRound(encodedValue);
        T decodedValue = static_cast<T>(intValue) * ALPConstants::FACT_ARR[f] * FRAC_ARR[e];
        
        return decodedValue == value;
    }
};

// 改进的压缩状态类
template<typename T>
class ALPCompressionState {
public:
    uint8_t vectorExponent;
    uint8_t vectorFactor;
    short exceptionsCount;
    short bitWidth;
    int64_t frameOfReference;
    std::vector<int64_t> encodedIntegers;
    std::vector<T> exceptions;
    std::vector<short> exceptionsPositions;
    std::vector<ALPCombination> bestCombinations; // 最佳组合列表
    CompressionMode mode;
    
    ALPCompressionState(int vectorSize) 
        : vectorExponent(0), vectorFactor(0), exceptionsCount(0), bitWidth(0),
          frameOfReference(0), encodedIntegers(vectorSize), exceptions(vectorSize),
          exceptionsPositions(vectorSize), mode(CompressionMode::ALP) {}
    
    void reset() {
        vectorExponent = 0;
        vectorFactor = 0;
        exceptionsCount = 0;
        bitWidth = 0;
        frameOfReference = 0;
        bestCombinations.clear();
        mode = CompressionMode::ALP;
    }
};

// ALPrd相关类
namespace ALPrdConstants {
    static int ALP_VECTOR_SIZE = 1000;
    static const uint8_t DICTIONARY_BW = 3;
    static const uint8_t DICTIONARY_SIZE = 1 << DICTIONARY_BW;
    static const uint8_t CUTTING_LIMIT = 16;
    static const uint8_t EXCEPTION_SIZE = sizeof(short);
    static const uint8_t EXCEPTION_POSITION_SIZE = sizeof(short);
    
    void setVectorSize(int vectorSize) {
        ALP_VECTOR_SIZE = vectorSize;
    }
}

template<typename T>
class ALPrdCompressionState {
public:
    uint8_t rightBw;
    uint8_t leftBw;
    short exceptionsCount;
    std::vector<short> leftPartsDict;
    std::vector<short> exceptions;
    std::vector<short> exceptionsPositions;
    int leftBpSize;
    int rightBpSize;
    std::unordered_map<short, short> leftPartsDictMap;
    
    ALPrdCompressionState(int vectorSize)
        : rightBw(0), leftBw(0), exceptionsCount(0),
          leftPartsDict(ALPrdConstants::DICTIONARY_SIZE),
          exceptions(vectorSize), exceptionsPositions(vectorSize),
          leftBpSize(0), rightBpSize(0) {}
    
    void reset() {
        leftBpSize = 0;
        rightBpSize = 0;
        exceptionsCount = 0;
        leftPartsDictMap.clear();
    }
};

// ALPrd压缩实现
template<typename T>
class ALPrdCompression {
private:
    static constexpr uint8_t EXACT_TYPE_BITSIZE = sizeof(T) * 8;
    
    OutputBitStream* out;
    std::unique_ptr<ALPrdCompressionState<T>> state;
    size_t& size;

public:
    ALPrdCompression(OutputBitStream* outputStream, size_t& totalSize, int vectorSize) 
        : out(outputStream), size(totalSize) {
        ALPrdConstants::setVectorSize(vectorSize);
        state = std::make_unique<ALPrdCompressionState<T>>(vectorSize);
    }
    
    void reset() {
        state->reset();
    }
    
    size_t getSize() const { return 0; } // Size is managed by reference
    
    void entry(const std::vector<T>& row) {
        std::vector<uint64_t> rowBits;
        for (T value : row) {
            if constexpr (std::is_same_v<T, double>) {
                rowBits.push_back(*reinterpret_cast<const uint64_t*>(&value));
            } else {
                rowBits.push_back(*reinterpret_cast<const uint32_t*>(&value));
            }
        }
        findBestDictionary(rowBits, *state);
        compress(rowBits, rowBits.size(), *state);
    }

private:
    static double estimateCompressionSize(uint8_t rightBw, uint8_t leftBw, short exceptionsCount, size_t sampleCount) {
        double exceptionsSize = exceptionsCount * ((ALPrdConstants::EXCEPTION_POSITION_SIZE + ALPrdConstants::EXCEPTION_SIZE) * 8);
        return rightBw + leftBw + (exceptionsSize / sampleCount);
    }
    
    void findBestDictionary(const std::vector<uint64_t>& values, ALPrdCompressionState<T>& state) {
        int lBw = ALPrdConstants::DICTIONARY_BW;
        int rBw = EXACT_TYPE_BITSIZE;
        double bestDictSize = std::numeric_limits<double>::max();
        
        for (int i = 1; i <= ALPrdConstants::CUTTING_LIMIT; i++) {
            uint8_t candidateLBw = static_cast<uint8_t>(i);
            uint8_t candidateRBw = static_cast<uint8_t>(EXACT_TYPE_BITSIZE - i);
            double estimatedSize = buildLeftPartsDictionary(values, candidateRBw, candidateLBw, false, state);
            if (estimatedSize <= bestDictSize) {
                lBw = candidateLBw;
                rBw = candidateRBw;
                bestDictSize = estimatedSize;
            }
        }
        
        buildLeftPartsDictionary(values, static_cast<uint8_t>(rBw), static_cast<uint8_t>(lBw), true, state);
    }
    
    double buildLeftPartsDictionary(const std::vector<uint64_t>& values, uint8_t rightBw, uint8_t leftBw,
                                   bool persistDict, ALPrdCompressionState<T>& state) {
        std::unordered_map<uint64_t, int> leftPartsHash;
        std::vector<std::pair<int, uint64_t>> leftPartsSortedRepetitions;
        
        // Build hash for left parts
        for (uint64_t value : values) {
            uint64_t leftTmp = value >> rightBw;
            leftPartsHash[leftTmp]++;
        }
        
        // Convert to sorted vector
        for (const auto& [key, count] : leftPartsHash) {
            leftPartsSortedRepetitions.emplace_back(count, key);
        }
        std::sort(leftPartsSortedRepetitions.begin(), leftPartsSortedRepetitions.end(),
                 [](const auto& a, const auto& b) { return a.first > b.first; });
        
        // Count exceptions
        int exceptionsCount = 0;
        for (size_t i = ALPrdConstants::DICTIONARY_SIZE; i < leftPartsSortedRepetitions.size(); i++) {
            exceptionsCount += leftPartsSortedRepetitions[i].first;
        }
        
        if (persistDict) {
            int dictIdx = 0;
            int dictSize = std::min(static_cast<int>(ALPrdConstants::DICTIONARY_SIZE), static_cast<int>(leftPartsSortedRepetitions.size()));
            for (; dictIdx < dictSize; dictIdx++) {
                state.leftPartsDict[dictIdx] = static_cast<short>(leftPartsSortedRepetitions[dictIdx].second);
                state.leftPartsDictMap[state.leftPartsDict[dictIdx]] = static_cast<short>(dictIdx);
            }
            for (size_t i = dictIdx; i < leftPartsSortedRepetitions.size(); i++) {
                state.leftPartsDictMap[static_cast<short>(leftPartsSortedRepetitions[i].second)] = static_cast<short>(i);
            }
            state.leftBw = leftBw;
            state.rightBw = rightBw;
            state.exceptionsCount = static_cast<short>(exceptionsCount);
        }
        
        return estimateCompressionSize(rightBw, ALPrdConstants::DICTIONARY_BW, static_cast<short>(exceptionsCount), values.size());
    }
    
    void compress(const std::vector<uint64_t>& input, int nValues, ALPrdCompressionState<T>& state) {
        std::vector<uint64_t> rightParts(ALPrdConstants::ALP_VECTOR_SIZE);
        std::vector<short> leftParts(ALPrdConstants::ALP_VECTOR_SIZE);
        
        // Cut the values
        for (int i = 0; i < nValues; i++) {
            uint64_t tmp = input[i];
            rightParts[i] = tmp & ((1ULL << state.rightBw) - 1);
            leftParts[i] = static_cast<short>(tmp >> state.rightBw);
        }
        
        // Dictionary encoding
        short exceptionsCount = 0;
        for (int i = 0; i < nValues; i++) {
            short dictionaryIndex;
            short dictionaryKey = leftParts[i];
            
            auto it = state.leftPartsDictMap.find(dictionaryKey);
            if (it == state.leftPartsDictMap.end()) {
                dictionaryIndex = ALPrdConstants::DICTIONARY_SIZE;
            } else {
                dictionaryIndex = it->second;
            }
            leftParts[i] = dictionaryIndex;
            
            if (dictionaryIndex >= ALPrdConstants::DICTIONARY_SIZE) {
                leftParts[i] = 0;
                state.exceptions[exceptionsCount] = dictionaryKey;
                state.exceptionsPositions[exceptionsCount] = static_cast<short>(i);
                exceptionsCount++;
            }
        }
        
        // Write to output
        size += out->writeBit(false);
        size += out->writeInt(nValues, 32);
        size += out->writeInt(state.rightBw, 8);
        for (int i = 0; i < nValues; i++) {
            size += out->writeInt(leftParts[i], ALPrdConstants::DICTIONARY_BW);
            size += out->writeLong(rightParts[i], state.rightBw);
        }
        for (int i = 0; i < ALPrdConstants::DICTIONARY_SIZE; i++) {
            size += out->writeInt(state.leftPartsDict[i], state.leftBw);
        }
        size += out->writeInt(exceptionsCount, 16);
        for (int i = 0; i < exceptionsCount; i++) {
            size += out->writeInt(state.exceptions[i], state.leftBw);
            size += out->writeInt(state.exceptionsPositions[i], 16);
        }
    }
};

// ALPrd Decompression class template - moved before ALPDecompression
template<typename T>
class ALPrdDecompression {
private:
    static constexpr uint8_t EXACT_TYPE_BITSIZE = sizeof(T) * 8;
    
    InputBitStream* in;
    int nValues;
    uint8_t rightBW;
    std::vector<uint64_t> rightEncoded;
    std::vector<int> leftEncoded;
    std::vector<int> leftPartsDict;
    int exceptionsCount;
    std::vector<int> exceptions;
    std::vector<int> exceptionsPositions;

public:
    ALPrdDecompression(InputBitStream* inputStream) : in(inputStream) {}
    
    void deserialize() {
        nValues = static_cast<int>(in->readInt(32));
        rightBW = static_cast<uint8_t>(in->readInt(8));
        
        leftEncoded.resize(nValues);
        rightEncoded.resize(nValues);
        
        for (int i = 0; i < nValues; i++) {
            leftEncoded[i] = static_cast<int>(in->readInt(ALPrdConstants::DICTIONARY_BW));
            if constexpr (std::is_same_v<T, double>) {
                rightEncoded[i] = in->readLong(rightBW);
            } else {
                rightEncoded[i] = static_cast<uint64_t>(in->readInt(rightBW));
            }
        }
        
        int leftBW = EXACT_TYPE_BITSIZE - rightBW;
        leftPartsDict.resize(ALPrdConstants::DICTIONARY_SIZE);
        for (int i = 0; i < ALPrdConstants::DICTIONARY_SIZE; i++) {
            leftPartsDict[i] = static_cast<int>(in->readInt(leftBW));
        }
        
        exceptionsCount = static_cast<int>(in->readInt(16));
        exceptions.resize(exceptionsCount);
        exceptionsPositions.resize(exceptionsCount);
        
        for (int i = 0; i < exceptionsCount; i++) {
            exceptions[i] = static_cast<int>(in->readInt(leftBW));
            exceptionsPositions[i] = static_cast<int>(in->readInt(16));
        }
    }
    
    std::vector<T> decompress() {
        if constexpr (std::is_same_v<T, double>) {
            std::vector<uint64_t> outputLong(nValues);
            std::vector<T> output(nValues);
            
            // Decode by combining left and right parts
            for (int i = 0; i < nValues; i++) {
                uint64_t left = static_cast<uint64_t>(leftPartsDict[leftEncoded[i]]);
                uint64_t right = rightEncoded[i];
                outputLong[i] = (left << rightBW) | right;
            }
            
            // Patch exceptions
            for (int i = 0; i < exceptionsCount; i++) {
                uint64_t right = rightEncoded[exceptionsPositions[i]];
                uint64_t left = static_cast<uint64_t>(exceptions[i]);
                outputLong[exceptionsPositions[i]] = (left << rightBW) | right;
            }
            
            // Convert to double
            for (int i = 0; i < nValues; i++) {
                output[i] = *reinterpret_cast<const double*>(&outputLong[i]);
            }
            return output;
        } else {
            std::vector<uint32_t> outputLong(nValues);
            std::vector<T> output(nValues);
            
            // Decode by combining left and right parts
            for (int i = 0; i < nValues; i++) {
                uint32_t left = static_cast<uint32_t>(leftPartsDict[leftEncoded[i]]);
                uint32_t right = static_cast<uint32_t>(rightEncoded[i]);
                outputLong[i] = (left << rightBW) | right;
            }
            
            // Patch exceptions
            for (int i = 0; i < exceptionsCount; i++) {
                uint32_t right = static_cast<uint32_t>(rightEncoded[exceptionsPositions[i]]);
                uint32_t left = static_cast<uint32_t>(exceptions[i]);
                outputLong[exceptionsPositions[i]] = (left << rightBW) | right;
            }
            
            // Convert to float
            for (int i = 0; i < nValues; i++) {
                output[i] = *reinterpret_cast<const float*>(&outputLong[i]);
            }
            return output;
        }
    }
};

// 主要的ALP压缩类 - 改进版本
template<typename T>
class ALPCompression {
private:
    static constexpr T MAGIC_NUMBER = std::is_same_v<T, double> ? 
        (static_cast<T>(1ULL << 51) + static_cast<T>(1ULL << 52)) : 
        (static_cast<T>((1 << 22) + (1 << 23)));
    
    static constexpr uint8_t MAX_EXPONENT = std::is_same_v<T, double> ? 18 : 10;
    static constexpr uint8_t EXACT_TYPE_BIT_SIZE = sizeof(T) * 8;
    
    static const std::vector<T> EXP_ARR;
    static const std::vector<T> FRAC_ARR;
    
    std::unique_ptr<OutputBitStream> out;
    std::unique_ptr<ALPrdCompression<T>> aLPrd;
    std::unique_ptr<ALPCompressionState<T>> state;
    size_t size;
    
    // 行组级别的最佳组合缓存
    std::vector<ALPCombination> rowGroupCombinations;

public:
    ALPCompression(int vectorSize) : size(0) {
        // printf("ALP_c_comp\n");
        out = std::make_unique<OutputBitStream>();
        aLPrd = std::make_unique<ALPrdCompression<T>>(out.get(), size, vectorSize);
        ALPConstants::selfAdaption(vectorSize);
        state = std::make_unique<ALPCompressionState<T>>(vectorSize);
    }
    
    void reset() {
        state->reset();
        aLPrd->reset();
        size = 0;
        rowGroupCombinations.clear();
    }
    
    size_t getSize() const { return size; }
    
    std::vector<uint8_t> getOutput() const {
        size_t byteCount = (size + 7) / 8;
        std::vector<uint8_t> result(byteCount);
        std::memcpy(result.data(), out->getBuffer(), byteCount);
        return result;
    }
    
    // 主要的压缩入口点 - 改进版本
    void entry(const std::vector<std::vector<T>>& rowGroup) {
        if (rowGroup.empty()) return;
        
        // 第一步：决定压缩模式
        state->mode = ModeDecider<T>::decideMode(rowGroup);
        
        // 写入行组大小
        size += out->writeLong(rowGroup.size(), 8);
        
        if (state->mode == CompressionMode::ALPrd) {
            // 使用ALPrd模式
            compressWithALPrd(rowGroup);
        } else {
            // 使用标准ALP模式
            compressWithALP(rowGroup);
        }
    }

private:
    // 使用ALPrd模式压缩
    void compressWithALPrd(const std::vector<std::vector<T>>& rowGroup) {
        for (const auto& row : rowGroup) {
            aLPrd->entry(row);
        }
    }
    
    // 使用标准ALP模式压缩
    void compressWithALP(const std::vector<std::vector<T>>& rowGroup) {
        // 第一级采样：行组级别
        if (rowGroupCombinations.empty()) {
            rowGroupCombinations = TwoLevelSampler<T>::firstLevelSampling(rowGroup);
        }
        
        // 为每个向量进行压缩
        for (const auto& vector : rowGroup) {
            compressVector(vector);
        }
    }
    
    // 压缩单个向量
    void compressVector(const std::vector<T>& inputVector) {
        int nValues = static_cast<int>(inputVector.size());
        if (nValues == 0) return;
        
        // 第二级采样：向量级别
        ALPCombination bestCombo;
        if (rowGroupCombinations.size() == 1) {
            // 只有一个候选，直接使用
            bestCombo = rowGroupCombinations[0];
        } else {
            // 进行第二级采样
            bestCombo = TwoLevelSampler<T>::secondLevelSampling(inputVector, rowGroupCombinations);
        }
        
        state->vectorExponent = bestCombo.e;
        state->vectorFactor = bestCombo.f;
        
        // 执行压缩
        compress(inputVector, nValues, *state);
    }
    
    // 改进的压缩函数
    void compress(const std::vector<T>& inputVector, int nValues, ALPCompressionState<T>& state) {
        // 步骤1：编码为整数
        encodeToIntegers(inputVector, nValues, state);
        
        // 步骤2：检测和处理异常
        handleExceptions(inputVector, nValues, state);
        
        // 步骤3：应用Frame of Reference
        applyFrameOfReference(nValues, state);
        
        // 步骤4：写入压缩数据
        writeCompressedData(nValues, state);
    }
    
    // 编码为整数
    void encodeToIntegers(const std::vector<T>& inputVector, int nValues, ALPCompressionState<T>& state) {
        for (int i = 0; i < nValues; i++) {
            T db = inputVector[i];
            T tmpEncodedValue = db * EXP_ARR[state.vectorExponent] * FRAC_ARR[state.vectorFactor];
            int64_t encodedValue = convertToLong(tmpEncodedValue);
            state.encodedIntegers[i] = encodedValue;
        }
    }
    
    // 改进的异常处理
    void handleExceptions(const std::vector<T>& inputVector, int nValues, ALPCompressionState<T>& state) {
        std::vector<short> exceptionPositions;
        
        // 检测异常
        for (int i = 0; i < nValues; i++) {
            T originalValue = inputVector[i];
            int64_t encodedValue = state.encodedIntegers[i];
            
            // 解码验证
            T decodedValue = static_cast<T>(encodedValue) * ALPConstants::FACT_ARR[state.vectorFactor] * FRAC_ARR[state.vectorExponent];
            
            // 检查是否为异常
            bool isException = (decodedValue != originalValue) || 
                              (std::is_same_v<T, double> ? 
                               (doubleToRawLongBits(originalValue) == LLONG_MIN) :
                               (floatToRawIntBits(originalValue) == INT_MIN));
            
            if (isException) {
                exceptionPositions.push_back(static_cast<short>(i));
            }
        }
        
        if (exceptionPositions.empty()) {
            state.exceptionsCount = 0;
            return;
        }
        
        // 找到第一个非异常值
        int64_t fillValue = 0;
        bool foundNonException = false;
        for (int i = 0; i < nValues; i++) {
            bool isException = std::find(exceptionPositions.begin(), exceptionPositions.end(), i) != exceptionPositions.end();
            if (!isException) {
                fillValue = state.encodedIntegers[i];
                foundNonException = true;
                break;
            }
        }
        
        // 如果所有值都是异常，使用第一个编码值
        if (!foundNonException && !exceptionPositions.empty()) {
            fillValue = state.encodedIntegers[0];
        }
        
        // 处理异常
        state.exceptionsCount = static_cast<short>(exceptionPositions.size());
        for (size_t i = 0; i < exceptionPositions.size(); ++i) {
            short pos = exceptionPositions[i];
            state.encodedIntegers[pos] = fillValue; // 用非异常值填充
            state.exceptions[i] = inputVector[pos]; // 保存原始异常值
            state.exceptionsPositions[i] = pos;    // 保存异常位置
        }
    }
    
    // 应用Frame of Reference
    void applyFrameOfReference(int nValues, ALPCompressionState<T>& state) {
        if (nValues == 0) return;
        
        // 找到最小和最大值
        int64_t minValue = state.encodedIntegers[0];
        int64_t maxValue = state.encodedIntegers[0];
        
        for (int i = 1; i < nValues; i++) {
            int64_t value = state.encodedIntegers[i];
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
        
        // 计算范围和需要的位宽
        int64_t range = maxValue - minValue;
        int bitWidth = getWidthNeeded(range);
        
        // 应用FOR：减去最小值
        for (int i = 0; i < nValues; i++) {
            state.encodedIntegers[i] -= minValue;
        }
        
        state.frameOfReference = minValue;
        state.bitWidth = static_cast<short>(bitWidth);
    }
    
    // 写入压缩数据
    void writeCompressedData(int nValues, ALPCompressionState<T>& state) {
        // 写入标志位（表示使用ALP而非ALPrd）
        size += out->writeBit(true);
        
        // 写入向量头信息
        size += out->writeInt(state.vectorExponent, 8);
        size += out->writeInt(state.vectorFactor, 8);
        size += out->writeInt(state.bitWidth, 16);
        size += out->writeLong(state.frameOfReference, 64);
        size += out->writeInt(nValues, 32);
        
        // 写入编码后的整数
        for (int i = 0; i < nValues; i++) {
            size += out->writeLong(state.encodedIntegers[i], state.bitWidth);
        }
        
        // 写入异常信息
        size += out->writeInt(state.exceptionsCount, 16);
        for (int i = 0; i < state.exceptionsCount; i++) {
            if constexpr (std::is_same_v<T, double>) {
                size += out->writeLong(doubleToRawLongBits(state.exceptions[i]), 64);
            } else {
                size += out->writeInt(floatToRawIntBits(state.exceptions[i]), 32);
            }
            size += out->writeLong(state.exceptionsPositions[i], 16);
        }
    }
    
    // 辅助函数
    static int64_t convertToLong(T value) {
        T n = value + MAGIC_NUMBER - MAGIC_NUMBER;
        return static_cast<int64_t>(n);
    }
    
    static int getWidthNeeded(int64_t number) {
        if (number <= 0) return 1;
        int bitCount = 0;
        while (number > 0) {
            bitCount++;
            number >>= 1;
        }
        return bitCount;
    }
    
    static int64_t doubleToRawLongBits(double value) {
        return *reinterpret_cast<const int64_t*>(&value);
    }
    
    static int32_t floatToRawIntBits(float value) {
        return *reinterpret_cast<const int32_t*>(&value);
    }
};

// 模板特化（保持与之前相同）
template<>
const std::vector<double> ALPCompression<double>::EXP_ARR = {
    1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0,
    100000000.0, 1000000000.0, 10000000000.0, 100000000000.0, 1000000000000.0,
    10000000000000.0, 100000000000000.0, 1000000000000000.0,
    10000000000000000.0, 100000000000000000.0, 1000000000000000000.0
};

template<>
const std::vector<double> ALPCompression<double>::FRAC_ARR = {
    1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
    0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.000000000001,
    0.0000000000001, 0.00000000000001, 0.000000000000001, 0.0000000000000001,
    0.00000000000000001, 0.000000000000000001, 0.0000000000000000001, 0.00000000000000000001
};

template<>
const std::vector<float> ALPCompression<float>::EXP_ARR = {
    1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f, 100000.0f, 1000000.0f, 10000000.0f,
    100000000.0f, 1000000000.0f, 10000000000.0f
};

template<>
const std::vector<float> ALPCompression<float>::FRAC_ARR = {
    1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f,
    0.00000001f, 0.000000001f, 0.0000000001f
};

// 解压类（与之前类似，但需要处理新的格式）
template<typename T>
class ALPDecompression {
private:
    static const std::vector<T> FRAC_ARR;
    
    std::unique_ptr<InputBitStream> in;
    std::unique_ptr<ALPrdDecompression<T>> ALPrdDe;

public:
    ALPDecompression(const uint8_t* data, size_t dataSize) {
        in = std::make_unique<InputBitStream>(data, dataSize);
        ALPrdDe = std::make_unique<ALPrdDecompression<T>>(in.get());
    }
    
    std::vector<std::vector<T>> entry() {
        std::vector<std::vector<T>> result;
        int rowGroupSize = static_cast<int>(in->readInt(8));
        
        for (int i = 0; i < rowGroupSize; i++) {
            int useALP = in->readBit();
            if (useALP == 1) {
                // 使用ALP解压
                auto decompressed = decompressALP();
                result.push_back(std::move(decompressed));
            } else {
                // 使用ALPrd解压
                ALPrdDe->deserialize();
                auto decompressed = ALPrdDe->decompress();
                result.push_back(std::move(decompressed));
            }
        }
        return result;
    }

private:
    std::vector<T> decompressALP() {
        // 读取向量头信息
        uint8_t vectorExponent = static_cast<uint8_t>(in->readInt(8));
        uint8_t vectorFactor = static_cast<uint8_t>(in->readInt(8));
        short bitWidth = static_cast<short>(in->readInt(16));
        int64_t frameOfReference = static_cast<int64_t>(in->readLong(64));
        int count = static_cast<int>(in->readInt(32));
        
        // 读取编码的整数
        std::vector<int64_t> encodedValues(count);
        for (int i = 0; i < count; i++) {
            encodedValues[i] = static_cast<int64_t>(in->readLong(bitWidth));
        }
        
        // 读取异常信息
        short exceptionsCount = static_cast<short>(in->readInt(16));
        std::vector<T> exceptions(exceptionsCount);
        std::vector<short> exceptionsPositions(exceptionsCount);
        
        for (int i = 0; i < exceptionsCount; i++) {
            if constexpr (std::is_same_v<T, double>) {
                uint64_t bits = in->readLong(64);
                exceptions[i] = *reinterpret_cast<const double*>(&bits);
            } else {
                uint32_t bits = static_cast<uint32_t>(in->readInt(32));
                exceptions[i] = *reinterpret_cast<const float*>(&bits);
            }
            exceptionsPositions[i] = static_cast<short>(in->readLong(16));
        }
        
        // 解压过程
        std::vector<T> result(count);
        
        // 恢复Frame of Reference
        for (int i = 0; i < count; i++) {
            encodedValues[i] += frameOfReference;
        }
        
        // 解码
        uint64_t factor = ALPConstants::U_FACT_ARR[vectorFactor];
        T exponent = FRAC_ARR[vectorExponent];
        
        for (int i = 0; i < count; i++) {
            result[i] = static_cast<T>(encodedValues[i]) * factor * exponent;
        }
        
        // 恢复异常值
        for (int i = 0; i < exceptionsCount; i++) {
            result[exceptionsPositions[i]] = exceptions[i];
        }
        
        return result;
    }
};

// FRAC_ARR特化（与压缩类相同）
template<>
const std::vector<double> ALPDecompression<double>::FRAC_ARR = {
    1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001,
    0.00000001, 0.000000001, 0.0000000001, 0.00000000001, 0.000000000001,
    0.0000000000001, 0.00000000000001, 0.000000000000001, 0.0000000000000001,
    0.00000000000000001, 0.000000000000000001, 0.0000000000000000001, 0.00000000000000000001
};

template<>
const std::vector<float> ALPDecompression<float>::FRAC_ARR = {
    1.0f, 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f,
    0.00000001f, 0.000000001f, 0.0000000001f
};

// 便利工具类
class ALPUtil {
public:
    static std::vector<uint8_t> compressDouble(const std::vector<std::vector<double>>& data, int vectorSize = 1000) {
        ALPCompression<double> compressor(vectorSize);
        compressor.entry(data);
        return compressor.getOutput();
    }
    
    static std::vector<uint8_t> compressFloat(const std::vector<std::vector<float>>& data, int vectorSize = 1000) {
        ALPCompression<float> compressor(vectorSize);
        compressor.entry(data);
        return compressor.getOutput();
    }
    
    static std::vector<std::vector<double>> decompressDouble(const std::vector<uint8_t>& compressedData) {
        ALPDecompression<double> decompressor(compressedData.data(), compressedData.size());
        return decompressor.entry();
    }
    
    static std::vector<std::vector<float>> decompressFloat(const std::vector<uint8_t>& compressedData) {
        ALPDecompression<float> decompressor(compressedData.data(), compressedData.size());
        return decompressor.entry();
    }
};

using ALPCompressionDouble = ALPCompression<double>;
using ALPCompressionFloat = ALPCompression<float>;
using ALPDecompressionDouble = ALPDecompression<double>;
using ALPDecompressionFloat = ALPDecompression<float>;
using ALPrdCompressionDouble = ALPrdCompression<double>;
using ALPrdCompressionFloat = ALPrdCompression<float>;
using ALPrdDecompressionDouble = ALPrdDecompression<double>;
using ALPrdDecompressionFloat = ALPrdDecompression<float>;

#endif // ALP_COMPRESSION_H
