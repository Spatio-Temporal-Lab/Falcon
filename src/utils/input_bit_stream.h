#ifndef INPUT_BIT_STREAM_H
#define INPUT_BIT_STREAM_H

#include <vector>
#include <stdexcept>
#include <cstdint>

class InputBitStream {
public:
    explicit InputBitStream(const std::vector<uint8_t>& data);
    
    uint64_t Read(int numBits);
    
    long ReadLong(int numBits);

    

    bool isEnd() const;

//private:
    std::vector<uint8_t> data_; // 输入数据
    size_t cursor_; // 当前读取位置
    int bits_in_buffer_; // 当前缓冲区中的有效位数
    uint64_t buffer_; // 当前读取的缓冲区

    void FillBuffer();
};

#endif // INPUT_BIT_STREAM_H
