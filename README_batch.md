## 核心自适应函数

cpp

```cpp
int setChunk(int nbEle)
{
    size_t chunkSize=1024;
    size_t temp=nbEle/NUM_STREAMS;// (data+temp-1)/temp<NUm_streams
    while(chunkSize<=MAX_NUMS_PER_CHUNK 
            && chunkSize<=temp)
    {
        chunkSize*=2;
    }
    printf("chunkSize:%d\n",chunkSize);
    return chunkSize;
}
```

## 自适应策略分析

### 1. 基本原理

* **目标** : 确保流水线的并行度最大化，同时避免batch过大
* **约束条件** :
* `chunkSize <= MAX_NUMS_PER_CHUNK` (最大batch限制，目前设置为1024* 1024* 8，参考了一下nvcomp里面设置的限制大小为64MB)
  * `MAX_NUMS_PER_CHUNK`目前在尝试改成和GPU内存与流数量相关的一个变量
* `chunkSize <= temp` (确保有足够的并行度)

### 2. 计算逻辑

#### Step 1: 计算理想分批大小

cpp

```cpp
size_t temp = nbEle / NUM_STREAMS;
```

* 将总数据元素数除以流数量(16)
* 尽可能达到流数量个批

#### Step 2: 分批大小递增

```cpp
size_t chunkSize=1024;
```

确保批大小不会太小，并且和块大小（1024）适配，尽可能减少未达到1024个数据的块数量

因为一个批内的压缩是先按1024分块进行压缩，直到剩余数据不够1024（即最后一个块），

由于前面的批内的数据大小是1024的倍数，所以前面批进行压缩的时候每一个块都是1024个数据，

所以组合起来就和一次性压缩所有数据效果差不多。
