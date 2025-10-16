// #include "alp/encoder.hpp"
// #include "alp/rd.hpp"
// #include "alp_gpu.hpp"

// namespace alp_gpu {

// // 添加缺失的 kernel_write_rowgroup_headers 定义
// template<typename T>
// __global__ void kernel_write_rowgroup_headers(
//     const uint64_t* blk_sizes,
//     const uint64_t* blk_offsets,
//     int numBlocks,
//     int vectorSize,
//     uint8_t* out_bytes
// ) {
//     int blockId = blockIdx.x * blockDim.x + threadIdx.x;
//     if (blockId >= numBlocks) return;
    
//     int numVec = (blk_sizes[blockId] + vectorSize - 1) / vectorSize;
//     uint64_t bit_offset = blk_offsets[blockId];
    
//     SafeBitWriter bw(out_bytes, bit_offset, 8);
//     bw.putN((uint64_t)numVec, 8);
// }

// // 添加缺失的 alp_vector_analyze 函数定义
// template<typename T>
// __device__ inline void alp_vector_analyze(const T* v, int n, uint8_t e, uint8_t f,
//                                           short& bitw, long long& FOR,
//                                           int& exc_cnt) {
//     long long mn = LLONG_MAX, mx = LLONG_MIN;
//     exc_cnt = 0;
//     for(int i = 0; i < n; ++i) {
//         double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
//         long long I = fast_round_double(enc);
//         double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
//         if (dec == double(v[i])) { 
//             mn = (mn < I ? mn : I); 
//             mx = (mx > I ? mx : I); 
//         }
//         else ++exc_cnt;
//     }
//     unsigned long long range = (mn == LLONG_MAX) ? 0ULL : (unsigned long long)(mx - mn);
//     bitw = (short)width_needed_unsigned(range);
//     FOR = (mn == LLONG_MAX ? 0 : mn);
// }

// // 存储k个组合的结构体
// struct KCombinations {
//     uint8_t e[5];
//     uint8_t f[5];
//     uint8_t count;
//     uint8_t mode; // 0=ALP, 1=ALP-RD
// };

// // 修改后的采样kernel - 使用CPU提供的k组合
// template<typename T>
// __global__ void kernel_measure_with_cpu_combinations(
//     const T* data,
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes,
//     const KCombinations* k_combs,
//     int numBlocks,
//     int vectorSize,
//     uint64_t* out_bits
// ) {
//     int blockId = blockIdx.x;
//     if (blockId >= numBlocks) return;
    
//     const T* blk = data + blk_starts[blockId];
//     int n = (int)blk_sizes[blockId];
//     int numVec = (n + vectorSize - 1) / vectorSize;
//     KCombinations kcomb = k_combs[blockId];
    
//     __shared__ uint64_t sh_partial_sums[128];
//     sh_partial_sums[threadIdx.x] = 0;
    
//     if(kcomb.mode == 0) { // ALP模式
//         for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
//             int beg = v * vectorSize;
//             int len = min(vectorSize, n - beg);
            
//             uint64_t best_bits = UINT64_MAX;
            
//             if(kcomb.count == 1) {
//                 short bitw; long long FOR; int exc;
//                 alp_vector_analyze<T>(blk + beg, len, kcomb.e[0], kcomb.f[0], bitw, FOR, exc);
//                 best_bits = alp_vector_size_bits_safe<T>(len, kcomb.e[0], kcomb.f[0], bitw, exc);
//             } else {
//                 for(int ki = 0; ki < kcomb.count && ki < 5; ki++) {
//                     short bitw; long long FOR; int exc;
//                     alp_vector_analyze<T>(blk + beg, len, kcomb.e[ki], kcomb.f[ki], bitw, FOR, exc);
//                     uint64_t bits = alp_vector_size_bits_safe<T>(len, kcomb.e[ki], kcomb.f[ki], bitw, exc);
//                     if(bits < best_bits) {
//                         best_bits = bits;
//                     }
//                 }
//             }
            
//             sh_partial_sums[threadIdx.x] += best_bits;
//         }
//     } else { // ALP-RD模式
//         for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
//             int beg = v * vectorSize;
//             int len = min(vectorSize, n - beg);
            
//             uint64_t tmp[MAX_VEC];
//             for(int i = 0; i < len; i++) {
//                 if constexpr (std::is_same_v<T,double>) 
//                     tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
//                 else 
//                     tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
//             }
            
//             ALPrdDict<T> D;
//             alprd_find_best<T>(tmp, len, D);
            
//             int exc = 0;
//             for(int i = 0; i < len; i++) {
//                 uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
//                 bool inDict = false;
//                 for(int k = 0; k < DICT_SZ; k++) {
//                     if(D.dict[k] == left) { inDict = true; break; }
//                 }
//                 if(!inDict) exc++;
//             }
            
//             uint64_t bits = alprd_vector_size_bits_safe<T>(len, D, exc);
//             sh_partial_sums[threadIdx.x] += bits;
//         }
//     }
    
//     __syncthreads();
    
//     // 归约
//     for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if(threadIdx.x < stride) {
//             sh_partial_sums[threadIdx.x] += sh_partial_sums[threadIdx.x + stride];
//         }
//         __syncthreads();
//     }
    
//     if(threadIdx.x == 0) {
//         out_bits[blockId] = 8 + sh_partial_sums[0];
//     }
// }

// // 计算向量元数据的kernel - 使用CPU k组合
// template<typename T>
// __global__ void kernel_compute_metadata_with_combinations(
//     const T* data,
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes,
//     const KCombinations* k_combs,
//     int numBlocks,
//     int vectorSize,
//     uint64_t* vec_bit_sizes,
//     uint8_t* vec_e,
//     uint8_t* vec_f,
//     uint16_t* vec_bitw,
//     int64_t* vec_FOR,
//     uint32_t* vec_exc_cnt
// ) {
//     int blockId = blockIdx.x;
//     if (blockId >= numBlocks) return;
    
//     const T* blk = data + blk_starts[blockId];
//     int n = (int)blk_sizes[blockId];
//     int numVec = (n + vectorSize - 1) / vectorSize;
//     KCombinations kcomb = k_combs[blockId];
    
//     // 计算全局向量索引基址
//     uint64_t vec_base = 0;
//     for(int b = 0; b < blockId; b++) {
//         vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
//     }
    
//     // 每个线程并行处理不同的向量
//     for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
//         uint64_t global_vec_idx = vec_base + v;
//         int beg = v * vectorSize;
//         int len = min(vectorSize, n - beg);
        
//         if(kcomb.mode == 0) { // ALP模式
//             uint8_t best_e = 0;
//             uint8_t best_f = 0;
//             short bitw;
//             long long FOR;
//             int exc;
            
//             if(kcomb.count == 0) {
//                 // CPU没有提供有效的k组合，GPU自行计算
//                 alp_vector_choose_best_bits<T>(blk + beg, len, best_e, best_f, bitw, FOR, exc);
//             } else if(kcomb.count == 1) {
//                 // 只有一个组合，直接使用
//                 best_e = kcomb.e[0];
//                 best_f = kcomb.f[0];
//                 alp_vector_analyze<T>(blk + beg, len, best_e, best_f, bitw, FOR, exc);
//             } else {
//                 // 从k个组合中选择最佳
//                 uint64_t best_bits = UINT64_MAX;
                
//                 // 采样测试每个组合
//                 T samples[32];
//                 int sample_count = min(sampling_config::SAMPLES_PER_VECTOR, len);
//                 int sample_step = max(1, len / sample_count);
                
//                 for(int i = 0; i < sample_count; i++) {
//                     samples[i] = (blk + beg)[i * sample_step];
//                 }
                
//                 // 测试每个k组合
//                 for(int ki = 0; ki < kcomb.count && ki < 5; ki++) {
//                     short test_bitw;
//                     long long test_FOR;
//                     int test_exc;
//                     alp_vector_analyze<T>(samples, sample_count, 
//                                          kcomb.e[ki], kcomb.f[ki], 
//                                          test_bitw, test_FOR, test_exc);
                    
//                     uint64_t bits = alp_vector_size_bits_safe<T>(sample_count, 
//                                                                  kcomb.e[ki], kcomb.f[ki], 
//                                                                  test_bitw, test_exc);
//                     if(bits < best_bits) {
//                         best_bits = bits;
//                         best_e = kcomb.e[ki];
//                         best_f = kcomb.f[ki];
//                     }
//                 }
                
//                 // 使用选定的(e,f)分析完整向量
//                 alp_vector_analyze<T>(blk + beg, len, best_e, best_f, bitw, FOR, exc);
//             }
            
//             // 计算实际压缩大小
//             uint64_t bits = alp_vector_size_bits_safe<T>(len, best_e, best_f, bitw, exc);
            
//             vec_bit_sizes[global_vec_idx] = bits;
//             vec_e[global_vec_idx] = best_e;
//             vec_f[global_vec_idx] = best_f;
//             vec_bitw[global_vec_idx] = bitw;
//             vec_FOR[global_vec_idx] = FOR;
//             vec_exc_cnt[global_vec_idx] = exc;
            
//         } else { // ALP-RD模式
//             // 转换数据为整数格式
//             uint64_t tmp[MAX_VEC];
//             for(int i = 0; i < len; i++) {
//                 if constexpr (std::is_same_v<T,double>) 
//                     tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
//                 else 
//                     tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
//             }
            
//             // 计算最佳字典
//             ALPrdDict<T> D;
//             alprd_find_best<T>(tmp, len, D);
            
//             // 计算异常数量
//             int exc = 0;
//             for(int i = 0; i < len; i++) {
//                 uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
//                 bool inDict = false;
//                 for(int k = 0; k < DICT_SZ; k++) {
//                     if(D.dict[k] == left) {
//                         inDict = true;
//                         break;
//                     }
//                 }
//                 if(!inDict) exc++;
//             }
            
//             // 计算实际的ALP-RD压缩大小
//             uint64_t bits = alprd_vector_size_bits_safe<T>(len, D, exc);
            
//             vec_bit_sizes[global_vec_idx] = bits;
//             vec_e[global_vec_idx] = 0xFF;  // 标记为ALP-RD
//             vec_f[global_vec_idx] = 0xFF;
//             vec_bitw[global_vec_idx] = 0;
//             vec_FOR[global_vec_idx] = 0;
//             vec_exc_cnt[global_vec_idx] = exc;
//         }
//     }
// }

// // 写入kernel - 使用预计算的元数据
// template<typename T>
// __global__ void kernel_emit_with_metadata(
//     const T* data,
//     const uint64_t* blk_starts,
//     const uint64_t* blk_sizes,
//     const KCombinations* k_combs,
//     const uint64_t* vec_bit_offsets,
//     const uint8_t* vec_e,
//     const uint8_t* vec_f,
//     const uint16_t* vec_bitw,
//     const int64_t* vec_FOR,
//     int numBlocks,
//     int vectorSize,
//     uint8_t* out_bytes
// ) {
//     int blockId = blockIdx.x;
//     if (blockId >= numBlocks) return;
    
//     const T* blk = data + blk_starts[blockId];
//     int n = (int)blk_sizes[blockId];
//     int numVec = (n + vectorSize - 1) / vectorSize;
//     KCombinations kcomb = k_combs[blockId];
    
//     uint64_t vec_base = 0;
//     for(int b = 0; b < blockId; b++) {
//         vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
//     }
    
//     for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
//         uint64_t global_vec_idx = vec_base + v;
//         int beg = v * vectorSize;
//         int len = min(vectorSize, n - beg);
        
//         uint64_t bit_offset = vec_bit_offsets[global_vec_idx];
//         SafeBitWriter bw(out_bytes, bit_offset, 100000);
        
//         if(kcomb.mode == 0) { // ALP
//             uint8_t e = vec_e[global_vec_idx];
//             uint8_t f = vec_f[global_vec_idx];
//             short bitw = vec_bitw[global_vec_idx];
//             long long FOR = vec_FOR[global_vec_idx];
            
//             alp_vector_write_safe<T>(bw, blk + beg, len, e, f, bitw, FOR);
            
//         } else { // ALP-RD
//             uint64_t tmp[MAX_VEC];
//             for(int i = 0; i < len; i++) {
//                 if constexpr (std::is_same_v<T,double>) 
//                     tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
//                 else 
//                     tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
//             }
            
//             ALPrdDict<T> D;
//             alprd_find_best<T>(tmp, len, D);
//             alprd_vector_write_safe<T>(bw, tmp, len, D);
//         }
//     }
// }

// // ===================== 主压缩函数 =====================
// template<typename T>
// static Compressed compress_impl_optimized(const T* h_data, size_t n, const Params& p) {
//     Compressed c;
//     if (n == 0) { c.vectorSize = p.vectorSize; return c; }
    
//     const int V = p.vectorSize;
//     const int B = p.blockSize > 0 ? p.blockSize : int(n);
//     const int numBlocks = int((n + B - 1) / B);
    
//     // 计算总向量数
//     uint64_t total_vectors = 0;
//     std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
//     size_t pos = 0;
//     for(int i = 0; i < numBlocks; i++) {
//         h_starts[i] = pos;
//         uint64_t sz = std::min<uint64_t>(B, n - pos);
//         h_sizes[i] = sz;
//         total_vectors += (sz + V - 1) / V;
//         pos += sz;
//     }
    
//     // CPU阶段：使用官方预测器获取k组合
//     std::vector<KCombinations> h_k_combs(numBlocks);
    
//     for(int b = 0; b < numBlocks; b++) {
//         const T* blk_data = h_data + h_starts[b];
//         int blk_size = h_sizes[b];
        
//         alp::state<T> state;
//         state.vector_size = V;
//         std::vector<T> sample_arr(alp::config::ROWGROUP_SIZE);
        
//         // 调用官方预测器
//         alp::encoder<T>::init(
//             const_cast<T*>(blk_data), 
//             0, 
//             blk_size, 
//             sample_arr.data(), 
//             state
//         );
        
//         // 根据官方预测器的结果设置模式
//         if(state.scheme == alp::Scheme::ALP_RD) {
//             h_k_combs[b].mode = 1;  // ALP-RD模式
//             h_k_combs[b].count = 0;  // ALP-RD不需要k组合
//             // 初始化为安全值
//             for(int i = 0; i < 5; i++) {
//                 h_k_combs[b].e[i] = 0;
//                 h_k_combs[b].f[i] = 0;
//             }
//         } else {
//             h_k_combs[b].mode = 0;  // ALP模式
            
//             if(state.best_k_combinations.empty() || state.k_combinations == 0) {
//                 // 官方预测器没有给出有效的k组合，让GPU自行计算
//                 h_k_combs[b].count = 0;
//                 for(int i = 0; i < 5; i++) {
//                     h_k_combs[b].e[i] = 0;
//                     h_k_combs[b].f[i] = 0;
//                 }
//             } else {
//                 // 使用官方预测器提供的k组合
//                 h_k_combs[b].count = std::min<int>(state.k_combinations, 5);
//                 for(int i = 0; i < h_k_combs[b].count; i++) {
//                     h_k_combs[b].e[i] = state.best_k_combinations[i].first;
//                     h_k_combs[b].f[i] = state.best_k_combinations[i].second;
                    
//                     // 验证值的有效性
//                     if(h_k_combs[b].e[i] > 18) h_k_combs[b].e[i] = 18;
//                     if(h_k_combs[b].f[i] > h_k_combs[b].e[i]) {
//                         h_k_combs[b].f[i] = h_k_combs[b].e[i];
//                     }
//                 }
                
//                 // 填充剩余的组合槽位
//                 for(int i = h_k_combs[b].count; i < 5; i++) {
//                     h_k_combs[b].e[i] = 0;
//                     h_k_combs[b].f[i] = 0;
//                 }
//             }
//         }
//     }
    
//     // GPU阶段：并行压缩
//     T* d_data = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
//     CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
    
//     uint64_t *d_starts = nullptr, *d_sizes = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_starts, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_sizes, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_starts, h_starts.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_sizes, h_sizes.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
//     KCombinations* d_k_combs = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_k_combs, numBlocks * sizeof(KCombinations)));
//     CUDA_CHECK(cudaMemcpy(d_k_combs, h_k_combs.data(), numBlocks * sizeof(KCombinations), cudaMemcpyHostToDevice));
    
//     // 第一步：测量压缩大小（可选，用于调试）
//     uint64_t* d_bits = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_bits, numBlocks * sizeof(uint64_t)));
    
//     dim3 grid1(numBlocks);
//     dim3 block1(128);
    
//     kernel_measure_with_cpu_combinations<T><<<grid1, block1>>>(
//         d_data, d_starts, d_sizes, d_k_combs,
//         numBlocks, V, d_bits
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());
    
//     // 第二步：计算每个向量的元数据
//     uint64_t* d_vec_bit_sizes = nullptr;
//     uint8_t* d_vec_e = nullptr;
//     uint8_t* d_vec_f = nullptr;
//     uint16_t* d_vec_bitw = nullptr;
//     int64_t* d_vec_FOR = nullptr;
//     uint32_t* d_vec_exc_cnt = nullptr;
    
//     CUDA_CHECK(cudaMalloc(&d_vec_bit_sizes, total_vectors * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_vec_e, total_vectors * sizeof(uint8_t)));
//     CUDA_CHECK(cudaMalloc(&d_vec_f, total_vectors * sizeof(uint8_t)));
//     CUDA_CHECK(cudaMalloc(&d_vec_bitw, total_vectors * sizeof(uint16_t)));
//     CUDA_CHECK(cudaMalloc(&d_vec_FOR, total_vectors * sizeof(int64_t)));
//     CUDA_CHECK(cudaMalloc(&d_vec_exc_cnt, total_vectors * sizeof(uint32_t)));
    
//     kernel_compute_metadata_with_combinations<T><<<grid1, block1>>>(
//         d_data, d_starts, d_sizes, d_k_combs,
//         numBlocks, V,
//         d_vec_bit_sizes, d_vec_e, d_vec_f,
//         d_vec_bitw, d_vec_FOR, d_vec_exc_cnt
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());
    
//     // 第三步：获取实际计算的位大小（不使用估算）
//     std::vector<uint64_t> h_vec_bit_sizes(total_vectors);
//     CUDA_CHECK(cudaMemcpy(h_vec_bit_sizes.data(), d_vec_bit_sizes, 
//                           total_vectors * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    
//     // 第四步：计算位偏移
//     std::vector<uint64_t> h_block_offsets(numBlocks);
//     std::vector<uint64_t> h_vec_offsets(total_vectors);
    
//     uint64_t block_offset = 0;
//     uint64_t vec_idx = 0;
    
//     for(int b = 0; b < numBlocks; b++) {
//         h_block_offsets[b] = block_offset;
//         uint64_t current_offset = block_offset + 8; // 行组头
        
//         int numVec = (h_sizes[b] + V - 1) / V;
//         for(int v = 0; v < numVec; v++) {
//             h_vec_offsets[vec_idx] = current_offset;
//             current_offset += h_vec_bit_sizes[vec_idx];  // 使用GPU计算的实际大小
//             vec_idx++;
//         }
        
//         // 对齐到32位边界
//         uint64_t block_bits = current_offset - block_offset;
//         uint64_t padding = (32 - (block_bits & 31)) & 31;
//         block_offset = current_offset + padding;
//     }
    
//     const uint64_t total_bits = block_offset;
//     const uint64_t total_bytes = (total_bits + 7) / 8;
    
//     uint64_t* d_vec_offsets = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_vec_offsets, total_vectors * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_vec_offsets, h_vec_offsets.data(), 
//                          total_vectors * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
//     // 第五步：写入压缩数据
//     uint8_t* d_out = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_out, total_bytes));
//     CUDA_CHECK(cudaMemset(d_out, 0, total_bytes));
    
//     // 写入行组头
//     uint64_t* d_block_offsets = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets.data(), 
//                          numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
    
//     kernel_write_rowgroup_headers<T><<<(numBlocks+255)/256, 256>>>(
//         d_sizes, d_block_offsets, numBlocks, V, d_out
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());
    
//     // 写入向量数据
//     kernel_emit_with_metadata<T><<<grid1, block1>>>(
//         d_data, d_starts, d_sizes, d_k_combs,
//         d_vec_offsets, d_vec_e, d_vec_f,
//         d_vec_bitw, d_vec_FOR,
//         numBlocks, V, d_out
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());
    
//     // 拷回结果
//     c.data.resize(total_bytes);
//     CUDA_CHECK(cudaMemcpy(c.data.data(), d_out, total_bytes, cudaMemcpyDeviceToHost));
    
//     // 填充元数据
//     c.offsets = std::move(h_block_offsets);
//     c.bit_sizes.resize(numBlocks);
//     vec_idx = 0;
//     for(int b = 0; b < numBlocks; b++) {
//         uint64_t block_bits = 8;  // 行组头
//         int numVec = (h_sizes[b] + V - 1) / V;
//         for(int v = 0; v < numVec; v++) {
//             block_bits += h_vec_bit_sizes[vec_idx++];
//         }
//         c.bit_sizes[b] = block_bits;
//     }
//     c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
//     c.vectorSize = V;
    
//     // 清理GPU内存
//     CUDA_CHECK(cudaFree(d_block_offsets));
//     CUDA_CHECK(cudaFree(d_vec_offsets));
//     CUDA_CHECK(cudaFree(d_out));
//     CUDA_CHECK(cudaFree(d_vec_exc_cnt));
//     CUDA_CHECK(cudaFree(d_vec_FOR));
//     CUDA_CHECK(cudaFree(d_vec_bitw));
//     CUDA_CHECK(cudaFree(d_vec_f));
//     CUDA_CHECK(cudaFree(d_vec_e));
//     CUDA_CHECK(cudaFree(d_vec_bit_sizes));
//     CUDA_CHECK(cudaFree(d_bits));
//     CUDA_CHECK(cudaFree(d_k_combs));
//     CUDA_CHECK(cudaFree(d_sizes));
//     CUDA_CHECK(cudaFree(d_starts));
//     CUDA_CHECK(cudaFree(d_data));
    
//     return c;
// }
// // API接口
// Compressed compress_double_hybrid(const double* data, size_t n, const Params& p) {
//     return compress_impl_optimized<double>(data, n, p);
// }

// Compressed compress_float_hybrid(const float* data, size_t n, const Params& p) {
//     return compress_impl_optimized<float>(data, n, p);
// }

// template<typename T>
// __global__ void kernel_decompress_hybrid(const uint8_t* bytes,
//                                          const uint64_t* blk_starts_bits,
//                                          const uint64_t* blk_bits,
//                                          const uint64_t* out_starts,
//                                          const int vectorSize,
//                                          T* out_data, 
//                                          int numBlocks) {
//     int blockId = blockIdx.x;
//     if (blockId >= numBlocks) return;

//     uint64_t bit_offset = blk_starts_bits[blockId];
//     BitReader br{bytes, bit_offset};
    
//     int numVec = (int)br.getN(8);
    
//     if (numVec <= 0 || numVec > 10000) {
//         if (blockId == 0) {
//             printf("[DEVICE-ERROR] Invalid numVec: %d\n", numVec);
//         }
//         return;
//     }

//     uint64_t out_pos = out_starts[blockId];
    
//     for(int v = 0; v < numVec; v++) {
//         int useALP = br.get1();

//         if (useALP) {
//             uint8_t e = (uint8_t)br.getN(8);
//             uint8_t f = (uint8_t)br.getN(8);
//             short bitw = (short)br.getN(16);
//             long long FOR = (long long)br.getN(64);
//             int n = (int)br.getN(32);
            
//             if (n <= 0 || n > MAX_VEC) return;

//             for(int k = 0; k < n; k++) {
//                 uint64_t enc = br.getN(bitw);
//                 long long I = FOR + (long long)enc;
//                 double dec = double(I) * (1.0/double(D_FRAC_ARR[f])) * D_FRAC_ARR[e];
//                 out_data[out_pos + k] = (T)dec;
//             }

//             int exc = (int)br.getN(16);
//             for(int t = 0; t < exc; t++) {
//                 uint64_t raw = std::is_same_v<T,double> ? br.getN(64) : br.getN(32);
//                 int pos = (int)br.getN(16);
                
//                 if (pos < n) {
//                     if constexpr (std::is_same_v<T,double>) {
//                         double val = *reinterpret_cast<double*>(&raw);
//                         out_data[out_pos + pos] = (T)val;
//                     } else {
//                         uint32_t rv = (uint32_t)raw;
//                         float val = *reinterpret_cast<float*>(&rv);
//                         out_data[out_pos + pos] = (T)val;
//                     }
//                 }
//             }
//             out_pos += n;
//         } else {
//             int n = (int)br.getN(32);
//             if (n <= 0 || n > MAX_VEC) return;

//             uint8_t rbw = (uint8_t)br.getN(8);
//             uint64_t right[MAX_VEC]; 
//             uint16_t leftIdx[MAX_VEC];
            
//             for(int k = 0; k < n; k++) {
//                 leftIdx[k] = (uint16_t)br.getN(DICT_BW);
//                 right[k] = br.getN(rbw);
//             }

//             uint8_t lbw = uint8_t(sizeof(T)*8 - rbw);
//             uint64_t dict[DICT_SZ];
//             for(int k = 0; k < DICT_SZ; k++) {
//                 dict[k] = br.getN(lbw);
//             }

//             int exc = (int)br.getN(16);
//             uint16_t exc_pos[MAX_VEC]; 
//             uint64_t exc_left[MAX_VEC];
//             for(int t = 0; t < exc; t++) {
//                 exc_left[t] = br.getN(lbw);
//                 exc_pos[t] = (uint16_t)br.getN(16);
//             }

//             for(int k = 0; k < n; k++) {
//                 uint64_t left = (leftIdx[k] < DICT_SZ) ? dict[leftIdx[k]] : 0ULL;
//                 uint64_t raw = (left << rbw) | right[k];
                
//                 if constexpr (std::is_same_v<T,double>) {
//                     double val = *reinterpret_cast<double*>(&raw);
//                     out_data[out_pos + k] = (T)val;
//                 } else {
//                     uint32_t r32 = (uint32_t)raw;
//                     float val = *reinterpret_cast<float*>(&r32);
//                     out_data[out_pos + k] = (T)val;
//                 }
//             }

//             for(int t = 0; t < exc; t++) {
//                 int p = exc_pos[t];
//                 uint64_t raw = (exc_left[t] << rbw) | right[p];
                
//                 if (p < n) {
//                     if constexpr (std::is_same_v<T,double>) {
//                         double val = *reinterpret_cast<double*>(&raw);
//                         out_data[out_pos + p] = (T)val;
//                     } else {
//                         uint32_t r32 = (uint32_t)raw;
//                         float val = *reinterpret_cast<float*>(&r32);
//                         out_data[out_pos + p] = (T)val;
//                     }
//                 }
//             }
//             out_pos += n;
//         }
//     }
// }

// template<typename T>
// static void decompress_impl_hybrid(const Compressed& c, T* h_out, size_t n, const Params& p) {
//     if (n == 0) return;
//     const int numBlocks = (int)c.offsets.size();
//     assert((size_t)numBlocks == c.elem_counts.size());

//     if (c.data.empty()) {
//         std::cerr << "[ERROR] Compressed data is empty!" << std::endl;
//         return;
//     }

//     std::fill(h_out, h_out + n, T(-999.0));

//     uint8_t* d_bytes = nullptr; 
//     CUDA_CHECK(cudaMalloc(&d_bytes, c.data.size()));
//     CUDA_CHECK(cudaMemcpy(d_bytes, c.data.data(), c.data.size(), cudaMemcpyHostToDevice));

//     uint64_t *d_boff = nullptr, *d_bsiz = nullptr, *d_ost = nullptr;
//     CUDA_CHECK(cudaMalloc(&d_boff, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMalloc(&d_bsiz, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_boff, c.offsets.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));
//     CUDA_CHECK(cudaMemcpy(d_bsiz, c.bit_sizes.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

//     std::vector<uint64_t> h_outStarts(numBlocks);
//     uint64_t acc = 0; 
//     for(int i = 0; i < numBlocks; i++) { 
//         h_outStarts[i] = acc; 
//         acc += c.elem_counts[i]; 
//     }
    
//     if (acc != n) {
//         std::cerr << "[ERROR] elem_counts sum(" << acc << ") != output elements(" << n << ")" << std::endl;
//         return;
//     }

//     CUDA_CHECK(cudaMalloc(&d_ost, numBlocks * sizeof(uint64_t)));
//     CUDA_CHECK(cudaMemcpy(d_ost, h_outStarts.data(), numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice));

//     T* d_out = nullptr; 
//     CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(T)));
//     CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(T)));

//     // Just call the decompression kernel directly - it's already linked from alp_gpu.cu
//     // Use the local kernel
//     dim3 gs(numBlocks), bs(1);
//     kernel_decompress_hybrid<T><<<gs, bs>>>(
//         d_bytes, d_boff, d_bsiz, d_ost, p.vectorSize, d_out, numBlocks
//     );
//     CUDA_CHECK(cudaDeviceSynchronize());

//     CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(T), cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(d_out));
//     CUDA_CHECK(cudaFree(d_ost));
//     CUDA_CHECK(cudaFree(d_bsiz));
//     CUDA_CHECK(cudaFree(d_boff));
//     CUDA_CHECK(cudaFree(d_bytes));
// }

// void decompress_double_hybrid(const Compressed& c, double* out, size_t n, const Params& p) {
//     decompress_impl_hybrid<double>(c, out, n, p);
// }

// void decompress_float_hybrid(const Compressed& c, float* out, size_t n, const Params& p) {
//     decompress_impl_hybrid<float>(c, out, n, p);
// }

// } // namespace alp_gpu