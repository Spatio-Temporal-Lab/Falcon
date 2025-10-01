/*
    //(1.0/double(D_FRAC_ARR[f]))
    #include "alp_gpu.hpp"

    using std::uint8_t; using std::uint32_t; using std::uint64_t;

    namespace alp_gpu {

    // 设备兼容的键值对结构，替代std::pair
    struct EFPair {
        uint8_t exponent, factor;
        __device__ EFPair() : exponent(0), factor(0) {}
        __device__ EFPair(uint8_t e, uint8_t f) : exponent(e), factor(f) {}
        __device__ bool equals(const EFPair& other) const {
            return exponent == other.exponent && factor == other.factor;
        }
    };

    // 修正后的设备端ALP精确匹配检查
    template<typename T>
    __device__ inline bool alp_exact_equal(T v, uint8_t e, uint8_t f)
    {
        if constexpr (std::is_same_v<T,double>) {
            double enc = v * D_EXP_ARR[e] * D_FRAC_ARR[f];
            long long I = fast_round_double(enc);
            double dec = double(I) *  D_FACT_ARR[f] * D_FRAC_ARR[e];
            return dec==v;
        } else {
            float enc = v * float(D_EXP_ARR[e]) * float(D_FRAC_ARR[f]);
            int   I   = __float2int_rn(enc);
            float dec = float(I) * (D_FACT_ARR[f]) * float(D_FRAC_ARR[e]);
            return dec==v;
        }
    }

    template<typename T>
    __device__ inline void alp_vector_analyze(const T* v, int n, uint8_t e, uint8_t f,
                                            short& bitw, long long& FOR,
                                            int& exc_cnt){
        long long mn=LLONG_MAX, mx=LLONG_MIN;
        exc_cnt=0;
        for(int i=0;i<n;++i){
            double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
            long long I = fast_round_double(enc);
            double dec = double(I) * D_FACT_ARR[f] * D_FRAC_ARR[e];
            if (dec==double(v[i])) { 
                mn=(mn<I?mn:I); 
                mx=(mx>I?mx:I); 
            } else {
                ++exc_cnt;
            }
        }
        unsigned long long range = (mn==LLONG_MAX)? 0ULL : (unsigned long long)(mx - mn);
        bitw = (short)width_needed_unsigned(range);
        FOR  = (mn==LLONG_MAX?0:mn);
    }

    // 添加缺失的函数定义
    template<typename T>
    __device__ inline void alp_vector_choose_best_bits(
        const T* v, int n,
        uint8_t& best_e, uint8_t& best_f,
        short& bitw, long long& FOR, int& exc)
    {
        const int val_bits = std::is_same_v<T,double> ? 64 : 32;
        double best_score = 1e300;
        best_e=0; best_f=0; bitw=0; FOR=0; exc=0;

        for(int8_t e_idx = Constants<T>::MAX_EXPONENT; e_idx >= 0; --e_idx){
            for(int8_t f_idx = e_idx; f_idx >= 0; --f_idx){
                short _bw; long long _FOR; int _exc;
                alp_vector_analyze<T>(v, n, e_idx, f_idx, _bw, _FOR, _exc);
                
                double score = double(n)*_bw + double(_exc)*(val_bits + 16);
                
                if (score < best_score){
                    best_score = score;
                    best_e = e_idx; 
                    best_f = f_idx; 
                    bitw = _bw; 
                    FOR = _FOR; 
                    exc = _exc;
                }
            }
        }
    }

    template<typename T>
    __device__ inline uint64_t alp_vector_size_bits_safe(int n, uint8_t e, uint8_t f,
                                                        short bitw, int exc_cnt){
        const int val_bits = std::is_same_v<T,double> ? 64 : 32;
        return 145ULL + uint64_t(n) * uint64_t(bitw) + uint64_t(exc_cnt) * (val_bits + 16);
    }

    template<typename T>
    __device__ inline bool alp_vector_write_safe(SafeBitWriter& bw, const T* v, int n,
                                                uint8_t e, uint8_t f, short bitw, long long FOR){
        assert(n <= MAX_VEC);

        if (!bw.put1(1)) return false; // useALP = 1
        if (!bw.putN((uint64_t)e, 8)) return false;
        if (!bw.putN((uint64_t)f, 8)) return false;
        if (!bw.putN((uint64_t)bitw, 16)) return false;
        if (!bw.putN((uint64_t)FOR, 64)) return false;
        if (!bw.putN((uint64_t)n, 32)) return false;

        int exc_cnt=0;
        int      exc_pos[MAX_VEC];
        uint64_t exc_val[MAX_VEC];

        for(int i=0;i<n;++i){
            double enc = double(v[i]) * D_EXP_ARR[e] * D_FRAC_ARR[f];
            long long I = fast_round_double(enc);
            double dec = double(I) * D_FACT_ARR[f] * D_FRAC_ARR[e];
            if (dec==double(v[i])) {
                uint64_t packed = (uint64_t)(I - FOR);
                if (!bw.putN(packed, bitw)) return false;
            } else {
                if (!bw.putN(0, bitw)) return false;
                if constexpr (std::is_same_v<T,double>) {
                    uint64_t raw = *reinterpret_cast<const uint64_t*>(&v[i]);
                    exc_val[exc_cnt] = raw;
                } else {
                    uint32_t raw = *reinterpret_cast<const uint32_t*>(&v[i]);
                    exc_val[exc_cnt] = raw;
                }
                exc_pos[exc_cnt] = i;
                ++exc_cnt;
            }
        }
        if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
        for(int k=0;k<exc_cnt;++k){
            if constexpr (std::is_same_v<T,double>) {
                if (!bw.putN(exc_val[k], 64)) return false;
            } else {
                if (!bw.putN(exc_val[k], 32)) return false;
            }
            if (!bw.putN((uint64_t)exc_pos[k], 16)) return false;
        }
        return true;
    }

        template<typename T>
        __device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D){
            double best_score = 1e100; 
            int best_rbw = int(sizeof(T)*8) - 1;
            uint32_t best_dict[DICT_SZ] = {0};

            for(int lbw=1; lbw<=CUT_LIM; ++lbw){
                int rbw = int(sizeof(T)*8) - lbw;
                uint32_t lmask = mask_lo(lbw);

                uint32_t uniq_left[MAX_VEC]; 
                int cnt[MAX_VEC];
                int u = 0;

                for(int i=0;i<n;++i){
                    uint32_t left = (uint32_t)((in[i] >> rbw) & lmask);
                    int j=0; 
                    for(; j<u; ++j) {
                        if (uniq_left[j]==left) { 
                            ++cnt[j]; 
                            break; 
                        }
                    }
                    if (j==u){ 
                        uniq_left[u]=left; 
                        cnt[u]=1; 
                        ++u; 
                    }
                }
                
                uint32_t dict[DICT_SZ]={0};
                int used = (DICT_SZ < u ? DICT_SZ : u);
                for(int k=0;k<used;++k){
                    int best_cnt=-1, best_id=-1;
                    for(int j=0;j<u;++j){
                        bool taken=false;
                        for(int t=0;t<k;++t) {
                            if (dict[t]==uniq_left[j]) { 
                                taken=true; 
                                break; 
                            }
                        }
                        if (taken) continue;
                        if (cnt[j]>best_cnt){ 
                            best_cnt=cnt[j]; 
                            best_id=j; 
                        }
                    }
                    if (best_id >= 0) {
                        dict[k] = uniq_left[best_id];
                    }
                }
                
                int keep=0;
                for(int k=0;k<used;++k){
                    for(int j=0;j<u;++j) {
                        if (uniq_left[j]==dict[k]) { 
                            keep += cnt[j]; 
                            break; 
                        }
                    }
                }
                int exc = n - keep;

                double bits = 1 + 32 + 8 + double(n)*(DICT_BW + rbw) + 
                            double(DICT_SZ)*lbw + 16.0*exc + double(lbw)*exc;

                if (bits < best_score){
                    best_score = bits;
                    best_rbw   = rbw;
                    for(int k=0;k<DICT_SZ;++k) best_dict[k]=dict[k];
                }
            }
            D.rightBW = (uint8_t)best_rbw;
            D.leftBW  = (uint8_t)(int(sizeof(T)*8) - best_rbw);
            for(int k=0;k<DICT_SZ;++k) D.dict[k]=best_dict[k];
        }

    template<typename T>
    __device__ inline uint64_t alprd_vector_size_bits_safe(int n, const ALPrdDict<T>& D, int exc_cnt){
        uint64_t base = 57ULL + uint64_t(n)*(DICT_BW + D.rightBW) + 
                        DICT_SZ*D.leftBW + uint64_t(exc_cnt)*(D.leftBW+16);
        return base;
    }

    template<typename T>
    __device__ inline bool alprd_vector_write_safe(SafeBitWriter& bw, const uint64_t* in, int n,
                                                const ALPrdDict<T>& D){
        assert(n <= MAX_VEC);
        
        if (!bw.put1(0)) return false; // useALP=0
        if (!bw.putN((uint64_t)n, 32)) return false;
        if (!bw.putN((uint64_t)D.rightBW, 8)) return false;

        int exc_cnt=0; 
        uint16_t exc_pos[MAX_VEC]; 
        uint32_t exc_left[MAX_VEC];
        uint64_t right_mask = (D.rightBW==64)? ~0ULL : ((1ULL<<D.rightBW)-1ULL);
        uint32_t left_mask  = mask_lo(D.leftBW);

        for(int i=0;i<n;++i){
            uint64_t right = in[i] & right_mask;
            uint32_t left  = (uint32_t)((in[i] >> D.rightBW) & left_mask);
            short idx = DICT_SZ;
            for(int k=0;k<DICT_SZ;++k){ 
                if (D.dict[k]==left){ 
                    idx=(short)k; 
                    break; 
                } 
            }
            if (idx<DICT_SZ){
                if (!bw.putN((uint64_t)idx, DICT_BW)) return false;
                if (!bw.putN(right, D.rightBW)) return false;
            }else{
                if (!bw.putN(0, DICT_BW)) return false;
                if (!bw.putN(right, D.rightBW)) return false;
                exc_pos[exc_cnt]  = (uint16_t)i;
                exc_left[exc_cnt] = left;
                ++exc_cnt;
            }
        }
        
        for(int k=0;k<DICT_SZ;++k) {
            if (!bw.putN((uint64_t)D.dict[k], D.leftBW)) return false;
        }
        
        if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
        for(int i=0;i<exc_cnt;++i){
            if (!bw.putN((uint64_t)exc_left[i], D.leftBW)) return false;
            if (!bw.putN((uint64_t)exc_pos[i], 16)) return false;
        }
        return true;
    }

    // 修正的采样函数，使用设备兼容的数组替代std::map
    template<typename T>
    __device__ void rowgroup_sample_and_find_k_combinations(
        const T* rowgroup_data, 
        int rowgroup_size,
        int vectorSize,
        EFCombination* best_k_combinations,
        int& k_actual,
        CompressionMode& mode
    ) {
        const int available_alp_vectors = (rowgroup_size + vectorSize - 1) / vectorSize;
        
        // 使用设备兼容的数组替代std::map
        EFPair combinations_keys[19*19];
        int combinations_values[19*19];
        int combinations_count = 0;
        
        size_t best_estimated_compression_size = Constants<T>::RD_SIZE_THRESHOLD_LIMIT + 1;
        //选择向量idx
        for (size_t smp_n = 0; smp_n < config::ROWGROUP_VECTOR_SAMPLES && 
            smp_n * config::ROWGROUP_SAMPLES_JUMP < available_alp_vectors; smp_n++) {
            
            size_t vector_idx = smp_n * config::ROWGROUP_SAMPLES_JUMP;
            if (vector_idx >= available_alp_vectors) break;
            
            size_t vector_start = vector_idx * vectorSize;
            size_t current_vector_size = (vectorSize < (rowgroup_size - vector_start)) ? 
                                        vectorSize : (rowgroup_size - vector_start);
            
            if (current_vector_size < 2) continue;
            //没有32个数据就全采样
            const size_t samples_size = (config::SAMPLES_PER_VECTOR < current_vector_size) ? 
                                    config::SAMPLES_PER_VECTOR : current_vector_size;
            const int sample_increments = (current_vector_size + samples_size - 1) / samples_size;
            
            T samples[32];
            for (size_t i = 0; i < samples_size; ++i) {
                samples[i] = rowgroup_data[vector_start + i * sample_increments];
            }
            
            uint8_t found_exponent = 0;
            uint8_t found_factor = 0;
            uint64_t sample_estimated_compression_size = Constants<T>::RD_SIZE_THRESHOLD_LIMIT + 1;
            
            for (int8_t exp_ref = Constants<T>::MAX_EXPONENT; exp_ref >= 0; exp_ref--) {
                for (int8_t factor_idx = exp_ref; factor_idx >= 0; factor_idx--) {
                    uint16_t exceptions_count = 0;
                    uint16_t non_exceptions_count = 0;
                    uint32_t estimated_bits_per_value = 0;
                    uint64_t estimated_compression_size = 0;
                    long long max_encoded_value = LLONG_MIN;
                    long long min_encoded_value = LLONG_MAX;
                    
                    for (size_t i = 0; i < samples_size; i++) {
                        const T actual_value = samples[i];
                        double enc = actual_value * D_EXP_ARR[exp_ref] * D_FRAC_ARR[factor_idx];
                        long long encoded_value = fast_round_double(enc);
                        double decoded_value = double(encoded_value) * D_FACT_ARR[factor_idx] * D_FRAC_ARR[exp_ref];
                        
                        if (decoded_value == double(actual_value)) {
                            non_exceptions_count++;
                            if (encoded_value > max_encoded_value) max_encoded_value = encoded_value;
                            if (encoded_value < min_encoded_value) min_encoded_value = encoded_value;
                        } else {
                            exceptions_count++;
                        }
                    }
                    
                    if (non_exceptions_count < 2) continue;
                    
                    unsigned long long range = (unsigned long long)(max_encoded_value - min_encoded_value);
                    estimated_bits_per_value = width_needed_unsigned(range);
                    estimated_compression_size += samples_size * estimated_bits_per_value;
                    estimated_compression_size += exceptions_count * (Constants<T>::EXCEPTION_SIZE + Constants<T>::EXCEPTION_POSITION_SIZE);
                    
                    if ((estimated_compression_size < sample_estimated_compression_size) ||
                        (estimated_compression_size == sample_estimated_compression_size && 
                        found_exponent < exp_ref) ||
                        ((estimated_compression_size == sample_estimated_compression_size && 
                        found_exponent == exp_ref) && 
                        found_factor < factor_idx)) 
                    {
                        sample_estimated_compression_size = estimated_compression_size;
                        found_exponent = exp_ref;
                        found_factor = factor_idx;
                        if (sample_estimated_compression_size < best_estimated_compression_size) {
                            best_estimated_compression_size = sample_estimated_compression_size;
                        }
                    }
                }
            }
            
            // 记录找到的最佳组合，使用设备兼容的方法
            EFPair key(found_exponent, found_factor);
            bool found = false;
            for(int i = 0; i < combinations_count; i++) {
                if(combinations_keys[i].equals(key)) {
                    combinations_values[i]++;
                    found = true;
                    break;
                }
            }
            if(!found && combinations_count < 19*19) {
                combinations_keys[combinations_count] = key;
                combinations_values[combinations_count] = 1;
                combinations_count++;
            }
        }
        // if(combinations_count!=config::ROWGROUP_VECTOR_SAMPLES)
        // {
        //     printf("combinations_count:%d != %d\n",combinations_count,config::ROWGROUP_VECTOR_SAMPLES);
        // }
        if (best_estimated_compression_size >= Constants<T>::RD_SIZE_THRESHOLD_LIMIT) {
            mode = CompressionMode::ALPrd;
            k_actual = 0;
            return;
        }
        
        mode = CompressionMode::ALP;
        
        // 将组合转换为向量并排序
        EFCombination all_combinations[19*19];
        int num_combinations = 0;
        
        for(int i = 0; i < combinations_count; i++) {
            all_combinations[num_combinations].e = combinations_keys[i].exponent;
            all_combinations[num_combinations].f = combinations_keys[i].factor;
            all_combinations[num_combinations].count = combinations_values[i];
            all_combinations[num_combinations].score = 0;
            num_combinations++;
        }
        
        // 与CPU版本相同的排序逻辑
        for (int i = 0; i < num_combinations - 1; i++) {
            for (int j = i + 1; j < num_combinations; j++) {
                bool should_swap = false;
                if ((all_combinations[j].count > all_combinations[i].count)||
                    ((all_combinations[j].count == all_combinations[i].count) && (all_combinations[j].e < all_combinations[i].e))||
                    ((all_combinations[j].count == all_combinations[i].count) && (all_combinations[j].e == all_combinations[i].e) && (all_combinations[j].f < all_combinations[i].f))) 
                {
                    should_swap = true;
                }
                
                if (should_swap) {
                    EFCombination tmp = all_combinations[i];
                    all_combinations[i] = all_combinations[j];
                    all_combinations[j] = tmp;
                }
            }
        }
        
        k_actual = (config::MAX_K_COMBINATIONS < num_combinations) ? 
                config::MAX_K_COMBINATIONS : num_combinations;
        for (int i = 0; i < k_actual; i++) {
            best_k_combinations[i] = all_combinations[i];
        }
    }

    // 修正的二级采样函数
    template<typename T>
    __device__ void vector_choose_from_k_combinations(
        const T* vec_data,
        int vec_size,
        const EFCombination* k_combinations,
        int k,
        uint8_t& best_e,
        uint8_t& best_f,
        short& bitw,
        long long& FOR,
        int& exc
    ) {
        if (k == 0) {
            // 如果没有k组合，使用完整分析
            alp_vector_choose_best_bits<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
            return;
        }
        
        if (k == 1) {
            best_e = k_combinations[0].e;
            best_f = k_combinations[0].f;
            alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
            return;
        }
        
        // 二级采样
        T samples[32];
        const int sample_count = (config::SAMPLES_PER_VECTOR < vec_size) ? 
                                config::SAMPLES_PER_VECTOR : vec_size;
        const int sample_increments = (vec_size + sample_count - 1) / sample_count;
        
        for (int i = 0; i < sample_count; i++) {
            samples[i] = vec_data[i * sample_increments];
        }
        
        double best_score = 1e30;
        int worse_count = 0;
        
        for (int kid = 0; kid < k; kid++) {
            uint8_t e = k_combinations[kid].e;
            uint8_t f = k_combinations[kid].f;
            
            short test_bitw;
            long long test_FOR;
            int test_exc;
            alp_vector_analyze<T>(samples, sample_count, e, f, test_bitw, test_FOR, test_exc);
            
            int val_bits = std::is_same_v<T,double> ? 64 : 32;
            double score = sample_count * test_bitw + test_exc * (val_bits + 16);
            
            if (score < best_score) {
                best_score = score;
                best_e = e;
                best_f = f;
                worse_count = 0;
            } else {
                worse_count++;
                if (worse_count >= config::SAMPLING_EARLY_EXIT_THRESHOLD) {
                    break;
                }
            }
        }
        
        // 在完整向量上分析最终参数
        alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
    }

    static constexpr int THREADS_PER_BLOCK = 128;

    __global__ void kernel_write_rowgroup_headers(
        const uint64_t* blk_sizes,
        const uint64_t* blk_offsets,
        int numBlocks,
        int vectorSize,
        uint8_t* out_bytes
    ) {
        int blockId = blockIdx.x * blockDim.x + threadIdx.x;
        if (blockId >= numBlocks) return;
        
        int numVec = (blk_sizes[blockId] + vectorSize - 1) / vectorSize;
        uint64_t bit_offset = blk_offsets[blockId];
        
        SafeBitWriter bw(out_bytes, bit_offset, 8);
        bw.putN((uint64_t)numVec, 8);
    }

    // ==================== 第一阶段：采样Kernel ====================
    template<typename T>
    __global__ void kernel_rowgroup_sampling(
        const T* data,
        const uint64_t* blk_starts,
        const uint64_t* blk_sizes,
        int numBlocks,
        int vectorSize,
        // 输出每个rowgroup的采样结果
        uint8_t* out_modes,           // 0=ALP, 1=ALPrd
        uint8_t* out_k_actual,        // 每个rowgroup的k组合数量
        uint8_t* out_k_combinations,  // [numBlocks][5][2] 存储(e,f)组合
        // ALPrd字典输出（只在需要时使用）
        uint8_t* out_alprd_right_bw,
        uint8_t* out_alprd_left_bw,
        uint32_t* out_alprd_dicts     // [numBlocks][8] 字典
    ) {
        int blockId = blockIdx.x;
        if (blockId >= numBlocks) return;
        
        const T* blk = data + blk_starts[blockId];
        int n = (int)blk_sizes[blockId];
        
        // 每个block用一个线程完成采样（避免线程间通信复杂性）
        if (threadIdx.x == 0) {
            EFCombination k_combinations[5];
            int k_actual;
            CompressionMode mode;
            
            // 核心采样逻辑（只执行一次）
            rowgroup_sample_and_find_k_combinations<T>(
                blk, n, vectorSize, k_combinations, k_actual, mode
            );
            
            // 保存采样结果
            out_modes[blockId] = (mode == CompressionMode::ALPrd) ? 1 : 0;
            out_k_actual[blockId] = k_actual;
            
            if (mode == CompressionMode::ALP) {
                // 保存ALP的k组合
                for(int i = 0; i < k_actual; i++) {
                    int base_idx = blockId * 5 * 2 + i * 2;
                    out_k_combinations[base_idx] = k_combinations[i].e;
                    out_k_combinations[base_idx + 1] = k_combinations[i].f;
                }
            } else {
                // ALPrd模式：计算和保存字典
                uint64_t tmp[MAX_VEC];
                const int samples_to_process = min(n, MAX_VEC);
                
                for(int i = 0; i < samples_to_process; i++) {
                    if constexpr (std::is_same_v<T,double>) 
                        tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[i]);
                    else 
                        tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[i]);
                }
                
                ALPrdDict<T> D;
                alprd_find_best<T>(tmp, samples_to_process, D);
                
                out_alprd_right_bw[blockId] = D.rightBW;
                out_alprd_left_bw[blockId] = D.leftBW;
                
                // 保存字典
                for(int i = 0; i < DICT_SZ; i++) {
                    out_alprd_dicts[blockId * DICT_SZ + i] = D.dict[i];
                }
            }
        }
    }

    // ==================== 第二阶段：向量参数选择和大小计算 ====================
    template<typename T>
    __global__ void kernel_vector_parameter_selection(
        const T* data,
        const uint64_t* blk_starts,
        const uint64_t* blk_sizes,
        const uint8_t* modes,
        const uint8_t* k_actual,
        const uint8_t* k_combinations,
        const uint8_t* alprd_right_bw,
        const uint8_t* alprd_left_bw,
        const uint32_t* alprd_dicts,
        int numBlocks,
        int vectorSize,
        uint64_t total_vectors,
        // 输出每个向量的参数
        uint8_t* vec_e,
        uint8_t* vec_f,
        uint16_t* vec_bitw,
        int64_t* vec_FOR,
        uint16_t* vec_exc_cnt,
        uint64_t* vec_bit_sizes
    ) {
        int blockId = blockIdx.x;
        if (blockId >= numBlocks) return;
        
        const T* blk = data + blk_starts[blockId];
        int n = (int)blk_sizes[blockId];
        int numVec = (n + vectorSize - 1) / vectorSize;
        CompressionMode mode = (modes[blockId] == 1) ? CompressionMode::ALPrd : CompressionMode::ALP;
        
        // 计算全局向量索引基址
        uint64_t vec_base = 0;
        for(int b = 0; b < blockId; b++) {
            vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
        }
        
        if (mode == CompressionMode::ALP) {
            // 重建k组合数组
            EFCombination local_k_combinations[5];
            int local_k_actual = k_actual[blockId];
            
            for(int i = 0; i < local_k_actual; i++) {
                int base_idx = blockId * 5 * 2 + i * 2;
                local_k_combinations[i].e = k_combinations[base_idx];
                local_k_combinations[i].f = k_combinations[base_idx + 1];
            }
            
            // 并行处理每个向量
            for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
                uint64_t global_vec_idx = vec_base + v;
                int beg = v * vectorSize;
                int rem = n - beg;
                int len = (vectorSize < rem ? vectorSize : rem);
                
                uint8_t e, f;
                short bitw;
                long long FOR;
                int exc;
                
                // 对应CPU的 find_best_exponent_factor_from_combinations
                vector_choose_from_k_combinations<T>(
                    blk + beg, len,
                    local_k_combinations, local_k_actual,
                    e, f, bitw, FOR, exc
                );
                
                // 保存参数
                vec_e[global_vec_idx] = e;
                vec_f[global_vec_idx] = f;
                vec_bitw[global_vec_idx] = bitw;
                vec_FOR[global_vec_idx] = FOR;
                vec_exc_cnt[global_vec_idx] = exc;
                vec_bit_sizes[global_vec_idx] = alp_vector_size_bits_safe<T>(len, e, f, bitw, exc);
            }
        } else {
            // ALPrd模式：使用预计算的字典
            ALPrdDict<T> D;
            D.rightBW = alprd_right_bw[blockId];
            D.leftBW = alprd_left_bw[blockId];
            for(int i = 0; i < DICT_SZ; i++) {
                D.dict[i] = alprd_dicts[blockId * DICT_SZ + i];
            }
            
            for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
                uint64_t global_vec_idx = vec_base + v;
                int beg = v * vectorSize;
                int rem = n - beg;
                int len = (vectorSize < rem ? vectorSize : rem);
                
                // 计算异常数量
                int exc = 0;
                for(int i = 0; i < len; i++) {
                    uint64_t raw;
                    if constexpr (std::is_same_v<T,double>) 
                        raw = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                    else 
                        raw = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
                    
                    uint32_t left = (uint32_t)((raw >> D.rightBW) & mask_lo(D.leftBW));
                    bool inDict = false;
                    for(int k = 0; k < DICT_SZ; k++) {
                        if(D.dict[k] == left) {
                            inDict = true;
                            break;
                        }
                    }
                    if(!inDict) exc++;
                }
                
                // ALPrd标记
                vec_e[global_vec_idx] = 0xFF;
                vec_f[global_vec_idx] = 0xFF;
                vec_exc_cnt[global_vec_idx] = exc;
                vec_bit_sizes[global_vec_idx] = alprd_vector_size_bits_safe<T>(len, D, exc);
            }
        }
    }

    // ==================== 第三阶段：压缩数据写入 ====================
    template<typename T>
    __global__ void kernel_compress_and_write(
        const T* data,
        const uint64_t* blk_starts,
        const uint64_t* blk_sizes,
        const uint8_t* modes,
        const uint8_t* alprd_right_bw,
        const uint8_t* alprd_left_bw,
        const uint32_t* alprd_dicts,
        const uint64_t* vec_bit_offsets,
        const uint8_t* vec_e,
        const uint8_t* vec_f,
        const uint16_t* vec_bitw,
        const int64_t* vec_FOR,
        int numBlocks,
        int vectorSize,
        uint8_t* out_bytes
    ) {
        int blockId = blockIdx.x;
        if (blockId >= numBlocks) return;
        
        const T* blk = data + blk_starts[blockId];
        int n = (int)blk_sizes[blockId];
        int numVec = (n + vectorSize - 1) / vectorSize;
        CompressionMode mode = (modes[blockId] == 1) ? CompressionMode::ALPrd : CompressionMode::ALP;
        
        // 计算全局向量索引基址
        uint64_t vec_base = 0;
        for(int b = 0; b < blockId; b++) {
            vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
        }
        
        // 并行写入每个向量
        for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
            uint64_t global_vec_idx = vec_base + v;
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = (vectorSize < rem ? vectorSize : rem);
            
            uint64_t bit_offset = vec_bit_offsets[global_vec_idx];
            SafeBitWriter bw(out_bytes, bit_offset, 100000); // 充足的缓冲区
            
            if(mode == CompressionMode::ALP) {
                // ALP压缩写入
                uint8_t e = vec_e[global_vec_idx];
                uint8_t f = vec_f[global_vec_idx];
                short bitw = vec_bitw[global_vec_idx];
                long long FOR = vec_FOR[global_vec_idx];
                
                alp_vector_write_safe<T>(bw, blk + beg, len, e, f, bitw, FOR);
                
            } else {
                // ALPrd压缩写入：使用预计算的字典
                uint64_t tmp[MAX_VEC];
                for(int i = 0; i < len; i++) {
                    if constexpr (std::is_same_v<T,double>) 
                        tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                    else 
                        tmp[i] = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
                }
                
                ALPrdDict<T> D;
                D.rightBW = alprd_right_bw[blockId];
                D.leftBW = alprd_left_bw[blockId];
                for(int i = 0; i < DICT_SZ; i++) {
                    D.dict[i] = alprd_dicts[blockId * DICT_SZ + i];
                }
                
                alprd_vector_write_safe<T>(bw, tmp, len, D);
            }
        }
    }

    template<typename T>
    __global__ void kernel_decompress_debug(const uint8_t* bytes,
                                            const uint64_t* blk_starts_bits,
                                            const uint64_t* blk_bits,
                                            const uint64_t* out_starts,
                                            const int vectorSize,
                                            T* out_data, 
                                            int numBlocks) {
        int blockId = blockIdx.x;
        if (blockId >= numBlocks) return;

        uint64_t bit_offset = blk_starts_bits[blockId];
        BitReader br{bytes, bit_offset};
        
        int numVec = (int)br.getN(8);

        if (numVec <= 0 || numVec > 10000) {
            return;
        }

        uint64_t out_pos = out_starts[blockId];
        
        for(int v = 0; v < numVec; v++) {
            int useALP = br.get1();

            if (useALP) {
                uint8_t e = (uint8_t)br.getN(8);
                uint8_t f = (uint8_t)br.getN(8);
                short bitw = (short)br.getN(16);
                long long FOR = (long long)br.getN(64);
                int n = (int)br.getN(32);

                if (n <= 0 || n > MAX_VEC) return;

                for(int k = 0; k < n; k++) {
                    uint64_t enc = br.getN(bitw);
                    long long I = FOR + (long long)enc;
                    double dec = double(I) *  D_FACT_ARR[f] * D_FRAC_ARR[e];
                    out_data[out_pos + k] = (T)dec;
                }

                int exc = (int)br.getN(16);
                for(int t = 0; t < exc; t++) {
                    uint64_t raw = std::is_same_v<T,double> ? br.getN(64) : br.getN(32);
                    int pos = (int)br.getN(16);
                    
                    if (pos < n) {
                        if constexpr (std::is_same_v<T,double>) {
                            double val = *reinterpret_cast<double*>(&raw);
                            out_data[out_pos + pos] = (T)val;
                        } else {
                            uint32_t rv = (uint32_t)raw;
                            float val = *reinterpret_cast<float*>(&rv);
                            out_data[out_pos + pos] = (T)val;
                        }
                    }
                }
                out_pos += n;
            } else {
                int n = (int)br.getN(32);
                if (n <= 0 || n > MAX_VEC) return;

                uint8_t rbw = (uint8_t)br.getN(8);
                uint64_t right[MAX_VEC]; 
                uint16_t leftIdx[MAX_VEC];
                
                for(int k = 0; k < n; k++) {
                    leftIdx[k] = (uint16_t)br.getN(DICT_BW);
                    right[k] = br.getN(rbw);
                }

                uint8_t lbw = uint8_t(sizeof(T)*8 - rbw);
                uint64_t dict[DICT_SZ];
                for(int k = 0; k < DICT_SZ; k++) {
                    dict[k] = br.getN(lbw);
                }

                int exc = (int)br.getN(16);
                uint16_t exc_pos[MAX_VEC]; 
                uint64_t exc_left[MAX_VEC];
                for(int t = 0; t < exc; t++) {
                    exc_left[t] = br.getN(lbw);
                    exc_pos[t] = (uint16_t)br.getN(16);
                }

                for(int k = 0; k < n; k++) {
                    uint64_t left = (leftIdx[k] < DICT_SZ) ? dict[leftIdx[k]] : 0ULL;
                    uint64_t raw = (left << rbw) | right[k];
                    
                    if constexpr (std::is_same_v<T,double>) {
                        double val = *reinterpret_cast<double*>(&raw);
                        out_data[out_pos + k] = (T)val;
                    } else {
                        uint32_t r32 = (uint32_t)raw;
                        float val = *reinterpret_cast<float*>(&r32);
                        out_data[out_pos + k] = (T)val;
                    }
                }

                for(int t = 0; t < exc; t++) {
                    int p = exc_pos[t];
                    uint64_t raw = (exc_left[t] << rbw) | right[p];
                    
                    if (p < n) {
                        if constexpr (std::is_same_v<T,double>) {
                            double val = *reinterpret_cast<double*>(&raw);
                            out_data[out_pos + p] = (T)val;
                        } else {
                            uint32_t r32 = (uint32_t)raw;
                            float val = *reinterpret_cast<float*>(&r32);
                            out_data[out_pos + p] = (T)val;
                        }
                    }
                }
                out_pos += n;
            }
        }
    }

    //  ==================== 1.压缩实现 ====================
    //  核心设备端压缩实现（通用函数）

    template<typename T>
    static CompressedDevice compress_core_device(const T* d_data, size_t n, const Params& p, cudaStream_t stream) {
        CompressedDevice c;
        if (n == 0) { 
            c.vectorSize = p.vectorSize; 
            return c; 
        }
        
        const int V = p.vectorSize;
        const int B = p.blockSize > 0 ? p.blockSize : int(n);
        const int numBlocks = int((n + B - 1) / B);
        
        // 计算块信息
        uint64_t total_vectors = 0;
        std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
        size_t pos = 0;
        for(int i = 0; i < numBlocks; i++) {
            h_starts[i] = pos;
            uint64_t sz = std::min<uint64_t>(B, n - pos);
            h_sizes[i] = sz;
            total_vectors += (sz + V - 1) / V;
            pos += sz;
        }
        
        // 拷贝块信息到设备
        uint64_t *d_starts = nullptr, *d_sizes = nullptr;
        CUDA_CHECK(cudaMalloc(&d_starts, numBlocks * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sizes, numBlocks * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_starts, h_starts.data(), numBlocks * sizeof(uint64_t), 
                                cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_sizes, h_sizes.data(), numBlocks * sizeof(uint64_t), 
                                cudaMemcpyHostToDevice, stream));
        
        // ==================== 阶段1：采样 ====================
        uint8_t* d_modes = nullptr;
        uint8_t* d_k_actual = nullptr;
        uint8_t* d_k_combinations = nullptr;
        uint8_t* d_alprd_right_bw = nullptr;
        uint8_t* d_alprd_left_bw = nullptr;
        uint32_t* d_alprd_dicts = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_modes, numBlocks));
        CUDA_CHECK(cudaMalloc(&d_k_actual, numBlocks));
        CUDA_CHECK(cudaMalloc(&d_k_combinations, numBlocks * 5 * 2)); // [blocks][5 combinations][e,f]
        CUDA_CHECK(cudaMalloc(&d_alprd_right_bw, numBlocks));
        CUDA_CHECK(cudaMalloc(&d_alprd_left_bw, numBlocks));
        CUDA_CHECK(cudaMalloc(&d_alprd_dicts, numBlocks * DICT_SZ * sizeof(uint32_t)));
        
        dim3 sampling_grid(numBlocks);
        dim3 sampling_block(32); // 每个rowgroup用少量线程
        
        kernel_rowgroup_sampling<T><<<sampling_grid, sampling_block, 0, stream>>>(
            d_data, d_starts, d_sizes, numBlocks, V,
            d_modes, d_k_actual, d_k_combinations,
            d_alprd_right_bw, d_alprd_left_bw, d_alprd_dicts
        );
        
        // ============== 新增：记录rowgroup采样结果 ==============
        if (p.enable_recording && get_gpu_recording_enabled()) {
            // 拷贝GPU结果到主机端进行记录
            std::vector<uint8_t> h_modes(numBlocks);
            std::vector<uint8_t> h_k_actual(numBlocks);
            std::vector<uint8_t> h_k_combinations(numBlocks * 5 * 2);
            
            CUDA_CHECK(cudaMemcpyAsync(h_modes.data(), d_modes, numBlocks, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_k_actual.data(), d_k_actual, numBlocks, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_k_combinations.data(), d_k_combinations, numBlocks * 5 * 2, 
                                    cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // 为每个rowgroup记录k组合
            for(int blockId = 0; blockId < numBlocks; blockId++) {
                if(h_modes[blockId] == 0) { // ALP模式
                    std::vector<std::pair<std::pair<int, int>, int>> combinations;
                    int k_count = h_k_actual[blockId];
                    for(int i = 0; i < k_count; i++) {
                        int base_idx = blockId * 5 * 2 + i * 2;
                        int e = h_k_combinations[base_idx];
                        int f = h_k_combinations[base_idx + 1];
                        combinations.push_back({{e, f}, 1}); // count设为1，实际使用中可能需要更复杂的统计
                    }
                    record_gpu_rowgroup_combinations(combinations);
                } else {
                    // ALPrd模式不记录

                }
            }
        }
        
        // ==================== 阶段2：向量参数选择 ====================
        uint8_t* d_vec_e = nullptr;
        uint8_t* d_vec_f = nullptr;
        uint16_t* d_vec_bitw = nullptr;
        int64_t* d_vec_FOR = nullptr;
        uint16_t* d_vec_exc_cnt = nullptr;
        uint64_t* d_vec_bit_sizes = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_vec_e, total_vectors));
        CUDA_CHECK(cudaMalloc(&d_vec_f, total_vectors));
        CUDA_CHECK(cudaMalloc(&d_vec_bitw, total_vectors * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_vec_FOR, total_vectors * sizeof(int64_t)));
        CUDA_CHECK(cudaMalloc(&d_vec_exc_cnt, total_vectors * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_vec_bit_sizes, total_vectors * sizeof(uint64_t)));
        
        dim3 param_grid(numBlocks);
        dim3 param_block(THREADS_PER_BLOCK);
        
        kernel_vector_parameter_selection<T><<<param_grid, param_block, 0, stream>>>(
            d_data, d_starts, d_sizes,
            d_modes, d_k_actual, d_k_combinations,
            d_alprd_right_bw, d_alprd_left_bw, d_alprd_dicts,
            numBlocks, V, total_vectors,
            d_vec_e, d_vec_f, d_vec_bitw, d_vec_FOR, d_vec_exc_cnt, d_vec_bit_sizes
        );
        
        // ============== 新增：记录向量(e,f)选择 ==============
        if (p.enable_recording && get_gpu_recording_enabled()) {
            // 拷贝向量参数到主机端
            std::vector<uint8_t> h_vec_e(total_vectors);
            std::vector<uint8_t> h_vec_f(total_vectors);
            std::vector<uint8_t> h_modes_for_vectors(numBlocks);
            
            CUDA_CHECK(cudaMemcpyAsync(h_vec_e.data(), d_vec_e, total_vectors, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_vec_f.data(), d_vec_f, total_vectors, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_modes_for_vectors.data(), d_modes, numBlocks, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // 记录每个向量的(e,f)选择
            uint64_t vec_idx = 0;
            for(int blockId = 0; blockId < numBlocks; blockId++) {
                int numVec = (h_sizes[blockId] + V - 1) / V;
                for(int v = 0; v < numVec; v++) {
                    if(h_modes_for_vectors[blockId] == 0) { // ALP模式
                        uint8_t e = h_vec_e[vec_idx];
                        uint8_t f = h_vec_f[vec_idx];
                        record_gpu_vector_ef(e, f);
                    }
                    // ALPrd模式的向量用0xFF标记，不记录
                    vec_idx++;
                }
            }
        }
        
        // 拷贝bit_sizes用于计算偏移
        std::vector<uint64_t> h_vec_bit_sizes(total_vectors);
        CUDA_CHECK(cudaMemcpyAsync(h_vec_bit_sizes.data(), d_vec_bit_sizes, 
                                total_vectors * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // ==================== 计算偏移量 ====================
        std::vector<uint64_t> h_block_offsets(numBlocks);
        std::vector<uint64_t> h_vec_offsets(total_vectors);
        
        uint64_t block_offset = 0;
        uint64_t vec_idx = 0;
        
        for(int b = 0; b < numBlocks; b++) {
            h_block_offsets[b] = block_offset;
            uint64_t current_offset = block_offset + 8; // block header
            
            int numVec = (h_sizes[b] + V - 1) / V;
            for(int v = 0; v < numVec; v++) {
                h_vec_offsets[vec_idx] = current_offset;
                current_offset += h_vec_bit_sizes[vec_idx];
                vec_idx++;
            }
            
            // 块级padding
            uint64_t block_bits = current_offset - block_offset;
            uint64_t padding = (32 - (block_bits & 31)) & 31;
            block_offset = current_offset + padding;
        }
        
        const uint64_t total_bytes = (block_offset + 7) / 8;
        
        // ==================== 阶段3：压缩写入 ====================
        CUDA_CHECK(cudaMalloc(&c.d_data, total_bytes));
        CUDA_CHECK(cudaMemsetAsync(c.d_data, 0, total_bytes, stream));
        c.data_size = total_bytes;
        
        // 拷贝偏移量到设备
        uint64_t* d_vec_offsets = nullptr;
        uint64_t* d_block_offsets = nullptr;
        CUDA_CHECK(cudaMalloc(&d_vec_offsets, total_vectors * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpyAsync(d_vec_offsets, h_vec_offsets.data(), 
                                total_vectors * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_block_offsets, h_block_offsets.data(), 
                                numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
        
        // 写入块头
        kernel_write_rowgroup_headers<<<(numBlocks + 255) / 256, 256, 0, stream>>>(
            d_sizes, d_block_offsets, numBlocks, V, c.d_data
        );
        
        // 压缩写入数据
        kernel_compress_and_write<T><<<param_grid, param_block, 0, stream>>>(
            d_data, d_starts, d_sizes,
            d_modes, d_alprd_right_bw, d_alprd_left_bw, d_alprd_dicts,
            d_vec_offsets, d_vec_e, d_vec_f, d_vec_bitw, d_vec_FOR,
            numBlocks, V, c.d_data
        );
        
        // 设置元数据
        c.offsets = std::move(h_block_offsets);
        c.bit_sizes.resize(numBlocks);
        vec_idx = 0;
        for(int b = 0; b < numBlocks; b++) {
            uint64_t block_bits = 8; // header
            int numVec = (h_sizes[b] + V - 1) / V;
            for(int v = 0; v < numVec; v++) {
                block_bits += h_vec_bit_sizes[vec_idx++];
            }
            c.bit_sizes[b] = block_bits;
        }
        c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
        c.vectorSize = V;
        
        // ==================== 清理内存 ====================
        CUDA_CHECK(cudaFree(d_block_offsets));
        CUDA_CHECK(cudaFree(d_vec_offsets));
        CUDA_CHECK(cudaFree(d_vec_bit_sizes));
        CUDA_CHECK(cudaFree(d_vec_exc_cnt));
        CUDA_CHECK(cudaFree(d_vec_FOR));
        CUDA_CHECK(cudaFree(d_vec_bitw));
        CUDA_CHECK(cudaFree(d_vec_f));
        CUDA_CHECK(cudaFree(d_vec_e));
        CUDA_CHECK(cudaFree(d_alprd_dicts));
        CUDA_CHECK(cudaFree(d_alprd_left_bw));
        CUDA_CHECK(cudaFree(d_alprd_right_bw));
        CUDA_CHECK(cudaFree(d_k_combinations));
        CUDA_CHECK(cudaFree(d_k_actual));
        CUDA_CHECK(cudaFree(d_modes));
        CUDA_CHECK(cudaFree(d_sizes));
        CUDA_CHECK(cudaFree(d_starts));
        
        return c;
    }

    //  主机端
    template<typename T>
    static Compressed compress_impl(const T* h_data, size_t n, const Params& p) {
        if (n == 0) {
            Compressed c;
            c.vectorSize = p.vectorSize;
            return c;
        }
        
        // 启用记录功能（如果参数中启用）
        if (p.enable_recording) {
            enable_gpu_alp_recording(true);
        }
        
        // 分配设备内存并拷贝数据
        T* d_data = nullptr;
        CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
        CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
        
        // 调用核心设备端实现
        CompressedDevice device_result = compress_core_device<T>(d_data, n, p, 0);
        
        // 转换为主机端格式
        Compressed host_result = device_to_host(device_result);
        
        // 清理
        CUDA_CHECK(cudaFree(d_data));
        
        return host_result;
    }

    //  设备端
    template<typename T>
    static CompressedDevice compress_device_impl(const T* d_data, size_t n, const Params& p, cudaStream_t stream) {
        // 启用记录功能（如果参数中启用）
        if (p.enable_recording) {
            enable_gpu_alp_recording(true);
        }
        
        // 直接调用核心实现
        return compress_core_device<T>(d_data, n, p, stream);
    }

    //  2.解压实现
    template<typename T>
    static void decompress_core_device(const uint8_t* d_compressed_data,
                                    const std::vector<uint64_t>& offsets,
                                    const std::vector<uint64_t>& bit_sizes,
                                    const std::vector<uint32_t>& elem_counts,
                                    int vectorSize,
                                    T* d_output,
                                    size_t output_size,
                                    cudaStream_t stream) {
        if (output_size == 0) return;
        
        const int numBlocks = (int)offsets.size();
        if (numBlocks == 0) return;
        
        // 分配并拷贝元数据到设备
        uint64_t *d_offsets = nullptr, *d_bit_sizes = nullptr, *d_output_starts = nullptr;
        CUDA_CHECK(cudaMalloc(&d_offsets, numBlocks * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_bit_sizes, numBlocks * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_output_starts, numBlocks * sizeof(uint64_t)));
        
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, offsets.data(), numBlocks * sizeof(uint64_t), 
                                cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_bit_sizes, bit_sizes.data(), numBlocks * sizeof(uint64_t), 
                                cudaMemcpyHostToDevice, stream));
        
        // 计算输出起始位置
        std::vector<uint64_t> output_starts(numBlocks);
        uint64_t accumulated = 0;
        for(int i = 0; i < numBlocks; i++) {
            output_starts[i] = accumulated;
            accumulated += elem_counts[i];
        }
        
        CUDA_CHECK(cudaMemcpyAsync(d_output_starts, output_starts.data(), numBlocks * sizeof(uint64_t), 
                                cudaMemcpyHostToDevice, stream));
        
        // 初始化输出数据（用于调试）
        CUDA_CHECK(cudaMemsetAsync(d_output, 0, output_size * sizeof(T), stream));
        
        // 启动解压kernel
        dim3 grid(numBlocks);
        dim3 block(1); // 每个block用单线程处理一个压缩块
        
        kernel_decompress_debug<T><<<grid, block, 0, stream>>>(
            d_compressed_data,
            d_offsets,
            d_bit_sizes, 
            d_output_starts,
            vectorSize,
            d_output,
            numBlocks
        );
        
        // 清理设备内存
        CUDA_CHECK(cudaFree(d_output_starts));
        CUDA_CHECK(cudaFree(d_bit_sizes));
        CUDA_CHECK(cudaFree(d_offsets));
    }

    // 主机端解压实现
    template<typename T>
    static void decompress_impl(const Compressed& compressed, 
                                    T* h_output, 
                                    size_t output_size, 
                                    const Params& params) {
        if (output_size == 0 || compressed.empty()) return;
        
        // 分配设备内存
        uint8_t* d_compressed_data = nullptr;
        T* d_output = nullptr;
        
        CUDA_CHECK(cudaMalloc(&d_compressed_data, compressed.data.size()));
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(T)));
        
        // 拷贝压缩数据到设备
        CUDA_CHECK(cudaMemcpy(d_compressed_data, compressed.data.data(), 
                            compressed.data.size(), cudaMemcpyHostToDevice));
        
        // 调用核心解压实现
        decompress_core_device<T>(
            d_compressed_data,
            compressed.offsets,
            compressed.bit_sizes,
            compressed.elem_counts,
            compressed.vectorSize,
            d_output,
            output_size,
            0  // 默认stream
        );
        
        // 等待GPU完成并拷贝结果回主机
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost));
        
        // 清理设备内存
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_compressed_data));
    }

    // 设备端解压实现
    template<typename T>
    static void decompress_device_impl(const CompressedDevice& compressed_device, 
                                            T* d_output, 
                                            size_t output_size, 
                                            const Params& params, 
                                            cudaStream_t stream) {
        // 直接调用核心实现，数据已在设备上
        decompress_core_device<T>(
            compressed_device.d_data,
            compressed_device.offsets,
            compressed_device.bit_sizes,
            compressed_device.elem_counts,
            compressed_device.vectorSize,
            d_output,
            output_size,
            stream
        );
    }

    //  主机端
        Compressed compress_double(const double* data, size_t n, const Params& p) { 
            return compress_impl<double>(data, n, p); 
        }
        Compressed compress_float(const float* data, size_t n, const Params& p) { 
            return compress_impl<float>(data, n, p); 
        }
        void decompress_double(const Compressed& c, double* out, size_t n, const Params& p) { 
            decompress_impl<double>(c, out, n, p); 
        }
        void decompress_float(const Compressed& c, float* out, size_t n, const Params& p) { 
            decompress_impl<float>(c, out, n, p); 
        }

    //  设备端
        CompressedDevice compress_double_device(const double* d_data, size_t n, const Params& p, cudaStream_t stream) {
            return compress_device_impl<double>(d_data, n, p, stream);
        }

        CompressedDevice compress_float_device(const float* d_data, size_t n, const Params& p, cudaStream_t stream) {
            return compress_device_impl<float>(d_data, n, p, stream);
        }

        void decompress_double_device(const CompressedDevice& c, double* d_out, size_t n, const Params& p, cudaStream_t stream) {
            decompress_device_impl<double>(c, d_out, n, p, stream);
        }

        void decompress_float_device(const CompressedDevice& c, float* d_out, size_t n, const Params& p, cudaStream_t stream) {
            decompress_device_impl<float>(c, d_out, n, p, stream);
        }

    // 数据交互函数
    Compressed device_to_host(const CompressedDevice& cd) {
        Compressed c;
        c.data.resize(cd.data_size);
        cudaMemcpy(c.data.data(), cd.d_data, cd.data_size, cudaMemcpyDeviceToHost);
        c.offsets = cd.offsets;
        c.bit_sizes = cd.bit_sizes;
        c.elem_counts = cd.elem_counts;
        c.vectorSize = cd.vectorSize;
        return c;
    }

    CompressedDevice host_to_device(const Compressed& c) {
        CompressedDevice cd;
        cd.data_size = c.data.size();
        cudaMalloc(&cd.d_data, cd.data_size);
        cudaMemcpy(cd.d_data, c.data.data(), cd.data_size, cudaMemcpyHostToDevice);
        cd.offsets = c.offsets;
        cd.bit_sizes = c.bit_sizes;
        cd.elem_counts = c.elem_counts;
        cd.vectorSize = c.vectorSize;
        return cd;
    }

    } // namespace alp_gpu
*/
#include "alp_gpu.hpp"

using std::uint8_t; using std::uint32_t; using std::uint64_t;

namespace alp_gpu {

// 设备兼容的键值对结构，替代std::pair
struct EFPair {
    uint8_t exponent, factor;
    __device__ EFPair() : exponent(0), factor(0) {}
    __device__ EFPair(uint8_t e, uint8_t f) : exponent(e), factor(f) {}
    __device__ bool equals(const EFPair& other) const {
        return exponent == other.exponent && factor == other.factor;
    }
};

// 修正后的设备端ALP精确匹配检查 - 支持32位和64位
template<typename T>
__device__ inline bool alp_exact_equal(T v, uint8_t e, uint8_t f)
{
    if constexpr (std::is_same_v<T,double>) {
        double enc = v * D_EXP_ARR[e] * D_FRAC_ARR[f];
        long long I = fast_round_double(enc);
        double dec = (double)I * (double)D_FACT_ARR[f] * D_FRAC_ARR[e];
        return dec == v;
    } else {
        float enc = v * F_EXP_ARR[e] * F_FRAC_ARR[f];
        int I = fast_round_float(enc);
        float dec = (float)I * (float)(F_FACT_ARR[f]) * F_FRAC_ARR[e];
        return dec == v;
    }
}

template<typename T>
__device__ inline void alp_vector_analyze(const T* v, int n, uint8_t e, uint8_t f,
                                          short& bitw, long long& FOR,
                                          int& exc_cnt){
    long long mn=LLONG_MAX, mx=LLONG_MIN;
    exc_cnt=0;
    
    for(int i=0;i<n;++i){
        if constexpr (std::is_same_v<T,double>) {
            double enc = (double)v[i] * D_EXP_ARR[e] * D_FRAC_ARR[f];
            long long I = fast_round_double(enc);
            double dec = (double)I * (double)D_FACT_ARR[f] * D_FRAC_ARR[e];
            if (dec == (double)v[i]) { 
                mn = (mn < I ? mn : I); 
                mx = (mx > I ? mx : I); 
            } else {
                ++exc_cnt;
            }
        } else {
            float enc = (float)v[i] * F_EXP_ARR[e] * F_FRAC_ARR[f];
            int I = fast_round_float(enc);
            float dec = (float)I * (float)(F_FACT_ARR[f]) * F_FRAC_ARR[e];
            if (dec == (float)v[i]) { 
                long long I_ll = (long long)I;
                mn = (mn < I_ll ? mn : I_ll); 
                mx = (mx > I_ll ? mx : I_ll); 
            } else {
                ++exc_cnt;
            }
        }
    }
    unsigned long long range = (mn==LLONG_MAX)? 0ULL : (unsigned long long)(mx - mn);
    bitw = (short)width_needed_unsigned(range);
    FOR  = (mn==LLONG_MAX?0:mn);
}

// 添加缺失的函数定义
template<typename T>
__device__ inline void alp_vector_choose_best_bits(
    const T* v, int n,
    uint8_t& best_e, uint8_t& best_f,
    short& bitw, long long& FOR, int& exc)
{
    const int val_bits = std::is_same_v<T,double> ? 64 : 32;
    double best_score = 1e300;
    best_e=0; best_f=0; bitw=0; FOR=0; exc=0;

    for(int8_t e_idx = Constants<T>::MAX_EXPONENT; e_idx >= 0; --e_idx){
        for(int8_t f_idx = e_idx; f_idx >= 0; --f_idx){
            short _bw; long long _FOR; int _exc;
            alp_vector_analyze<T>(v, n, e_idx, f_idx, _bw, _FOR, _exc);
            
            double score = (double)n*_bw + (double)_exc*(val_bits + 16);
            
            if (score < best_score){
                best_score = score;
                best_e = e_idx; 
                best_f = f_idx; 
                bitw = _bw; 
                FOR = _FOR; 
                exc = _exc;
            }
        }
    }
}

template<typename T>
__device__ inline uint64_t alp_vector_size_bits_safe(int n, uint8_t e, uint8_t f,
                                                     short bitw, int exc_cnt){
    const int val_bits = std::is_same_v<T,double> ? 64 : 32;
    return 145ULL + (uint64_t)n * (uint64_t)bitw + (uint64_t)exc_cnt * (val_bits + 16);
}

template<typename T>
__device__ inline bool alp_vector_write_safe(SafeBitWriter& bw, const T* v, int n,
                                            uint8_t e, uint8_t f, short bitw, long long FOR){
    assert(n <= MAX_VEC);

    if (!bw.put1(1)) return false; // useALP = 1
    if (!bw.putN((uint64_t)e, 8)) return false;
    if (!bw.putN((uint64_t)f, 8)) return false;
    if (!bw.putN((uint64_t)bitw, 16)) return false;
    if (!bw.putN((uint64_t)FOR, 64)) return false;
    if (!bw.putN((uint64_t)n, 32)) return false;

    int exc_cnt=0;
    int      exc_pos[MAX_VEC];
    uint64_t exc_val[MAX_VEC];

    for(int i=0;i<n;++i){
        if constexpr (std::is_same_v<T,double>) {
            double enc = (double)v[i] * D_EXP_ARR[e] * D_FRAC_ARR[f];
            long long I = fast_round_double(enc);
            double dec = (double)I * (double)D_FACT_ARR[f] * D_FRAC_ARR[e];
            if (dec == (double)v[i]) {
                uint64_t packed = (uint64_t)(I - FOR);
                if (!bw.putN(packed, bitw)) return false;
            } else {
                if (!bw.putN(0, bitw)) return false;
                uint64_t raw = *reinterpret_cast<const uint64_t*>(&v[i]);
                exc_val[exc_cnt] = raw;
                exc_pos[exc_cnt] = i;
                ++exc_cnt;
            }
        } else {
            float enc = (float)v[i] * F_EXP_ARR[e] * F_FRAC_ARR[f];
            int I = fast_round_float(enc);
            float dec = (float)I * (float)(F_FACT_ARR[f])* F_FRAC_ARR[e];
            if (dec == (float)v[i]) {
                long long I_ll = (long long)I;
                uint64_t packed = (uint64_t)(I_ll - FOR);
                if (!bw.putN(packed, bitw)) return false;
            } else {
                if (!bw.putN(0, bitw)) return false;
                uint32_t raw = *reinterpret_cast<const uint32_t*>(&v[i]);
                exc_val[exc_cnt] = raw;
                exc_pos[exc_cnt] = i;
                ++exc_cnt;
            }
        }
    }
    
    if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
    for(int k=0;k<exc_cnt;++k){
        if constexpr (std::is_same_v<T,double>) {
            if (!bw.putN(exc_val[k], 64)) return false;
        } else {
            if (!bw.putN(exc_val[k], 32)) return false;
        }
        if (!bw.putN((uint64_t)exc_pos[k], 16)) return false;
    }
    return true;
}

template<typename T>
__device__ inline void alprd_find_best(const uint64_t* in, int n, ALPrdDict<T>& D){
    double best_score = 1e100; 
    int best_rbw = (int)sizeof(T)*8 - 1;
    uint32_t best_dict[DICT_SZ] = {0};

    for(int lbw=1; lbw<=CUT_LIM; ++lbw){
        int rbw = (int)sizeof(T)*8 - lbw;
        uint32_t lmask = mask_lo(lbw);

        uint32_t uniq_left[MAX_VEC]; 
        int cnt[MAX_VEC];
        int u = 0;

        for(int i=0;i<n;++i){
            uint32_t left = (uint32_t)((in[i] >> rbw) & lmask);
            int j=0; 
            for(; j<u; ++j) {
                if (uniq_left[j]==left) { 
                    ++cnt[j]; 
                    break; 
                }
            }
            if (j==u){ 
                uniq_left[u]=left; 
                cnt[u]=1; 
                ++u; 
            }
        }
        
        uint32_t dict[DICT_SZ]={0};
        int used = (DICT_SZ < u ? DICT_SZ : u);
        for(int k=0;k<used;++k){
            int best_cnt=-1, best_id=-1;
            for(int j=0;j<u;++j){
                bool taken=false;
                for(int t=0;t<k;++t) {
                    if (dict[t]==uniq_left[j]) { 
                        taken=true; 
                        break; 
                    }
                }
                if (taken) continue;
                if (cnt[j]>best_cnt){ 
                    best_cnt=cnt[j]; 
                    best_id=j; 
                }
            }
            if (best_id >= 0) {
                dict[k] = uniq_left[best_id];
            }
        }
        
        int keep=0;
        for(int k=0;k<used;++k){
            for(int j=0;j<u;++j) {
                if (uniq_left[j]==dict[k]) { 
                    keep += cnt[j]; 
                    break; 
                }
            }
        }
        int exc = n - keep;

        double bits = 1 + 32 + 8 + (double)n*(DICT_BW + rbw) + 
                     (double)DICT_SZ*lbw + 16.0*exc + (double)lbw*exc;

        if (bits < best_score){
            best_score = bits;
            best_rbw   = rbw;
            for(int k=0;k<DICT_SZ;++k) best_dict[k]=dict[k];
        }
    }
    D.rightBW = (uint8_t)best_rbw;
    D.leftBW  = (uint8_t)((int)sizeof(T)*8 - best_rbw);
    for(int k=0;k<DICT_SZ;++k) D.dict[k]=best_dict[k];
}

template<typename T>
__device__ inline uint64_t alprd_vector_size_bits_safe(int n, const ALPrdDict<T>& D, int exc_cnt){
    uint64_t base = 57ULL + (uint64_t)n*(DICT_BW + D.rightBW) + 
                    DICT_SZ*D.leftBW + (uint64_t)exc_cnt*(D.leftBW+16);
    return base;
}

template<typename T>
__device__ inline bool alprd_vector_write_safe(SafeBitWriter& bw, const uint64_t* in, int n,
                                              const ALPrdDict<T>& D){
    assert(n <= MAX_VEC);
    
    if (!bw.put1(0)) return false; // useALP=0
    if (!bw.putN((uint64_t)n, 32)) return false;
    if (!bw.putN((uint64_t)D.rightBW, 8)) return false;

    int exc_cnt=0; 
    uint16_t exc_pos[MAX_VEC]; 
    uint32_t exc_left[MAX_VEC];
    uint64_t right_mask = (D.rightBW==64)? ~0ULL : ((1ULL<<D.rightBW)-1ULL);
    uint32_t left_mask  = mask_lo(D.leftBW);

    for(int i=0;i<n;++i){
        uint64_t right = in[i] & right_mask;
        uint32_t left  = (uint32_t)((in[i] >> D.rightBW) & left_mask);
        short idx = DICT_SZ;
        for(int k=0;k<DICT_SZ;++k){ 
            if (D.dict[k]==left){ 
                idx=(short)k; 
                break; 
            } 
        }
        if (idx<DICT_SZ){
            if (!bw.putN((uint64_t)idx, DICT_BW)) return false;
            if (!bw.putN(right, D.rightBW)) return false;
        }else{
            if (!bw.putN(0, DICT_BW)) return false;
            if (!bw.putN(right, D.rightBW)) return false;
            exc_pos[exc_cnt]  = (uint16_t)i;
            exc_left[exc_cnt] = left;
            ++exc_cnt;
        }
    }
    
    for(int k=0;k<DICT_SZ;++k) {
        if (!bw.putN((uint64_t)D.dict[k], D.leftBW)) return false;
    }
    
    if (!bw.putN((uint64_t)exc_cnt, 16)) return false;
    for(int i=0;i<exc_cnt;++i){
        if (!bw.putN((uint64_t)exc_left[i], D.leftBW)) return false;
        if (!bw.putN((uint64_t)exc_pos[i], 16)) return false;
    }
    return true;
}

// 修正的采样函数，使用设备兼容的数组替代std::map
template<typename T>
__device__ void rowgroup_sample_and_find_k_combinations(
    const T* rowgroup_data, 
    int rowgroup_size,
    int vectorSize,
    EFCombination* best_k_combinations,
    int& k_actual,
    CompressionMode& mode
) {
    const int available_alp_vectors = (rowgroup_size + vectorSize - 1) / vectorSize;
    
    // 使用设备兼容的数组替代std::map
    EFPair combinations_keys[19*19];
    int combinations_values[19*19];
    int combinations_count = 0;
    
    size_t best_estimated_compression_size = Constants<T>::RD_SIZE_THRESHOLD_LIMIT + 1;
    // 选择向量idx
    for (size_t smp_n = 0; smp_n < config::ROWGROUP_VECTOR_SAMPLES && 
        smp_n * config::ROWGROUP_SAMPLES_JUMP < available_alp_vectors; smp_n++) {
        
        size_t vector_idx = smp_n * config::ROWGROUP_SAMPLES_JUMP;
        if (vector_idx >= available_alp_vectors) break;
        
        size_t vector_start = vector_idx * vectorSize;
        size_t current_vector_size = (vectorSize < (rowgroup_size - vector_start)) ? 
                                    vectorSize : (rowgroup_size - vector_start);
        
        if (current_vector_size < 2) continue;
        // 没有32个数据就全采样
        const size_t samples_size = (config::SAMPLES_PER_VECTOR < current_vector_size) ? 
                                config::SAMPLES_PER_VECTOR : current_vector_size;
        const int sample_increments = (current_vector_size + samples_size - 1) / samples_size;
        
        T samples[32];
        for (size_t i = 0; i < samples_size; ++i) {
            samples[i] = rowgroup_data[vector_start + i * sample_increments];
        }
        
        uint8_t found_exponent = 0;
        uint8_t found_factor = 0;
        uint64_t sample_estimated_compression_size = Constants<T>::RD_SIZE_THRESHOLD_LIMIT + 1;
        
        for (int8_t exp_ref = Constants<T>::MAX_EXPONENT; exp_ref >= 0; exp_ref--) {
            for (int8_t factor_idx = exp_ref; factor_idx >= 0; factor_idx--) {
                uint16_t exceptions_count = 0;
                uint16_t non_exceptions_count = 0;
                uint32_t estimated_bits_per_value = 0;
                uint64_t estimated_compression_size = 0;
                long long max_encoded_value = LLONG_MIN;
                long long min_encoded_value = LLONG_MAX;
                
                for (size_t i = 0; i < samples_size; i++) {
                    const T actual_value = samples[i];
                    
                    if constexpr (std::is_same_v<T,double>) {
                        double enc = actual_value * D_EXP_ARR[exp_ref] * D_FRAC_ARR[factor_idx];
                        long long encoded_value = fast_round_double(enc);
                        double decoded_value = (double)encoded_value * (double)D_FACT_ARR[factor_idx] * D_FRAC_ARR[exp_ref];
                        
                        if (decoded_value == (double)actual_value) {
                            non_exceptions_count++;
                            if (encoded_value > max_encoded_value) max_encoded_value = encoded_value;
                            if (encoded_value < min_encoded_value) min_encoded_value = encoded_value;
                        } else {
                            exceptions_count++;
                        }
                    } else {
                        float enc = actual_value * F_EXP_ARR[exp_ref] * F_FRAC_ARR[factor_idx];
                        int encoded_value = fast_round_float(enc);
                        float decoded_value = (float)encoded_value * (float)(F_FACT_ARR[factor_idx]) * F_FRAC_ARR[exp_ref];
                        
                        if (decoded_value == (float)actual_value) {
                            non_exceptions_count++;
                            long long encoded_value_ll = (long long)encoded_value;
                            if (encoded_value_ll > max_encoded_value) max_encoded_value = encoded_value_ll;
                            if (encoded_value_ll < min_encoded_value) min_encoded_value = encoded_value_ll;
                        } else {
                            exceptions_count++;
                        }
                    }
                }
                
                if (non_exceptions_count < 2) continue;
                
                unsigned long long range = (unsigned long long)(max_encoded_value - min_encoded_value);
                estimated_bits_per_value = width_needed_unsigned(range);
                estimated_compression_size += samples_size * estimated_bits_per_value;
                estimated_compression_size += exceptions_count * (Constants<T>::EXCEPTION_SIZE + Constants<T>::EXCEPTION_POSITION_SIZE);
                
                if ((estimated_compression_size < sample_estimated_compression_size) ||
                    (estimated_compression_size == sample_estimated_compression_size && 
                    found_exponent < exp_ref) ||
                    ((estimated_compression_size == sample_estimated_compression_size && 
                    found_exponent == exp_ref) && 
                    found_factor < factor_idx)) 
                {
                    sample_estimated_compression_size = estimated_compression_size;
                    found_exponent = exp_ref;
                    found_factor = factor_idx;
                    if (sample_estimated_compression_size < best_estimated_compression_size) {
                        best_estimated_compression_size = sample_estimated_compression_size;
                    }
                }
            }
        }
        
        // 记录找到的最佳组合，使用设备兼容的方法
        EFPair key(found_exponent, found_factor);
        bool found = false;
        for(int i = 0; i < combinations_count; i++) {
            if(combinations_keys[i].equals(key)) {
                combinations_values[i]++;
                found = true;
                break;
            }
        }
        if(!found && combinations_count < 19*19) {
            combinations_keys[combinations_count] = key;
            combinations_values[combinations_count] = 1;
            combinations_count++;
        }
    }
    
    if (best_estimated_compression_size >= Constants<T>::RD_SIZE_THRESHOLD_LIMIT) {
        mode = CompressionMode::ALPrd;
        k_actual = 0;
        return;
    }
    
    mode = CompressionMode::ALP;
    
    // 将组合转换为向量并排序
    EFCombination all_combinations[19*19];
    int num_combinations = 0;
    
    for(int i = 0; i < combinations_count; i++) {
        all_combinations[num_combinations].e = combinations_keys[i].exponent;
        all_combinations[num_combinations].f = combinations_keys[i].factor;
        all_combinations[num_combinations].count = combinations_values[i];
        all_combinations[num_combinations].score = 0;
        num_combinations++;
    }
    
    // 与CPU版本相同的排序逻辑
    for (int i = 0; i < num_combinations - 1; i++) {
        for (int j = i + 1; j < num_combinations; j++) {
            bool should_swap = false;
            if ((all_combinations[j].count > all_combinations[i].count)||
                ((all_combinations[j].count == all_combinations[i].count) && (all_combinations[j].e < all_combinations[i].e))||
                ((all_combinations[j].count == all_combinations[i].count) && (all_combinations[j].e == all_combinations[i].e) && (all_combinations[j].f < all_combinations[i].f))) 
            {
                should_swap = true;
            }
            
            if (should_swap) {
                EFCombination tmp = all_combinations[i];
                all_combinations[i] = all_combinations[j];
                all_combinations[j] = tmp;
            }
        }
    }
    
    k_actual = (config::MAX_K_COMBINATIONS < num_combinations) ? 
               config::MAX_K_COMBINATIONS : num_combinations;
    for (int i = 0; i < k_actual; i++) {
        best_k_combinations[i] = all_combinations[i];
    }
}

// 修正的二级采样函数
template<typename T>
__device__ void vector_choose_from_k_combinations(
    const T* vec_data,
    int vec_size,
    const EFCombination* k_combinations,
    int k,
    uint8_t& best_e,
    uint8_t& best_f,
    short& bitw,
    long long& FOR,
    int& exc
) {
    if (k == 0) {
        // 如果没有k组合，使用完整分析
        alp_vector_choose_best_bits<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
        return;
    }
    
    if (k == 1) {
        best_e = k_combinations[0].e;
        best_f = k_combinations[0].f;
        alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
        return;
    }
    
    // 二级采样
    T samples[32];
    const int sample_count = (config::SAMPLES_PER_VECTOR < vec_size) ? 
                            config::SAMPLES_PER_VECTOR : vec_size;
    const int sample_increments = (vec_size + sample_count - 1) / sample_count;
    
    for (int i = 0; i < sample_count; i++) {
        samples[i] = vec_data[i * sample_increments];
    }
    
    double best_score = 1e30;
    int worse_count = 0;
    
    for (int kid = 0; kid < k; kid++) {
        uint8_t e = k_combinations[kid].e;
        uint8_t f = k_combinations[kid].f;
        
        short test_bitw;
        long long test_FOR;
        int test_exc;
        alp_vector_analyze<T>(samples, sample_count, e, f, test_bitw, test_FOR, test_exc);
        
        int val_bits = std::is_same_v<T,double> ? 64 : 32;
        double score = sample_count * test_bitw + test_exc * (val_bits + 16);
        
        if (score < best_score) {
            best_score = score;
            best_e = e;
            best_f = f;
            worse_count = 0;
        } else {
            worse_count++;
            if (worse_count >= config::SAMPLING_EARLY_EXIT_THRESHOLD) {
                break;
            }
        }
    }
    
    // 在完整向量上分析最终参数
    alp_vector_analyze<T>(vec_data, vec_size, best_e, best_f, bitw, FOR, exc);
}

static constexpr int THREADS_PER_BLOCK = 128;

__global__ void kernel_write_rowgroup_headers(
    const uint64_t* blk_sizes,
    const uint64_t* blk_offsets,
    int numBlocks,
    int vectorSize,
    uint8_t* out_bytes
) {
    int blockId = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockId >= numBlocks) return;
    
    int numVec = (blk_sizes[blockId] + vectorSize - 1) / vectorSize;
    uint64_t bit_offset = blk_offsets[blockId];
    
    SafeBitWriter bw(out_bytes, bit_offset, 8);
    bw.putN((uint64_t)numVec, 8);
}

// ==================== 修改：移除预计算字典，改回原版本方式 ====================
template<typename T>
__global__ void kernel_rowgroup_sampling(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    int numBlocks,
    int vectorSize,
    // 输出每个rowgroup的采样结果 (只保留ALP相关的输出)
    uint8_t* out_modes,           // 0=ALP, 1=ALPrd
    uint8_t* out_k_actual,        // 每个rowgroup的k组合数量
    uint8_t* out_k_combinations   // [numBlocks][5][2] 存储(e,f)组合
) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;
    
    const T* blk = data + blk_starts[blockId];
    int n = (int)blk_sizes[blockId];
    
    // 每个block用一个线程完成采样（避免线程间通信复杂性）
    if (threadIdx.x == 0) {
        EFCombination k_combinations[5];
        int k_actual;
        CompressionMode mode;
        
        // 核心采样逻辑（只执行一次）
        rowgroup_sample_and_find_k_combinations<T>(
            blk, n, vectorSize, k_combinations, k_actual, mode
        );
        
        // 保存采样结果
        out_modes[blockId] = (mode == CompressionMode::ALPrd) ? 1 : 0;
        out_k_actual[blockId] = k_actual;
        
        if (mode == CompressionMode::ALP) {
            // 保存ALP的k组合
            for(int i = 0; i < k_actual; i++) {
                int base_idx = blockId * 5 * 2 + i * 2;
                out_k_combinations[base_idx] = k_combinations[i].e;
                out_k_combinations[base_idx + 1] = k_combinations[i].f;
            }
        }
        // ALPrd模式：不预计算字典，与原版本一致
    }
}

// ==================== 修改：ALPrd每次重新计算字典 ====================
template<typename T>
__global__ void kernel_vector_parameter_selection(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint8_t* modes,
    const uint8_t* k_actual,
    const uint8_t* k_combinations,
    int numBlocks,
    int vectorSize,
    uint64_t total_vectors,
    // 输出每个向量的参数
    uint8_t* vec_e,
    uint8_t* vec_f,
    uint16_t* vec_bitw,
    int64_t* vec_FOR,
    uint16_t* vec_exc_cnt,
    uint64_t* vec_bit_sizes
) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;
    
    const T* blk = data + blk_starts[blockId];
    int n = (int)blk_sizes[blockId];
    int numVec = (n + vectorSize - 1) / vectorSize;
    CompressionMode mode = (modes[blockId] == 1) ? CompressionMode::ALPrd : CompressionMode::ALP;
    
    // 计算全局向量索引基址
    uint64_t vec_base = 0;
    for(int b = 0; b < blockId; b++) {
        vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
    }
    
    if (mode == CompressionMode::ALP) {
        // 重建k组合数组
        EFCombination local_k_combinations[5];
        int local_k_actual = k_actual[blockId];
        
        for(int i = 0; i < local_k_actual; i++) {
            int base_idx = blockId * 5 * 2 + i * 2;
            local_k_combinations[i].e = k_combinations[base_idx];
            local_k_combinations[i].f = k_combinations[base_idx + 1];
        }
        
        // 并行处理每个向量
        for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
            uint64_t global_vec_idx = vec_base + v;
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = (vectorSize < rem ? vectorSize : rem);
            
            uint8_t e, f;
            short bitw;
            long long FOR;
            int exc;
            
            // 对应CPU的 find_best_exponent_factor_from_combinations
            vector_choose_from_k_combinations<T>(
                blk + beg, len,
                local_k_combinations, local_k_actual,
                e, f, bitw, FOR, exc
            );
            
            // 保存参数
            vec_e[global_vec_idx] = e;
            vec_f[global_vec_idx] = f;
            vec_bitw[global_vec_idx] = bitw;
            vec_FOR[global_vec_idx] = FOR;
            vec_exc_cnt[global_vec_idx] = exc;
            vec_bit_sizes[global_vec_idx] = alp_vector_size_bits_safe<T>(len, e, f, bitw, exc);
        }
    } else {
        // ALPrd模式：每个向量重新计算字典（与原版本一致）
        for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
            uint64_t global_vec_idx = vec_base + v;
            int beg = v * vectorSize;
            int rem = n - beg;
            int len = (vectorSize < rem ? vectorSize : rem);
            
            // 转换为raw数据格式
            uint64_t tmp[MAX_VEC];
            for(int i = 0; i < len; i++) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else {
                    uint32_t raw32 = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
                    tmp[i] = raw32;  // 32位数据扩展到64位
                }
            }
            
            // 重新计算字典（与原版本一致）
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            
            // 计算异常数量
            int exc = 0;
            for(int i = 0; i < len; i++) {
                uint32_t left = (uint32_t)((tmp[i] >> D.rightBW) & mask_lo(D.leftBW));
                bool inDict = false;
                for(int k = 0; k < DICT_SZ; k++) {
                    if(D.dict[k] == left) {
                        inDict = true;
                        break;
                    }
                }
                if(!inDict) exc++;
            }
            
            // ALPrd标记
            vec_e[global_vec_idx] = 0xFF;
            vec_f[global_vec_idx] = 0xFF;
            vec_exc_cnt[global_vec_idx] = exc;
            vec_bit_sizes[global_vec_idx] = alprd_vector_size_bits_safe<T>(len, D, exc);
        }
    }
}

// ==================== 修改：ALPrd压缩时重新计算字典 ====================
template<typename T>
__global__ void kernel_compress_and_write(
    const T* data,
    const uint64_t* blk_starts,
    const uint64_t* blk_sizes,
    const uint8_t* modes,
    const uint64_t* vec_bit_offsets,
    const uint8_t* vec_e,
    const uint8_t* vec_f,
    const uint16_t* vec_bitw,
    const int64_t* vec_FOR,
    int numBlocks,
    int vectorSize,
    uint8_t* out_bytes
) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;
    
    const T* blk = data + blk_starts[blockId];
    int n = (int)blk_sizes[blockId];
    int numVec = (n + vectorSize - 1) / vectorSize;
    CompressionMode mode = (modes[blockId] == 1) ? CompressionMode::ALPrd : CompressionMode::ALP;
    
    // 计算全局向量索引基址
    uint64_t vec_base = 0;
    for(int b = 0; b < blockId; b++) {
        vec_base += (blk_sizes[b] + vectorSize - 1) / vectorSize;
    }
    
    // 并行写入每个向量
    for(int v = threadIdx.x; v < numVec; v += blockDim.x) {
        uint64_t global_vec_idx = vec_base + v;
        int beg = v * vectorSize;
        int rem = n - beg;
        int len = (vectorSize < rem ? vectorSize : rem);
        
        uint64_t bit_offset = vec_bit_offsets[global_vec_idx];
        SafeBitWriter bw(out_bytes, bit_offset, 100000); // 充足的缓冲区
        
        if(mode == CompressionMode::ALP) {
            // ALP压缩写入
            uint8_t e = vec_e[global_vec_idx];
            uint8_t f = vec_f[global_vec_idx];
            short bitw = vec_bitw[global_vec_idx];
            long long FOR = vec_FOR[global_vec_idx];
            
            alp_vector_write_safe<T>(bw, blk + beg, len, e, f, bitw, FOR);
            
        } else {
            // ALPrd压缩写入：重新计算字典（与原版本一致）
            uint64_t tmp[MAX_VEC];
            for(int i = 0; i < len; i++) {
                if constexpr (std::is_same_v<T,double>) 
                    tmp[i] = *reinterpret_cast<const uint64_t*>(&blk[beg+i]);
                else {
                    uint32_t raw32 = *reinterpret_cast<const uint32_t*>(&blk[beg+i]);
                    tmp[i] = raw32;  // 32位数据扩展到64位
                }
            }
            
            // 重新计算字典（与原版本一致）
            ALPrdDict<T> D;
            alprd_find_best<T>(tmp, len, D);
            alprd_vector_write_safe<T>(bw, tmp, len, D);
        }
    }
}

template<typename T>
__global__ void kernel_decompress_debug(const uint8_t* bytes,
                                        const uint64_t* blk_starts_bits,
                                        const uint64_t* blk_bits,
                                        const uint64_t* out_starts,
                                        const int vectorSize,
                                        T* out_data, 
                                        int numBlocks) {
    int blockId = blockIdx.x;
    if (blockId >= numBlocks) return;

    uint64_t bit_offset = blk_starts_bits[blockId];
    BitReader br{bytes, bit_offset};
    
    int numVec = (int)br.getN(8);

    if (numVec <= 0 || numVec > 10000) {
        return;
    }

    uint64_t out_pos = out_starts[blockId];
    
    for(int v = 0; v < numVec; v++) {
        int useALP = br.get1();

        if (useALP) {
            uint8_t e = (uint8_t)br.getN(8);
            uint8_t f = (uint8_t)br.getN(8);
            short bitw = (short)br.getN(16);
            long long FOR = (long long)br.getN(64);
            int n = (int)br.getN(32);

            if (n <= 0 || n > MAX_VEC) return;

            for(int k = 0; k < n; k++) {
                uint64_t enc = br.getN(bitw);
                long long I = FOR + (long long)enc;
                
                if constexpr (std::is_same_v<T,double>) {
                    double dec = (double)I * (1.0/(double)D_FRAC_ARR[f]) * D_FRAC_ARR[e];
                    out_data[out_pos + k] = (T)dec;
                } else {
                    float dec = (float)I * (float)(F_FACT_ARR[f])* F_FRAC_ARR[e];
                    out_data[out_pos + k] = (T)dec;
                }
            }

            int exc = (int)br.getN(16);
            for(int t = 0; t < exc; t++) {
                uint64_t raw = std::is_same_v<T,double> ? br.getN(64) : br.getN(32);
                int pos = (int)br.getN(16);
                
                if (pos < n) {
                    if constexpr (std::is_same_v<T,double>) {
                        double val = *reinterpret_cast<double*>(&raw);
                        out_data[out_pos + pos] = (T)val;
                    } else {
                        uint32_t rv = (uint32_t)raw;
                        float val = *reinterpret_cast<float*>(&rv);
                        out_data[out_pos + pos] = (T)val;
                    }
                }
            }
            out_pos += n;
        } else {
            int n = (int)br.getN(32);
            if (n <= 0 || n > MAX_VEC) return;

            uint8_t rbw = (uint8_t)br.getN(8);
            uint64_t right[MAX_VEC]; 
            uint16_t leftIdx[MAX_VEC];
            
            for(int k = 0; k < n; k++) {
                leftIdx[k] = (uint16_t)br.getN(DICT_BW);
                right[k] = br.getN(rbw);
            }

            uint8_t lbw = (uint8_t)sizeof(T)*8 - rbw;
            uint64_t dict[DICT_SZ];
            for(int k = 0; k < DICT_SZ; k++) {
                dict[k] = br.getN(lbw);
            }

            int exc = (int)br.getN(16);
            uint16_t exc_pos[MAX_VEC]; 
            uint64_t exc_left[MAX_VEC];
            for(int t = 0; t < exc; t++) {
                exc_left[t] = br.getN(lbw);
                exc_pos[t] = (uint16_t)br.getN(16);
            }

            for(int k = 0; k < n; k++) {
                uint64_t left = (leftIdx[k] < DICT_SZ) ? dict[leftIdx[k]] : 0ULL;
                uint64_t raw = (left << rbw) | right[k];
                
                if constexpr (std::is_same_v<T,double>) {
                    double val = *reinterpret_cast<double*>(&raw);
                    out_data[out_pos + k] = (T)val;
                } else {
                    uint32_t r32 = (uint32_t)raw;
                    float val = *reinterpret_cast<float*>(&r32);
                    out_data[out_pos + k] = (T)val;
                }
            }

            for(int t = 0; t < exc; t++) {
                int p = exc_pos[t];
                uint64_t raw = (exc_left[t] << rbw) | right[p];
                
                if (p < n) {
                    if constexpr (std::is_same_v<T,double>) {
                        double val = *reinterpret_cast<double*>(&raw);
                        out_data[out_pos + p] = (T)val;
                    } else {
                        uint32_t r32 = (uint32_t)raw;
                        float val = *reinterpret_cast<float*>(&r32);
                        out_data[out_pos + p] = (T)val;
                    }
                }
            }
            out_pos += n;
        }
    }
}

//  ==================== 核心设备端压缩实现（修改版） ====================
template<typename T>
static CompressedDevice compress_core_device(const T* d_data, size_t n, const Params& p, cudaStream_t stream) {
    CompressedDevice c;
    if (n == 0) { 
        c.vectorSize = p.vectorSize; 
        return c; 
    }
    
    const int V = p.vectorSize;
    const int B = p.blockSize > 0 ? p.blockSize : (int)n;
    const int numBlocks = (int)((n + B - 1) / B);
    
    // 计算块信息
    uint64_t total_vectors = 0;
    std::vector<uint64_t> h_starts(numBlocks), h_sizes(numBlocks);
    size_t pos = 0;
    for(int i = 0; i < numBlocks; i++) {
        h_starts[i] = pos;
        uint64_t sz = std::min<uint64_t>(B, n - pos);
        h_sizes[i] = sz;
        total_vectors += (sz + V - 1) / V;
        pos += sz;
    }
    
    // 拷贝块信息到设备
    uint64_t *d_starts = nullptr, *d_sizes = nullptr;
    CUDA_CHECK(cudaMalloc(&d_starts, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_sizes, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_starts, h_starts.data(), numBlocks * sizeof(uint64_t), 
                              cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_sizes, h_sizes.data(), numBlocks * sizeof(uint64_t), 
                              cudaMemcpyHostToDevice, stream));
    
    // ==================== 阶段1：采样（移除ALPrd预计算） ====================
    uint8_t* d_modes = nullptr;
    uint8_t* d_k_actual = nullptr;
    uint8_t* d_k_combinations = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_modes, numBlocks));
    CUDA_CHECK(cudaMalloc(&d_k_actual, numBlocks));
    CUDA_CHECK(cudaMalloc(&d_k_combinations, numBlocks * 5 * 2)); // [blocks][5 combinations][e,f]
    
    dim3 sampling_grid(numBlocks);
    dim3 sampling_block(32); // 每个rowgroup用少量线程
    
    kernel_rowgroup_sampling<T><<<sampling_grid, sampling_block, 0, stream>>>(
        d_data, d_starts, d_sizes, numBlocks, V,
        d_modes, d_k_actual, d_k_combinations
    );
    
    // ============== 记录rowgroup采样结果 ==============
    if (p.enable_recording && get_gpu_recording_enabled()) {
        // 拷贝GPU结果到主机端进行记录
        std::vector<uint8_t> h_modes(numBlocks);
        std::vector<uint8_t> h_k_actual(numBlocks);
        std::vector<uint8_t> h_k_combinations(numBlocks * 5 * 2);
        
        CUDA_CHECK(cudaMemcpyAsync(h_modes.data(), d_modes, numBlocks, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_k_actual.data(), d_k_actual, numBlocks, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_k_combinations.data(), d_k_combinations, numBlocks * 5 * 2, 
                                  cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 为每个rowgroup记录k组合
        for(int blockId = 0; blockId < numBlocks; blockId++) {
            if(h_modes[blockId] == 0) { // ALP模式
                std::vector<std::pair<std::pair<int, int>, int>> combinations;
                int k_count = h_k_actual[blockId];
                for(int i = 0; i < k_count; i++) {
                    int base_idx = blockId * 5 * 2 + i * 2;
                    int e = h_k_combinations[base_idx];
                    int f = h_k_combinations[base_idx + 1];
                    combinations.push_back({{e, f}, 1}); // count设为1
                }
                record_gpu_rowgroup_combinations(combinations);
            }
        }
    }
    
    // ==================== 阶段2：向量参数选择 ====================
    uint8_t* d_vec_e = nullptr;
    uint8_t* d_vec_f = nullptr;
    uint16_t* d_vec_bitw = nullptr;
    int64_t* d_vec_FOR = nullptr;
    uint16_t* d_vec_exc_cnt = nullptr;
    uint64_t* d_vec_bit_sizes = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_vec_e, total_vectors));
    CUDA_CHECK(cudaMalloc(&d_vec_f, total_vectors));
    CUDA_CHECK(cudaMalloc(&d_vec_bitw, total_vectors * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_FOR, total_vectors * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_exc_cnt, total_vectors * sizeof(uint16_t)));
    CUDA_CHECK(cudaMalloc(&d_vec_bit_sizes, total_vectors * sizeof(uint64_t)));
    
    dim3 param_grid(numBlocks);
    dim3 param_block(THREADS_PER_BLOCK);
    
    kernel_vector_parameter_selection<T><<<param_grid, param_block, 0, stream>>>(
        d_data, d_starts, d_sizes,
        d_modes, d_k_actual, d_k_combinations,
        numBlocks, V, total_vectors,
        d_vec_e, d_vec_f, d_vec_bitw, d_vec_FOR, d_vec_exc_cnt, d_vec_bit_sizes
    );
    
    // ============== 记录向量(e,f)选择 ==============
    if (p.enable_recording && get_gpu_recording_enabled()) {
        // 拷贝向量参数到主机端
        std::vector<uint8_t> h_vec_e(total_vectors);
        std::vector<uint8_t> h_vec_f(total_vectors);
        std::vector<uint8_t> h_modes_for_vectors(numBlocks);
        
        CUDA_CHECK(cudaMemcpyAsync(h_vec_e.data(), d_vec_e, total_vectors, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_vec_f.data(), d_vec_f, total_vectors, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_modes_for_vectors.data(), d_modes, numBlocks, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 记录每个向量的(e,f)选择
        uint64_t vec_idx = 0;
        for(int blockId = 0; blockId < numBlocks; blockId++) {
            int numVec = (h_sizes[blockId] + V - 1) / V;
            for(int v = 0; v < numVec; v++) {
                if(h_modes_for_vectors[blockId] == 0) { // ALP模式
                    uint8_t e = h_vec_e[vec_idx];
                    uint8_t f = h_vec_f[vec_idx];
                    record_gpu_vector_ef(e, f);
                }
                // ALPrd模式的向量用0xFF标记，不记录
                vec_idx++;
            }
        }
    }
    
    // 拷贝bit_sizes用于计算偏移
    std::vector<uint64_t> h_vec_bit_sizes(total_vectors);
    CUDA_CHECK(cudaMemcpyAsync(h_vec_bit_sizes.data(), d_vec_bit_sizes, 
                              total_vectors * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // ==================== 计算偏移量 ====================
    std::vector<uint64_t> h_block_offsets(numBlocks);
    std::vector<uint64_t> h_vec_offsets(total_vectors);
    
    uint64_t block_offset = 0;
    uint64_t vec_idx = 0;
    
    for(int b = 0; b < numBlocks; b++) {
        h_block_offsets[b] = block_offset;
        uint64_t current_offset = block_offset + 8; // block header
        
        int numVec = (h_sizes[b] + V - 1) / V;
        for(int v = 0; v < numVec; v++) {
            h_vec_offsets[vec_idx] = current_offset;
            current_offset += h_vec_bit_sizes[vec_idx];
            vec_idx++;
        }
        
        // 块级padding
        uint64_t block_bits = current_offset - block_offset;
        uint64_t padding = (32 - (block_bits & 31)) & 31;
        block_offset = current_offset + padding;
    }
    
    const uint64_t total_bytes = (block_offset + 7) / 8;
    
    // ==================== 阶段3：压缩写入 ====================
    CUDA_CHECK(cudaMalloc(&c.d_data, total_bytes));
    CUDA_CHECK(cudaMemsetAsync(c.d_data, 0, total_bytes, stream));
    c.data_size = total_bytes;
    
    // 拷贝偏移量到设备
    uint64_t* d_vec_offsets = nullptr;
    uint64_t* d_block_offsets = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vec_offsets, total_vectors * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_block_offsets, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpyAsync(d_vec_offsets, h_vec_offsets.data(), 
                              total_vectors * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_block_offsets, h_block_offsets.data(), 
                              numBlocks * sizeof(uint64_t), cudaMemcpyHostToDevice, stream));
    
    // 写入块头
    kernel_write_rowgroup_headers<<<(numBlocks + 255) / 256, 256, 0, stream>>>(
        d_sizes, d_block_offsets, numBlocks, V, c.d_data
    );
    
    // 压缩写入数据
    kernel_compress_and_write<T><<<param_grid, param_block, 0, stream>>>(
        d_data, d_starts, d_sizes,
        d_modes, d_vec_offsets, d_vec_e, d_vec_f, d_vec_bitw, d_vec_FOR,
        numBlocks, V, c.d_data
    );
    
    // 设置元数据
    c.offsets = std::move(h_block_offsets);
    c.bit_sizes.resize(numBlocks);
    vec_idx = 0;
    for(int b = 0; b < numBlocks; b++) {
        uint64_t block_bits = 8; // header
        int numVec = (h_sizes[b] + V - 1) / V;
        for(int v = 0; v < numVec; v++) {
            block_bits += h_vec_bit_sizes[vec_idx++];
        }
        c.bit_sizes[b] = block_bits;
    }
    c.elem_counts.assign(h_sizes.begin(), h_sizes.end());
    c.vectorSize = V;
    
    // ==================== 清理内存 ====================
    CUDA_CHECK(cudaFree(d_block_offsets));
    CUDA_CHECK(cudaFree(d_vec_offsets));
    CUDA_CHECK(cudaFree(d_vec_bit_sizes));
    CUDA_CHECK(cudaFree(d_vec_exc_cnt));
    CUDA_CHECK(cudaFree(d_vec_FOR));
    CUDA_CHECK(cudaFree(d_vec_bitw));
    CUDA_CHECK(cudaFree(d_vec_f));
    CUDA_CHECK(cudaFree(d_vec_e));
    CUDA_CHECK(cudaFree(d_k_combinations));
    CUDA_CHECK(cudaFree(d_k_actual));
    CUDA_CHECK(cudaFree(d_modes));
    CUDA_CHECK(cudaFree(d_sizes));
    CUDA_CHECK(cudaFree(d_starts));
    
    return c;
}

// ==================== 其余实现保持不变 ====================

template<typename T>
static Compressed compress_impl(const T* h_data, size_t n, const Params& p) {
    if (n == 0) {
        Compressed c;
        c.vectorSize = p.vectorSize;
        return c;
    }
    
    // 启用记录功能（如果参数中启用）
    if (p.enable_recording) {
        enable_gpu_alp_recording(true);
    }
    
    // 分配设备内存并拷贝数据
    T* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));
    
    // 调用核心设备端实现
    CompressedDevice device_result = compress_core_device<T>(d_data, n, p, 0);
    
    // 转换为主机端格式
    Compressed host_result = device_to_host(device_result);
    
    // 清理
    CUDA_CHECK(cudaFree(d_data));
    
    return host_result;
}

template<typename T>
static CompressedDevice compress_device_impl(const T* d_data, size_t n, const Params& p, cudaStream_t stream) {
    // 启用记录功能（如果参数中启用）
    if (p.enable_recording) {
        enable_gpu_alp_recording(true);
    }
    
    // 直接调用核心实现
    return compress_core_device<T>(d_data, n, p, stream);
}

template<typename T>
static void decompress_core_device(const uint8_t* d_compressed_data,
                                  const std::vector<uint64_t>& offsets,
                                  const std::vector<uint64_t>& bit_sizes,
                                  const std::vector<uint32_t>& elem_counts,
                                  int vectorSize,
                                  T* d_output,
                                  size_t output_size,
                                  cudaStream_t stream) {
    if (output_size == 0) return;
    
    const int numBlocks = (int)offsets.size();
    if (numBlocks == 0) return;
    
    // 分配并拷贝元数据到设备
    uint64_t *d_offsets = nullptr, *d_bit_sizes = nullptr, *d_output_starts = nullptr;
    CUDA_CHECK(cudaMalloc(&d_offsets, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_bit_sizes, numBlocks * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_output_starts, numBlocks * sizeof(uint64_t)));
    
    CUDA_CHECK(cudaMemcpyAsync(d_offsets, offsets.data(), numBlocks * sizeof(uint64_t), 
                              cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_bit_sizes, bit_sizes.data(), numBlocks * sizeof(uint64_t), 
                              cudaMemcpyHostToDevice, stream));
    
    // 计算输出起始位置
    std::vector<uint64_t> output_starts(numBlocks);
    uint64_t accumulated = 0;
    for(int i = 0; i < numBlocks; i++) {
        output_starts[i] = accumulated;
        accumulated += elem_counts[i];
    }
    
    CUDA_CHECK(cudaMemcpyAsync(d_output_starts, output_starts.data(), numBlocks * sizeof(uint64_t), 
                              cudaMemcpyHostToDevice, stream));
    
    // 初始化输出数据（用于调试）
    CUDA_CHECK(cudaMemsetAsync(d_output, 0, output_size * sizeof(T), stream));
    
    // 启动解压kernel
    dim3 grid(numBlocks);
    dim3 block(1); // 每个block用单线程处理一个压缩块
    
    kernel_decompress_debug<T><<<grid, block, 0, stream>>>(
        d_compressed_data,
        d_offsets,
        d_bit_sizes, 
        d_output_starts,
        vectorSize,
        d_output,
        numBlocks
    );
    
    // 清理设备内存
    CUDA_CHECK(cudaFree(d_output_starts));
    CUDA_CHECK(cudaFree(d_bit_sizes));
    CUDA_CHECK(cudaFree(d_offsets));
}

// 主机端解压实现
template<typename T>
static void decompress_impl(const Compressed& compressed, 
                              T* h_output, 
                              size_t output_size, 
                              const Params& params) {
    if (output_size == 0 || compressed.empty()) return;
    
    // 分配设备内存
    uint8_t* d_compressed_data = nullptr;
    T* d_output = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_compressed_data, compressed.data.size()));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(T)));
    
    // 拷贝压缩数据到设备
    CUDA_CHECK(cudaMemcpy(d_compressed_data, compressed.data.data(), 
                          compressed.data.size(), cudaMemcpyHostToDevice));
    
    // 调用核心解压实现
    decompress_core_device<T>(
        d_compressed_data,
        compressed.offsets,
        compressed.bit_sizes,
        compressed.elem_counts,
        compressed.vectorSize,
        d_output,
        output_size,
        0  // 默认stream
    );
    
    // 等待GPU完成并拷贝结果回主机
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // 清理设备内存
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_compressed_data));
}

// 设备端解压实现
template<typename T>
static void decompress_device_impl(const CompressedDevice& compressed_device, 
                                        T* d_output, 
                                        size_t output_size, 
                                        const Params& params, 
                                        cudaStream_t stream) {
    // 直接调用核心实现，数据已在设备上
    decompress_core_device<T>(
        compressed_device.d_data,
        compressed_device.offsets,
        compressed_device.bit_sizes,
        compressed_device.elem_counts,
        compressed_device.vectorSize,
        d_output,
        output_size,
        stream
    );
}

// 主机端API
Compressed compress_double(const double* data, size_t n, const Params& p) { 
    return compress_impl<double>(data, n, p); 
}
Compressed compress_float(const float* data, size_t n, const Params& p) { 
    return compress_impl<float>(data, n, p); 
}
void decompress_double(const Compressed& c, double* out, size_t n, const Params& p) { 
    decompress_impl<double>(c, out, n, p); 
}
void decompress_float(const Compressed& c, float* out, size_t n, const Params& p) { 
    decompress_impl<float>(c, out, n, p); 
}

// 设备端API
CompressedDevice compress_double_device(const double* d_data, size_t n, const Params& p, cudaStream_t stream) {
    return compress_device_impl<double>(d_data, n, p, stream);
}

CompressedDevice compress_float_device(const float* d_data, size_t n, const Params& p, cudaStream_t stream) {
    return compress_device_impl<float>(d_data, n, p, stream);
}

void decompress_double_device(const CompressedDevice& c, double* d_out, size_t n, const Params& p, cudaStream_t stream) {
    decompress_device_impl<double>(c, d_out, n, p, stream);
}

void decompress_float_device(const CompressedDevice& c, float* d_out, size_t n, const Params& p, cudaStream_t stream) {
    decompress_device_impl<float>(c, d_out, n, p, stream);
}

// 数据交互函数
Compressed device_to_host(const CompressedDevice& cd) {
    Compressed c;
    c.data.resize(cd.data_size);
    cudaMemcpy(c.data.data(), cd.d_data, cd.data_size, cudaMemcpyDeviceToHost);
    c.offsets = cd.offsets;
    c.bit_sizes = cd.bit_sizes;
    c.elem_counts = cd.elem_counts;
    c.vectorSize = cd.vectorSize;
    return c;
}

CompressedDevice host_to_device(const Compressed& c) {
    CompressedDevice cd;
    cd.data_size = c.data.size();
    cudaMalloc(&cd.d_data, cd.data_size);
    cudaMemcpy(cd.d_data, c.data.data(), cd.data_size, cudaMemcpyHostToDevice);
    cd.offsets = c.offsets;
    cd.bit_sizes = c.bit_sizes;
    cd.elem_counts = c.elem_counts;
    cd.vectorSize = c.vectorSize;
    return cd;
}

} // namespace alp_gpu
