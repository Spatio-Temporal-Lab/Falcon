//
// Created by lz on 24-9-26.
//
// GDFCompressor.cuh

#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <math.h>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <thrust/device_vector.h>

class GDFCompressor {
public:
    GDFCompressor(){}

    void compress(const std::vector<double>& input, std::vector<unsigned char>& output);

private:

    void setupDeviceMemory(
    const std::vector<double>& input,
    double*& d_input,
    unsigned char*& d_output,
    uint64_t*& d_bitSizes
    ); 
    void freeDeviceMemory(
    double* d_input,
    unsigned char* d_output,
    uint64_t* d_bitSizes
    );

};
