#pragma once
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "CUDA error at %s:%d: %s", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        fprintf(stderr, "%s\n", msg); \
        throw std::runtime_error(msg); \
    } \
} while(0)

#define CUDA_CHECK_LAST() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        char msg[512]; \
        snprintf(msg, sizeof(msg), "CUDA kernel error at %s:%d: %s", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        fprintf(stderr, "%s\n", msg); \
        throw std::runtime_error(msg); \
    } \
} while(0)
