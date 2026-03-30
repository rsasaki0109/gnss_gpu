#include "gnss_gpu/gnss_gpu.h"
#include <cstdio>

namespace gnss_gpu {

__global__ void hello_kernel() {
  printf("Hello from gnss_gpu (GPU thread %d)\n", threadIdx.x);
}

void hello() {
  hello_kernel<<<1, 1>>>();
  cudaDeviceSynchronize();
}

}  // namespace gnss_gpu
