// Standalone exact differential-test entry point; not called by scoring.
__global__ void lu_check_kernel(const double* matrices, const double* rhs,
    int n, int cooperative, double* result, int* integers) {
    __shared__ Scratch s;
    const int h = blockIdx.x, lane = threadIdx.x;
    if (lane == 0) s = Scratch{};
    __syncthreads();
    if (lane == 0) s.dimension = n;
    for (int k = lane; k < n*n; k += blockDim.x)
        s.normal[(k/n)*MAX_D+k%n] = matrices[h*n*n+k];
    if (lane < n) s.rhs[lane] = rhs[h*n+lane];
    __syncthreads();
    if (cooperative) {
        if (lane < 32) solve_normal_warp(s, lane);
    } else if (lane == 0) solve_normal(s);
    __syncthreads();
    if (lane == 0) { result[h*(1+MAX_D+MAX_D*MAX_D)] = s.logdet_normal; integers[h*(1+MAX_D)] = s.status; }
    if (lane < MAX_D) {
        result[h*(1+MAX_D+MAX_D*MAX_D)+1+lane] = s.solution[lane];
        integers[h*(1+MAX_D)+1+lane] = s.pivot[lane];
    }
    for (int k = lane; k < MAX_D*MAX_D; k += blockDim.x)
        result[h*(1+MAX_D+MAX_D*MAX_D)+1+MAX_D+k] = s.lu[k];
}

extern "C" __declspec(dllexport) int rbpf_lu_check(const double* matrices, const double* rhs,
    int count, int n, int cooperative, double* result, int* integers) {
    if (count < 1 || count > 8192 || n < 1 || n > MAX_D || (cooperative != 0 && cooperative != 1)) return -1;
    double *dm=nullptr, *dr=nullptr, *dout=nullptr;
    int* di=nullptr;
    const size_t mb=size_t(count)*n*n*sizeof(double), rb=size_t(count)*n*sizeof(double);
    const size_t ob=size_t(count)*(1+MAX_D+MAX_D*MAX_D)*sizeof(double), ib=size_t(count)*(1+MAX_D)*sizeof(int);
    cudaError_t error=cudaMalloc(&dm,mb);
    if (error == cudaSuccess) error=cudaMalloc(&dr,rb);
    if (error == cudaSuccess) error=cudaMalloc(&dout,ob);
    if (error == cudaSuccess) error=cudaMalloc(&di,ib);
    if (error == cudaSuccess) error=cudaMemcpy(dm,matrices,mb,cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error=cudaMemcpy(dr,rhs,rb,cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
        lu_check_kernel<<<count,128>>>(dm,dr,n,cooperative,dout,di);
        error=cudaGetLastError();
    }
    if (error == cudaSuccess) error=cudaMemcpy(result,dout,ob,cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) error=cudaMemcpy(integers,di,ib,cudaMemcpyDeviceToHost);
    cudaFree(dm); cudaFree(dr); cudaFree(dout); cudaFree(di);
    return int(error);
}
