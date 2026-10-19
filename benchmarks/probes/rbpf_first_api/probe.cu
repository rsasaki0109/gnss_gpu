// Experimental fused RBPF likelihood, independent of the shipped runtime.
// FP64 only, all rows/hypotheses and both original IRLS passes retained.
// CPU and GPU share formulas; their rounding need not match NumPy/LAPACK.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <new>
#include <cstring>

constexpr int MAX_ROWS = 128;
constexpr int MAX_GROUPS = 8;
constexpr int MAX_D = 3 + MAX_GROUPS;
constexpr int DIAG_SIZE = 3 + MAX_D + MAX_D * MAX_D;
constexpr double LOG2PI = 1.8378770664093453;

struct Scratch {
    double A[MAX_ROWS * MAX_D], Aw[MAX_ROWS * MAX_D];
    double b[MAX_ROWS], bw[MAX_ROWS], sigma[MAX_ROWS], rowll[MAX_ROWS];
    double prior[MAX_D * MAX_D], normal[MAX_D * MAX_D], lu[MAX_D * MAX_D];
    double rhs[MAX_D], solution[MAX_D], temporary[MAX_D];
    int pivot[MAX_D], kept[MAX_ROWS], group[MAX_ROWS];
    int count, dimension, status;
    double constant, logdet_prior, logdet_normal, likelihood;
};

// Partial-pivot LU also handles the full, slightly asymmetric floating-point
// normal matrix. Do not average triangles or add an unrequested jitter.
__host__ __device__ bool decompose(double* a, int n, int* pivot, double& logdet, int& sign) {
    logdet = 0.0;
    sign = 1;
    for (int j = 0; j < n; ++j) {
        int best = j;
        double magnitude = fabs(a[j*n+j]);
        for (int i = j+1; i < n; ++i) {
            const double candidate = fabs(a[i*n+j]);
            if (candidate > magnitude) { best = i; magnitude = candidate; }
        }
        if (!(magnitude > 0.0) || !isfinite(magnitude)) return false;
        pivot[j] = best;
        if (best != j) {
            for (int k = 0; k < n; ++k) {
                const double saved = a[j*n+k];
                a[j*n+k] = a[best*n+k]; a[best*n+k] = saved;
            }
            sign = -sign;
        }
        const double diagonal = a[j*n+j];
        if (diagonal < 0.0) sign = -sign;
        logdet += log(fabs(diagonal));
        for (int i = j+1; i < n; ++i) {
            a[i*n+j] /= diagonal;
            for (int k = j+1; k < n; ++k) a[i*n+k] -= a[i*n+j] * a[j*n+k];
        }
    }
    return isfinite(logdet);
}

__host__ __device__ void solve_lu(const double* a, int n, const int* pivot, double* b) {
    for (int i = 0; i < n; ++i) {
        if (pivot[i] != i) { const double saved = b[i]; b[i] = b[pivot[i]]; b[pivot[i]] = saved; }
    }
    for (int i = 0; i < n; ++i) for (int k = 0; k < i; ++k) b[i] -= a[i*n+k] * b[k];
    for (int i = n-1; i >= 0; --i) {
        for (int k = i+1; k < n; ++k) b[i] -= a[i*n+k] * b[k];
        b[i] /= a[i*n+i];
    }
}

// rows: Jx,Jy,Jz,v0,sigma,wavelength,bmeas; meta: isCP,pair,group.
// pair: y,Qdiag. hypothesis: centre3, original prior covariance9.
// config: beta,marg_uniform,huber_c,sigma_group,eps,mix_CP,mix_PR.
__host__ __device__ void initialize(Scratch& s, const double* rows, const int* meta,
    const double* dd, const double* hypothesis, const double* cfg, int m) {
    s.count = 0; s.dimension = 3; s.status = 0; s.constant = 0.0;
    s.logdet_prior = 0.0; s.logdet_normal = 0.0; s.likelihood = -1e12;
    int groups[MAX_GROUPS];
    int used = 0;
    for (int r = 0; r < m; ++r) {
        const bool cp = meta[3*r] != 0;
        if (cp && !isfinite(dd[meta[3*r+1]]) && cfg[1] != 0.0) {
            s.constant += -log(fmax(rows[7*r+5], 1e-3));
            continue;
        }
        s.kept[s.count] = r;
        s.group[s.count] = -1;
        if (cp) {
            const int group = meta[3*r+2];
            int index = 0;
            while (index < used && groups[index] != group) ++index;
            if (index == used) {
                if (used == MAX_GROUPS) { s.status = 2; return; }
                groups[used++] = group;
            }
            s.group[s.count] = index;
        }
        ++s.count;
    }
    s.dimension += used;
    for (int i = 0; i < MAX_D*MAX_D; ++i) s.prior[i] = 0.0;
    for (int i = 0; i < 9; ++i) s.lu[i] = hypothesis[3+i];
    int sign = 0;
    if (!decompose(s.lu, 3, s.pivot, s.logdet_prior, sign) || sign <= 0) { s.status = 1; return; }
    for (int col = 0; col < 3; ++col) {
        for (int i = 0; i < 3; ++i) s.temporary[i] = i == col ? 1.0 : 0.0;
        solve_lu(s.lu, 3, s.pivot, s.temporary);
        for (int i = 0; i < 3; ++i) s.prior[i*MAX_D+col] = s.temporary[i];
    }
    const double group_variance = cfg[3] * cfg[3];
    for (int i = 3; i < s.dimension; ++i) s.prior[i*MAX_D+i] = 1.0 / group_variance;
    s.logdet_prior += 2.0 * used * log(cfg[3]);
}

__host__ __device__ void prepare_row(Scratch& s, int r, const double* rows, const int* meta,
    const double* pairs, const double* dd, const double* hypothesis, const double* cfg) {
    const int source = s.kept[r];
    const double* row = rows + 7*source;
    const bool cp = meta[3*source] != 0;
    double value = row[3], sigma = row[4];
    if (cp) {
        const int p = meta[3*source+1];
        if (isfinite(dd[p])) {
            value = row[5] * (row[6] - dd[p]);
            sigma /= sqrt(cfg[0]);
        } else {
            value = row[5] * (row[6] - pairs[2*p]);
            sigma = sqrt(sigma*sigma + row[5]*row[5] * (fmax(pairs[2*p+1], 0.0) + 1.0/12.0)) / sqrt(cfg[0]);
        }
    }
    double shift = 0.0;
    for (int j = 0; j < 3; ++j) shift += row[j] * hypothesis[j];
    value -= shift;
    s.sigma[r] = sigma;
    s.b[r] = value / sigma;
    s.bw[r] = s.b[r];
    for (int j = 0; j < s.dimension; ++j) {
        double a = j < 3 ? row[j] / sigma : 0.0;
        if (j >= 3 && j-3 == s.group[r]) a = 1.0 / sigma;
        s.A[r*MAX_D+j] = a;
        s.Aw[r*MAX_D+j] = a;
    }
}

__host__ __device__ void normal_entry(Scratch& s, int index) {
    const int i = index / s.dimension, j = index % s.dimension;
    double total = 0.0;
    for (int r = 0; r < s.count; ++r) total += s.Aw[r*MAX_D+i] * s.Aw[r*MAX_D+j];
    s.normal[i*MAX_D+j] = s.prior[i*MAX_D+j] + total;
    if (j == 0) {
        total = 0.0;
        for (int r = 0; r < s.count; ++r) total += s.Aw[r*MAX_D+i] * s.bw[r];
        s.rhs[i] = total;
    }
}

__host__ __device__ void solve_normal(Scratch& s) {
    const int n = s.dimension;
    for (int i = 0; i < n; ++i) {
        s.solution[i] = s.rhs[i];
        for (int j = 0; j < n; ++j) s.lu[i*n+j] = s.normal[i*MAX_D+j];
    }
    int sign = 0;
    if (!decompose(s.lu, n, s.pivot, s.logdet_normal, sign) || sign <= 0) { s.status = 1; return; }
    solve_lu(s.lu, n, s.pivot, s.solution);
    for (int i = 0; i < n; ++i) if (!isfinite(s.solution[i])) s.status = 3;
}

__host__ __device__ double residual(const Scratch& s, int r) {
    double value = 0.0;
    for (int j = 0; j < s.dimension; ++j) value += s.A[r*MAX_D+j] * s.solution[j];
    return s.b[r] - value;
}

__host__ __device__ void reweight_row(Scratch& s, int r, const double* cfg) {
    const double weight = fmin(1.0, cfg[2] / fmax(fabs(residual(s,r)), 1e-12));
    const double sw = sqrt(weight);
    s.bw[r] = s.b[r] * sw;
    for (int j = 0; j < s.dimension; ++j) s.Aw[r*MAX_D+j] = s.A[r*MAX_D+j] * sw;
}

__host__ __device__ void likelihood_row(Scratch& s, int r, const int* meta, const double* cfg) {
    const double value = residual(s,r), sigma = s.sigma[r];
    const double outlier = fmax(meta[3*s.kept[r]] ? cfg[5] : cfg[6], sigma);
    const double base = log(1.0-cfg[4]) - 0.5*value*value - log(sigma);
    const double scaled = value*sigma/outlier;
    const double tail = log(cfg[4]) - 0.5*scaled*scaled - log(outlier);
    s.rowll[r] = fmax(base,tail) + log1p(exp(-fabs(base-tail))) - 0.5*LOG2PI;
}

__host__ __device__ void finish(Scratch& s) {
    double total = 0.0, quadratic = 0.0;
    for (int r = 0; r < s.count; ++r) total += s.rowll[r];
    for (int i = 0; i < s.dimension; ++i) {
        double value = 0.0;
        for (int j = 0; j < s.dimension; ++j) value += s.prior[i*MAX_D+j] * s.solution[j];
        quadratic += s.solution[i] * value;
    }
    s.likelihood = total + s.constant - 0.5*quadratic - 0.5*s.logdet_normal - 0.5*s.logdet_prior;
    if (!isfinite(s.likelihood)) s.status = 3;
}

__host__ __device__ void write_result(const Scratch& s, double* value, int* status, double* diagnostic) {
    *value = s.status ? -1e12 : s.likelihood;
    *status = s.status;
    if (diagnostic) {
        for (int i = 0; i < DIAG_SIZE; ++i) diagnostic[i] = 0.0;
        diagnostic[0] = s.dimension;
        diagnostic[1] = s.logdet_normal;
        diagnostic[2] = s.logdet_prior;
        if (!s.status) {
            for (int i = 0; i < s.dimension; ++i) {
                diagnostic[3+i] = s.solution[i];
                for (int j = 0; j < s.dimension; ++j) diagnostic[3+MAX_D+i*MAX_D+j] = s.normal[i*MAX_D+j];
            }
        }
    }
}

__global__ void likelihood_batch(const double* rows, const int* meta, const double* pairs,
    const double* dd, const double* hypotheses, const double* cfg, int m, int nb,
    double* output, int* status, double* diagnostic) {
    __shared__ Scratch s;
    const int h = blockIdx.x, lane = threadIdx.x;
    if (lane == 0) initialize(s,rows,meta,dd+h*nb,hypotheses+h*12,cfg,m);
    __syncthreads();
    if (!s.status) {
        for (int r = lane; r < s.count; r += blockDim.x)
            prepare_row(s,r,rows,meta,pairs,dd+h*nb,hypotheses+h*12,cfg);
        __syncthreads();
        const int stages = cfg[2] > 0.0 ? 3 : 1;
        for (int stage = 0; stage < stages; ++stage) {
            for (int k = lane; k < s.dimension*s.dimension; k += blockDim.x) normal_entry(s,k);
            __syncthreads();
            if (lane == 0) solve_normal(s);
            __syncthreads();
            if (s.status) break;
            if (stage+1 < stages) {
                for (int r = lane; r < s.count; r += blockDim.x) reweight_row(s,r,cfg);
                __syncthreads();
            }
        }
        if (!s.status) {
            for (int r = lane; r < s.count; r += blockDim.x) likelihood_row(s,r,meta,cfg);
            __syncthreads();
            if (lane == 0) finish(s);
        }
    }
    if (lane == 0) write_result(s,output+h,status+h,diagnostic ? diagnostic+h*DIAG_SIZE : nullptr);
}


// Register-row ordered LU,kept separate from original numeric code.
#include "register_lu.cuh"

__global__ void likelihood_register(const double* rows, const int* meta, const double* pairs,
    const double* dd, const double* hypotheses, const double* cfg, int m, int nb,
    double* output, int* status, double* diagnostic, int cooperative) {
    __shared__ Scratch s;
    const int h = blockIdx.x, lane = threadIdx.x;
    if (lane == 0) initialize(s,rows,meta,dd+h*nb,hypotheses+h*12,cfg,m);
    __syncthreads();
    if (!s.status) {
        for (int r = lane; r < s.count; r += blockDim.x)
            prepare_row(s,r,rows,meta,pairs,dd+h*nb,hypotheses+h*12,cfg);
        __syncthreads();
        const int stages = cfg[2] > 0.0 ? 3 : 1;
        for (int stage = 0; stage < stages; ++stage) {
            for (int k = lane; k < s.dimension*s.dimension; k += blockDim.x) normal_entry(s,k);
            __syncthreads();
            if (cooperative && s.dimension==11) { if (lane<32) solve_register<11>(s,lane); }
            else if (lane==0) solve_normal(s);
            __syncthreads();
            if (s.status) break;
            if (stage+1 < stages) {
                for (int r = lane; r < s.count; r += blockDim.x) reweight_row(s,r,cfg);
                __syncthreads();
            }
        }
        if (!s.status) {
            for (int r = lane; r < s.count; r += blockDim.x) likelihood_row(s,r,meta,cfg);
            __syncthreads();
            if (lane == 0) finish(s);
        }
    }
    if (lane == 0) write_result(s,output+h,status+h,diagnostic ? diagnostic+h*DIAG_SIZE : nullptr);
}


// Transport-only variant: one pageable host input/output staging allocation and
// one device input/output allocation. All numeric functions and the kernel above
// are literal copies of the first measured source.
struct Workspace {
    char *device_input=nullptr,*device_output=nullptr,*host_input=nullptr,*host_output=nullptr;
    int capacity;
    int kernel_mode=2;
    ~Workspace() {
        cudaFree(device_input); cudaFree(device_output);
        delete[] host_input; delete[] host_output;
    }
};

static size_t aligned8(size_t bytes) { return (bytes+7) & ~size_t(7); }

extern "C" __declspec(dllexport) void* rbpf_create(int capacity) {
    if (capacity < 1 || capacity > 4096) return nullptr;
    Workspace* w = new(std::nothrow) Workspace;
    if (!w) return nullptr;
    w->capacity=capacity;
    const size_t input_bytes=MAX_ROWS*7*sizeof(double)+aligned8(MAX_ROWS*3*sizeof(int))+
        MAX_ROWS*2*sizeof(double)+size_t(capacity)*MAX_ROWS*sizeof(double)+size_t(capacity)*12*sizeof(double)+7*sizeof(double);
    const size_t output_bytes=aligned8(size_t(capacity)*(sizeof(double)+sizeof(int)))+size_t(capacity)*DIAG_SIZE*sizeof(double);
    if (cudaMalloc(&w->device_input,input_bytes) != cudaSuccess ||
        cudaMalloc(&w->device_output,output_bytes) != cudaSuccess ||
        (w->host_input=new(std::nothrow) char[input_bytes]) == nullptr ||
        (w->host_output=new(std::nothrow) char[output_bytes]) == nullptr) {
        delete w; return nullptr;
    }
    return w;
}

extern "C" __declspec(dllexport) void rbpf_destroy(void* pointer) { delete static_cast<Workspace*>(pointer); }

// 0 original kernel,1 same new kernel/generic solve,2 register-row11 dispatch.
extern "C" __declspec(dllexport) int rbpf_register_select(void* pointer,int mode) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || mode<0 || mode>2) return -1;
    w->kernel_mode=mode;return 0;
}

extern "C" __declspec(dllexport) int rbpf_score(void* pointer, const double* rows, const int* meta,
    const double* pairs, const double* dd, const double* hypotheses, const double* cfg,
    int count, int m, int nb, int mode, double* out, int* status, double* diagnostic) {
    Workspace* w = static_cast<Workspace*>(pointer);
    if (!w || count < 1 || count > w->capacity || m < 0 || m > MAX_ROWS || nb < 1 || nb > MAX_ROWS || (mode != 0 && mode != 1)) return -1;
    if (mode == 0) {
        for (int h = 0; h < count; ++h) {
            Scratch s{};
            initialize(s,rows,meta,dd+h*nb,hypotheses+h*12,cfg,m);
            if (!s.status) {
                for (int r = 0; r < s.count; ++r) prepare_row(s,r,rows,meta,pairs,dd+h*nb,hypotheses+h*12,cfg);
                const int stages = cfg[2] > 0.0 ? 3 : 1;
                for (int stage = 0; stage < stages; ++stage) {
                    for (int k = 0; k < s.dimension*s.dimension; ++k) normal_entry(s,k);
                    solve_normal(s);
                    if (s.status) break;
                    if (stage+1 < stages) for (int r = 0; r < s.count; ++r) reweight_row(s,r,cfg);
                }
                if (!s.status) {
                    for (int r = 0; r < s.count; ++r) likelihood_row(s,r,meta,cfg);
                    finish(s);
                }
            }
            write_result(s,out+h,status+h,diagnostic ? diagnostic+h*DIAG_SIZE : nullptr);
        }
        return 0;
    }

    const size_t rows_bytes=size_t(m)*7*sizeof(double), meta_bytes=size_t(m)*3*sizeof(int);
    const size_t pairs_bytes=size_t(nb)*2*sizeof(double), dd_bytes=size_t(count)*nb*sizeof(double);
    const size_t hypotheses_bytes=size_t(count)*12*sizeof(double), cfg_bytes=7*sizeof(double);
    const size_t meta_offset=rows_bytes, pairs_offset=aligned8(meta_offset+meta_bytes);
    const size_t dd_offset=pairs_offset+pairs_bytes, hypotheses_offset=dd_offset+dd_bytes;
    const size_t cfg_offset=hypotheses_offset+hypotheses_bytes, input_bytes=cfg_offset+cfg_bytes;
    std::memcpy(w->host_input,rows,rows_bytes);
    std::memcpy(w->host_input+meta_offset,meta,meta_bytes);
    std::memcpy(w->host_input+pairs_offset,pairs,pairs_bytes);
    std::memcpy(w->host_input+dd_offset,dd,dd_bytes);
    std::memcpy(w->host_input+hypotheses_offset,hypotheses,hypotheses_bytes);
    std::memcpy(w->host_input+cfg_offset,cfg,cfg_bytes);
    cudaError_t error=cudaMemcpy(w->device_input,w->host_input,input_bytes,cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return int(error);
    const size_t values_bytes=size_t(count)*sizeof(double), status_bytes=size_t(count)*sizeof(int);
    const size_t diagnostic_offset=aligned8(values_bytes+status_bytes);
    const size_t output_bytes=diagnostic ? diagnostic_offset+size_t(count)*DIAG_SIZE*sizeof(double) : values_bytes+status_bytes;
    if (w->kernel_mode==0) {
    likelihood_batch<<<count,128>>>(
        reinterpret_cast<double*>(w->device_input),reinterpret_cast<int*>(w->device_input+meta_offset),
        reinterpret_cast<double*>(w->device_input+pairs_offset),reinterpret_cast<double*>(w->device_input+dd_offset),
        reinterpret_cast<double*>(w->device_input+hypotheses_offset),reinterpret_cast<double*>(w->device_input+cfg_offset),m,nb,
        reinterpret_cast<double*>(w->device_output),reinterpret_cast<int*>(w->device_output+values_bytes),
        diagnostic ? reinterpret_cast<double*>(w->device_output+diagnostic_offset) : nullptr);
    } else {
    likelihood_register<<<count,128>>>(
        reinterpret_cast<double*>(w->device_input),reinterpret_cast<int*>(w->device_input+meta_offset),
        reinterpret_cast<double*>(w->device_input+pairs_offset),reinterpret_cast<double*>(w->device_input+dd_offset),
        reinterpret_cast<double*>(w->device_input+hypotheses_offset),reinterpret_cast<double*>(w->device_input+cfg_offset),m,nb,
        reinterpret_cast<double*>(w->device_output),reinterpret_cast<int*>(w->device_output+values_bytes),
        diagnostic ? reinterpret_cast<double*>(w->device_output+diagnostic_offset) : nullptr,w->kernel_mode==2);
    }
    error=cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    error=cudaMemcpy(w->host_output,w->device_output,output_bytes,cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return int(error);
    std::memcpy(out,w->host_output,values_bytes);
    std::memcpy(status,w->host_output+values_bytes,status_bytes);
    if (diagnostic) std::memcpy(diagnostic,w->host_output+diagnostic_offset,size_t(count)*DIAG_SIZE*sizeof(double));
    return 0;
}
// Isolated validation of generic/register-row normal solves,never called by score.
__global__ void register_solve_probe(const double* input,const int* dims,int count,double* output,int* statuses) {
    __shared__ Scratch s;
    const int lane=threadIdx.x;
    const int item=blockIdx.x/2,variant=blockIdx.x%2,n=dims[item];
    if (lane==0) {
    s.dimension=n;s.status=0;s.logdet_normal=0.0;
    for (int i=0;i<MAX_D;++i) {s.solution[i]=0.0;s.pivot[i]=0;}
    for (int i=0;i<MAX_D*MAX_D;++i) s.lu[i]=0.0;
    for (int i=0;i<MAX_D*MAX_D;++i) s.normal[i]=input[item*132+i];
    for (int i=0;i<MAX_D;++i) s.rhs[i]=input[item*132+121+i];
    }
    __syncthreads();
    if (variant && n==11) solve_register<11>(s,lane);else if (lane==0) solve_normal(s);
    __syncthreads();
    if (lane) return;
    const int base=(item*2+variant)*144;
    output[base]=s.logdet_normal;
    for (int i=0;i<11;++i) output[base+1+i]=s.solution[i];
    for (int i=0;i<121;++i) output[base+12+i]=s.lu[i];
    for (int i=0;i<11;++i) output[base+133+i]=s.pivot[i];
    statuses[item*2+variant]=s.status;
}

extern "C" __declspec(dllexport) int rbpf_register_solve_probe(const double* input,const int* dims,int count,double* output,int* status) {
    if (!input || !dims || !output || !status || count<1 || count>256) return -1;
    for (int i=0;i<count;++i) if (dims[i]<3 || dims[i]>11) return -1;
    double *di=nullptr,*dout=nullptr;int *dn=nullptr,*ds=nullptr;
    cudaError_t error=cudaMalloc(&di,size_t(count)*132*sizeof(double));
    if (error==cudaSuccess) error=cudaMalloc(&dn,size_t(count)*sizeof(int));
    if (error==cudaSuccess) error=cudaMalloc(&dout,size_t(count)*2*144*sizeof(double));
    if (error==cudaSuccess) error=cudaMalloc(&ds,size_t(count)*2*sizeof(int));
    if (error==cudaSuccess) error=cudaMemcpy(di,input,size_t(count)*132*sizeof(double),cudaMemcpyHostToDevice);
    if (error==cudaSuccess) error=cudaMemcpy(dn,dims,size_t(count)*sizeof(int),cudaMemcpyHostToDevice);
    if (error==cudaSuccess) {
        register_solve_probe<<<count*2,32>>>(di,dn,count,dout,ds);
        error=cudaGetLastError();
    }
    if (error==cudaSuccess) error=cudaMemcpy(output,dout,size_t(count)*2*144*sizeof(double),cudaMemcpyDeviceToHost);
    if (error==cudaSuccess) error=cudaMemcpy(status,ds,size_t(count)*2*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(di);cudaFree(dn);cudaFree(dout);cudaFree(ds);return int(error);
}

// Host clocks only:no CUDA events,extra kernels,extra synchronization or math.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

struct HostTraceData {
    void* owner=nullptr;
    uint64_t generation=0,frequency=0,count=0,flags=0;
    bool active=false;
    uint64_t rows[8][4]{};
};
static thread_local HostTraceData host_trace;
static uint64_t filetime_ticks(const FILETIME& time) {
    return (uint64_t(time.dwHighDateTime)<<32)|uint64_t(time.dwLowDateTime);
}
static void host_trace_mark(uint64_t phase) noexcept {
    if (!host_trace.active || host_trace.count>=8) {
        host_trace.flags|=1;return;
    }
    LARGE_INTEGER wall{};
    FILETIME created{},exited{},kernel{},user{};
    if (!QueryPerformanceCounter(&wall) || wall.QuadPart<0) host_trace.flags|=2;
    if (!GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user)) host_trace.flags|=4;
    auto& row=host_trace.rows[host_trace.count++];
    row[0]=phase;row[1]=uint64_t(wall.QuadPart);
    row[2]=filetime_ticks(user);row[3]=filetime_ticks(kernel);
}
struct HostTraceScope {
    bool owns;
    explicit HostTraceScope(void* pointer) noexcept:owns(!host_trace.active) {
        if (!owns) { host_trace.flags|=16;return; }
        host_trace.owner=pointer;++host_trace.generation;
        host_trace.count=0;host_trace.flags=0;host_trace.active=true;
        LARGE_INTEGER frequency{};
        if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart<=0) host_trace.flags|=8;
        host_trace.frequency=uint64_t(frequency.QuadPart);
        host_trace_mark(0);
    }
    ~HostTraceScope() noexcept {
        if (owns) { host_trace_mark(7);host_trace.active=false; }
    }
};
extern "C" __declspec(dllexport) int rbpf_trace_read(void* pointer,uint64_t* output,int capacity) {
    if (host_trace.owner!=pointer || host_trace.generation==0) return 0;
    if (host_trace.active || host_trace.count>8) return -1;
    const int count=6+int(host_trace.count)*4;
    if (!output || capacity<count) return -2;
    uint64_t data[38]{};
    data[0]=1;data[1]=host_trace.generation;data[2]=host_trace.frequency;
    data[3]=host_trace.count;data[4]=host_trace.flags;data[5]=uint64_t(host_trace.active);
    std::memcpy(data+6,host_trace.rows,size_t(host_trace.count)*4*sizeof(uint64_t));
    std::memcpy(output,data,size_t(count)*sizeof(uint64_t));
    return count;
}
extern "C" __declspec(dllexport) int rbpf_score_trace(void* pointer, const double* rows, const int* meta,
    const double* pairs, const double* dd, const double* hypotheses, const double* cfg,
    int count, int m, int nb, int mode, double* out, int* status, double* diagnostic) {
    HostTraceScope trace_scope(pointer);
    Workspace* w = static_cast<Workspace*>(pointer);
    if (!w || count < 1 || count > w->capacity || m < 0 || m > MAX_ROWS || nb < 1 || nb > MAX_ROWS || (mode != 0 && mode != 1)) return -1;
    if (mode == 0) {
        for (int h = 0; h < count; ++h) {
            Scratch s{};
            initialize(s,rows,meta,dd+h*nb,hypotheses+h*12,cfg,m);
            if (!s.status) {
                for (int r = 0; r < s.count; ++r) prepare_row(s,r,rows,meta,pairs,dd+h*nb,hypotheses+h*12,cfg);
                const int stages = cfg[2] > 0.0 ? 3 : 1;
                for (int stage = 0; stage < stages; ++stage) {
                    for (int k = 0; k < s.dimension*s.dimension; ++k) normal_entry(s,k);
                    solve_normal(s);
                    if (s.status) break;
                    if (stage+1 < stages) for (int r = 0; r < s.count; ++r) reweight_row(s,r,cfg);
                }
                if (!s.status) {
                    for (int r = 0; r < s.count; ++r) likelihood_row(s,r,meta,cfg);
                    finish(s);
                }
            }
            write_result(s,out+h,status+h,diagnostic ? diagnostic+h*DIAG_SIZE : nullptr);
        }
        return 0;
    }

    const size_t rows_bytes=size_t(m)*7*sizeof(double), meta_bytes=size_t(m)*3*sizeof(int);
    const size_t pairs_bytes=size_t(nb)*2*sizeof(double), dd_bytes=size_t(count)*nb*sizeof(double);
    const size_t hypotheses_bytes=size_t(count)*12*sizeof(double), cfg_bytes=7*sizeof(double);
    const size_t meta_offset=rows_bytes, pairs_offset=aligned8(meta_offset+meta_bytes);
    const size_t dd_offset=pairs_offset+pairs_bytes, hypotheses_offset=dd_offset+dd_bytes;
    const size_t cfg_offset=hypotheses_offset+hypotheses_bytes, input_bytes=cfg_offset+cfg_bytes;
    host_trace_mark(1);
    std::memcpy(w->host_input,rows,rows_bytes);
    std::memcpy(w->host_input+meta_offset,meta,meta_bytes);
    std::memcpy(w->host_input+pairs_offset,pairs,pairs_bytes);
    std::memcpy(w->host_input+dd_offset,dd,dd_bytes);
    std::memcpy(w->host_input+hypotheses_offset,hypotheses,hypotheses_bytes);
    std::memcpy(w->host_input+cfg_offset,cfg,cfg_bytes);
    host_trace_mark(2);
    cudaError_t error=cudaMemcpy(w->device_input,w->host_input,input_bytes,cudaMemcpyHostToDevice);
    if (error != cudaSuccess) return int(error);
    const size_t values_bytes=size_t(count)*sizeof(double), status_bytes=size_t(count)*sizeof(int);
    const size_t diagnostic_offset=aligned8(values_bytes+status_bytes);
    const size_t output_bytes=diagnostic ? diagnostic_offset+size_t(count)*DIAG_SIZE*sizeof(double) : values_bytes+status_bytes;
    host_trace_mark(3);
    if (w->kernel_mode==0) {
    likelihood_batch<<<count,128>>>(
        reinterpret_cast<double*>(w->device_input),reinterpret_cast<int*>(w->device_input+meta_offset),
        reinterpret_cast<double*>(w->device_input+pairs_offset),reinterpret_cast<double*>(w->device_input+dd_offset),
        reinterpret_cast<double*>(w->device_input+hypotheses_offset),reinterpret_cast<double*>(w->device_input+cfg_offset),m,nb,
        reinterpret_cast<double*>(w->device_output),reinterpret_cast<int*>(w->device_output+values_bytes),
        diagnostic ? reinterpret_cast<double*>(w->device_output+diagnostic_offset) : nullptr);
    } else {
    likelihood_register<<<count,128>>>(
        reinterpret_cast<double*>(w->device_input),reinterpret_cast<int*>(w->device_input+meta_offset),
        reinterpret_cast<double*>(w->device_input+pairs_offset),reinterpret_cast<double*>(w->device_input+dd_offset),
        reinterpret_cast<double*>(w->device_input+hypotheses_offset),reinterpret_cast<double*>(w->device_input+cfg_offset),m,nb,
        reinterpret_cast<double*>(w->device_output),reinterpret_cast<int*>(w->device_output+values_bytes),
        diagnostic ? reinterpret_cast<double*>(w->device_output+diagnostic_offset) : nullptr,w->kernel_mode==2);
    }
    host_trace_mark(4);
    error=cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    host_trace_mark(5);
    error=cudaMemcpy(w->host_output,w->device_output,output_bytes,cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return int(error);
    host_trace_mark(6);
    std::memcpy(out,w->host_output,values_bytes);
    std::memcpy(status,w->host_output+values_bytes,status_bytes);
    if (diagnostic) std::memcpy(diagnostic,w->host_output+diagnostic_offset,size_t(count)*DIAG_SIZE*sizeof(double));
    return 0;
}

// Diagnostic-only first API,one exclusively owned workspace. prepare is CPU-only
// and must precede idle. A fresh original score repacks all active input bytes.
static size_t first_api_input_capacity(const Workspace* w) {
    return MAX_ROWS*7*sizeof(double)+aligned8(MAX_ROWS*3*sizeof(int))+
        MAX_ROWS*2*sizeof(double)+size_t(w->capacity)*MAX_ROWS*sizeof(double)+size_t(w->capacity)*12*sizeof(double)+7*sizeof(double);
}
extern "C" __declspec(dllexport) int rbpf_first_api_prepare(void* pointer,int bytes) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || bytes<1 || size_t(bytes)>first_api_input_capacity(w)) return -1;
    std::memset(w->host_input,0,size_t(bytes));return 0;
}
static void first_api_clock(uint64_t* row,uint64_t phase,uint64_t& flags) noexcept {
    LARGE_INTEGER wall{};
    FILETIME created{},exited{},kernel{},user{};
    if (!QueryPerformanceCounter(&wall) || wall.QuadPart<0) flags|=2;
    if (!GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user)) flags|=4;
    row[0]=phase;row[1]=uint64_t(wall.QuadPart);
    row[2]=filetime_ticks(user);row[3]=filetime_ticks(kernel);
}
// Header version/op/frequency/bytes/APIstatus/completionstatus/flags/APIcount,
// followed by3(stage,QPC,user100ns,kernel100ns) rows. No TLS trace mutation.
extern "C" __declspec(dllexport) int rbpf_first_api_probe(void* pointer,int op,uint64_t* output,int capacity) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || op<0 || op>5 || !output || capacity<20) return -1;
    const size_t bytes=op==4 ? 65536 : (op>=3 ? 1 : 0);
    if (bytes>first_api_input_capacity(w)) return -1;
    uint64_t data[20]{};
    LARGE_INTEGER frequency{};
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart<=0) data[6]|=8;
    data[0]=1;data[1]=uint64_t(op);data[2]=uint64_t(frequency.QuadPart);data[3]=uint64_t(bytes);
    cudaError_t error=cudaSuccess,completed=cudaSuccess;
    first_api_clock(data+8,0,data[6]);
    if (op==1) {error=cudaStreamQuery(nullptr);data[7]=1;}
    else if (op==2) {error=cudaDeviceSynchronize();data[7]=1;}
    else if (op==3 || op==4) {
        error=cudaMemcpy(w->device_input,w->host_input,bytes,cudaMemcpyHostToDevice);data[7]=1;
    } else if (op==5) {error=cudaMemset(w->device_input,0,1);data[7]=1;}
    first_api_clock(data+12,1,data[6]);
    if (op>=3 && error==cudaSuccess) {completed=cudaDeviceSynchronize();++data[7];}
    first_api_clock(data+16,2,data[6]);
    data[4]=uint64_t(error);data[5]=uint64_t(completed);
    std::memcpy(output,data,sizeof(data));return 20;
}
