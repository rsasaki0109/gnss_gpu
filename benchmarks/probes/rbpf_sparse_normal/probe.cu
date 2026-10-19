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


// Transport-only variant: one pageable host input/output staging allocation and
// one device input/output allocation. All numeric functions and the kernel above
// are literal copies of the first measured source.
// Ordered structural-zero elimination,kept separate from the old Scratch.
struct SparseScratch {
    Scratch base;
    int offsets[MAX_GROUPS+1],indices[MAX_ROWS],unsafe;
};

__device__ void prepare_group_index(SparseScratch& q) {
    Scratch& s=q.base;
    const int groups=s.dimension-3;
    int counts[MAX_GROUPS]{},cursor[MAX_GROUPS]{};
    q.unsafe=0;
    for (int r=0;r<s.count;++r) {
        const int group=s.group[r];
        if (group < -1 || group >= groups) q.unsafe=1;
        else if (group >= 0) ++counts[group];
    }
    q.offsets[0]=0;
    for (int g=0;g<groups;++g) q.offsets[g+1]=q.offsets[g]+counts[g];
    // Stable fill retains the original row order in every group.
    for (int r=0;r<s.count;++r) {
        const int group=s.group[r];
        if (group >= 0 && group < groups) q.indices[q.offsets[group]+cursor[group]++]=r;
    }
}

__device__ bool sparse_row_safe(const SparseScratch& q,int r) {
    const Scratch& s=q.base;
    // With <=128 rows,this bound keeps every dot-product partial sum finite.
    // Extreme finite values retain the old operations and their NaN behavior.
    if (!isfinite(s.bw[r]) || fabs(s.bw[r]) > 0x1p500) return false;
    for (int j=0;j<s.dimension;++j) {
        const double value=s.Aw[r*MAX_D+j];
        if (!isfinite(value) || fabs(value) > 0x1p500) return false;
        // Positive structural zeros only;unexpected shapes/signs use the old sum.
        if (j >= 3 && j-3 != s.group[r] && __double_as_longlong(value) != 0) return false;
    }
    return true;
}

__device__ void sparse_normal_entry(SparseScratch& q,int index) {
    Scratch& s=q.base;
    const int i=index/s.dimension,j=index%s.dimension;
    int begin=0,end=s.count;
    bool indexed=false;
    if (i >= 3 || j >= 3) {
        indexed=true;
        if (i >= 3 && j >= 3 && i != j) end=0;
        else {
            const int group=i >= 3 ? i-3 : j-3;
            begin=q.offsets[group]; end=q.offsets[group+1];
        }
    }
    double total=0.0;
    for (int k=begin;k<end;++k) {
        const int r=indexed ? q.indices[k] : k;
        total += s.Aw[r*MAX_D+i]*s.Aw[r*MAX_D+j];
    }
    s.normal[i*MAX_D+j]=s.prior[i*MAX_D+j]+total;
    if (j == 0) {
        total=0.0;
        for (int k=begin;k<end;++k) {
            const int r=indexed ? q.indices[k] : k;
            total += s.Aw[r*MAX_D+i]*s.bw[r];
        }
        s.rhs[i]=total;
    }
}

__global__ void sparse_likelihood_batch(const double* rows,const int* meta,const double* pairs,
    const double* dd,const double* hypotheses,const double* cfg,int m,int nb,
    double* output,int* status,double* diagnostic,int enabled,unsigned long long* counters) {
    __shared__ SparseScratch q;
    Scratch& s=q.base;
    const int h=blockIdx.x,lane=threadIdx.x;
    unsigned long long dense_stages=0,sparse_stages=0,skipped_products=0,unsafe_stages=0;
    if (lane == 0) {
        initialize(s,rows,meta,dd+h*nb,hypotheses+h*12,cfg,m);
        q.unsafe=0;
        if (enabled && !s.status) prepare_group_index(q);
    }
    __syncthreads();
    if (!s.status) {
        for (int r=lane;r<s.count;r+=blockDim.x) {
            prepare_row(s,r,rows,meta,pairs,dd+h*nb,hypotheses+h*12,cfg);
            if (enabled && !sparse_row_safe(q,r)) atomicExch(&q.unsafe,1);
        }
        __syncthreads();
        const int stages=cfg[2] > 0.0 ? 3 : 1;
        for (int stage=0;stage<stages;++stage) {
            const bool sparse=enabled && !q.unsafe;
            for (int k=lane;k<s.dimension*s.dimension;k+=blockDim.x) {
                if (sparse) sparse_normal_entry(q,k);
                else normal_entry(s,k);
            }
            __syncthreads();
            if (lane == 0) {
                if (counters) {
                    if (sparse) {
                        ++sparse_stages;
                        const int full=s.count*(s.dimension*s.dimension+s.dimension);
                        const int actual=12*s.count+8*q.offsets[s.dimension-3];
                        skipped_products+=full-actual;
                    } else { ++dense_stages; if (enabled && q.unsafe) ++unsafe_stages; }
                }
                solve_normal(s);
            }
            __syncthreads();
            if (s.status) break;
            if (stage+1 < stages) {
                for (int r=lane;r<s.count;r+=blockDim.x) {
                    reweight_row(s,r,cfg);
                    if (enabled && !sparse_row_safe(q,r)) atomicExch(&q.unsafe,1);
                }
                __syncthreads();
            }
        }
        if (!s.status) {
            for (int r=lane;r<s.count;r+=blockDim.x) likelihood_row(s,r,meta,cfg);
            __syncthreads();
            if (lane == 0) finish(s);
        }
    }
    if (lane == 0) {
        write_result(s,output+h,status+h,diagnostic ? diagnostic+h*DIAG_SIZE : nullptr);
        if (counters) {
            counters[h*4]=dense_stages; counters[h*4+1]=sparse_stages;
            counters[h*4+2]=skipped_products; counters[h*4+3]=unsafe_stages;
        }
    }
}

// Differential normal-matrix probe only;never invoked by a score.
constexpr int NORMAL_INPUT=MAX_ROWS*MAX_D+MAX_ROWS+MAX_D*MAX_D;
constexpr int NORMAL_OUTPUT=2*(MAX_D*MAX_D+MAX_D);
__global__ void sparse_normal_probe(const double* data,const int* groups,int m,int n,
    double* output,int* safe) {
    __shared__ SparseScratch q;
    Scratch& s=q.base;
    const int h=blockIdx.x,lane=threadIdx.x;
    const double* input=data+h*NORMAL_INPUT;
    for (int i=lane;i<MAX_ROWS*MAX_D;i+=blockDim.x) s.Aw[i]=input[i];
    for (int i=lane;i<MAX_ROWS;i+=blockDim.x) {
        s.bw[i]=input[MAX_ROWS*MAX_D+i]; s.group[i]=groups[h*MAX_ROWS+i];
    }
    for (int i=lane;i<MAX_D*MAX_D;i+=blockDim.x) s.prior[i]=input[MAX_ROWS*(MAX_D+1)+i];
    __syncthreads();
    if (lane == 0) { s.dimension=n; s.count=m; prepare_group_index(q); }
    __syncthreads();
    for (int r=lane;r<m;r+=blockDim.x) if (!sparse_row_safe(q,r)) atomicExch(&q.unsafe,1);
    __syncthreads();
    for (int pass=0;pass<2;++pass) {
        for (int k=lane;k<n*n;k+=blockDim.x) {
            if (pass && !q.unsafe) sparse_normal_entry(q,k);
            else normal_entry(s,k);
        }
        __syncthreads();
        for (int k=lane;k<MAX_D*MAX_D+MAX_D;k+=blockDim.x) {
            const double value=k < MAX_D*MAX_D ?
                (k/MAX_D < n && k%MAX_D < n ? s.normal[k] : 0.0) :
                (k-MAX_D*MAX_D < n ? s.rhs[k-MAX_D*MAX_D] : 0.0);
            output[h*NORMAL_OUTPUT+pass*(MAX_D*MAX_D+MAX_D)+k]=value;
        }
        __syncthreads();
    }
    if (lane == 0) safe[h]=!q.unsafe;
}

extern "C" __declspec(dllexport) int rbpf_sparse_normal_probe(const double* data,const int* groups,
    int count,int m,int n,double* output,int* safe) {
    if (!data || !groups || !output || !safe || count < 1 || count > 4096 || m < 0 || m > MAX_ROWS || n < 3 || n > MAX_D) return -1;
    double *device_data=nullptr,*device_output=nullptr;
    int *device_groups=nullptr,*device_safe=nullptr;
    const size_t ib=size_t(count)*NORMAL_INPUT*sizeof(double),gb=size_t(count)*MAX_ROWS*sizeof(int);
    const size_t ob=size_t(count)*NORMAL_OUTPUT*sizeof(double),sb=size_t(count)*sizeof(int);
    cudaError_t error=cudaMalloc(&device_data,ib);
    if (error == cudaSuccess) error=cudaMalloc(&device_groups,gb);
    if (error == cudaSuccess) error=cudaMalloc(&device_output,ob);
    if (error == cudaSuccess) error=cudaMalloc(&device_safe,sb);
    if (error == cudaSuccess) error=cudaMemcpy(device_data,data,ib,cudaMemcpyHostToDevice);
    if (error == cudaSuccess) error=cudaMemcpy(device_groups,groups,gb,cudaMemcpyHostToDevice);
    if (error == cudaSuccess) {
        sparse_normal_probe<<<count,128>>>(device_data,device_groups,m,n,device_output,device_safe);
        error=cudaGetLastError();
    }
    if (error == cudaSuccess) error=cudaMemcpy(output,device_output,ob,cudaMemcpyDeviceToHost);
    if (error == cudaSuccess) error=cudaMemcpy(safe,device_safe,sb,cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) cudaDeviceSynchronize();
    cudaFree(device_data); cudaFree(device_groups); cudaFree(device_output); cudaFree(device_safe);
    return int(error);
}

struct Workspace {
    char *device_input=nullptr,*device_output=nullptr,*host_input=nullptr,*host_output=nullptr;
    int capacity;
    int kernel=0;
    bool started=false,selected=false,profiling=false,profile_valid=false;
    unsigned long long *device_counts=nullptr,*host_counts=nullptr,counts[4]{};
    ~Workspace() {
        cudaFree(device_input); cudaFree(device_output); cudaFree(device_counts);
        delete[] host_counts;
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

extern "C" __declspec(dllexport) int rbpf_sparse_select(void* pointer,int kernel) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || w->started || w->selected || kernel < 0 || kernel > 2) return -1;
    w->selected=true; w->kernel=kernel; return 0;
}

extern "C" __declspec(dllexport) int rbpf_sparse_profile(void* pointer,int enabled) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || (enabled != 0 && enabled != 1)) return -1;
    w->profile_valid=false;
    if (enabled && !w->device_counts) {
        cudaError_t error=cudaMalloc(&w->device_counts,size_t(w->capacity)*4*sizeof(unsigned long long));
        if (error != cudaSuccess) return int(error);
    }
    if (enabled && !w->host_counts) {
        w->host_counts=new(std::nothrow) unsigned long long[size_t(w->capacity)*4];
        if (!w->host_counts) return -2;
    }
    w->profiling=enabled != 0; return 0;
}

extern "C" __declspec(dllexport) int rbpf_sparse_profile_get(void* pointer,unsigned long long* output) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || !output) return -1;
    if (!w->profile_valid) return 1;
    for (int i=0;i<4;++i) output[i]=w->counts[i];
    return 0;
}

extern "C" __declspec(dllexport) int rbpf_score(void* pointer, const double* rows, const int* meta,
    const double* pairs, const double* dd, const double* hypotheses, const double* cfg,
    int count, int m, int nb, int mode, double* out, int* status, double* diagnostic) {
    Workspace* w = static_cast<Workspace*>(pointer);
    if (!w || count < 1 || count > w->capacity || m < 0 || m > MAX_ROWS || nb < 1 || nb > MAX_ROWS || (mode != 0 && mode != 1)) return -1;
    w->started=true; w->profile_valid=false;
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
    if (w->kernel == 0) {
    likelihood_batch<<<count,128>>>(
        reinterpret_cast<double*>(w->device_input),reinterpret_cast<int*>(w->device_input+meta_offset),
        reinterpret_cast<double*>(w->device_input+pairs_offset),reinterpret_cast<double*>(w->device_input+dd_offset),
        reinterpret_cast<double*>(w->device_input+hypotheses_offset),reinterpret_cast<double*>(w->device_input+cfg_offset),m,nb,
        reinterpret_cast<double*>(w->device_output),reinterpret_cast<int*>(w->device_output+values_bytes),
        diagnostic ? reinterpret_cast<double*>(w->device_output+diagnostic_offset) : nullptr);
    } else {
    sparse_likelihood_batch<<<count,128>>>(
        reinterpret_cast<double*>(w->device_input),reinterpret_cast<int*>(w->device_input+meta_offset),
        reinterpret_cast<double*>(w->device_input+pairs_offset),reinterpret_cast<double*>(w->device_input+dd_offset),
        reinterpret_cast<double*>(w->device_input+hypotheses_offset),reinterpret_cast<double*>(w->device_input+cfg_offset),m,nb,
        reinterpret_cast<double*>(w->device_output),reinterpret_cast<int*>(w->device_output+values_bytes),
        diagnostic ? reinterpret_cast<double*>(w->device_output+diagnostic_offset) : nullptr,w->kernel == 2,w->profiling ? w->device_counts : nullptr);
    }
    error=cudaGetLastError();
    if (error != cudaSuccess) return int(error);
    error=cudaMemcpy(w->host_output,w->device_output,output_bytes,cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) return int(error);
    if (w->profiling && w->kernel != 0) {
        error=cudaMemcpy(w->host_counts,w->device_counts,size_t(count)*4*sizeof(unsigned long long),cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) return int(error);
        for (int i=0;i<4;++i) w->counts[i]=0;
        for (int h=0;h<count;++h) for (int i=0;i<4;++i) w->counts[i]+=w->host_counts[h*4+i];
        w->profile_valid=true;
    }
    std::memcpy(out,w->host_output,values_bytes);
    std::memcpy(status,w->host_output+values_bytes,status_bytes);
    if (diagnostic) std::memcpy(diagnostic,w->host_output+diagnostic_offset,size_t(count)*DIAG_SIZE*sizeof(double));
    return 0;
}
