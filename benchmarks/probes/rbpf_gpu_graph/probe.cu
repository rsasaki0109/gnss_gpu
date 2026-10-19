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
struct Layout {
    int count,m,nb;
    bool diagnostic;
    size_t rows_bytes,meta_bytes,pairs_bytes,dd_bytes,hypotheses_bytes,cfg_bytes;
    size_t meta_offset,pairs_offset,dd_offset,hypotheses_offset,cfg_offset,input_bytes;
    size_t values_bytes,status_bytes,diagnostic_offset,output_bytes;
    Layout(int count_,int m_,int nb_,bool diagnostic_):
        count(count_),m(m_),nb(nb_),diagnostic(diagnostic_) {
        rows_bytes=size_t(m)*7*sizeof(double); meta_bytes=size_t(m)*3*sizeof(int);
        pairs_bytes=size_t(nb)*2*sizeof(double); dd_bytes=size_t(count)*nb*sizeof(double);
        hypotheses_bytes=size_t(count)*12*sizeof(double); cfg_bytes=7*sizeof(double);
        meta_offset=rows_bytes; pairs_offset=(meta_offset+meta_bytes+7)&~size_t(7);
        dd_offset=pairs_offset+pairs_bytes; hypotheses_offset=dd_offset+dd_bytes;
        cfg_offset=hypotheses_offset+hypotheses_bytes; input_bytes=cfg_offset+cfg_bytes;
        values_bytes=size_t(count)*sizeof(double); status_bytes=size_t(count)*sizeof(int);
        diagnostic_offset=(values_bytes+status_bytes+7)&~size_t(7);
        output_bytes=diagnostic ? diagnostic_offset+size_t(count)*DIAG_SIZE*sizeof(double) : values_bytes+status_bytes;
    }
    bool same_shape(const Layout& other) const {
        return count==other.count && m==other.m && nb==other.nb && diagnostic==other.diagnostic;
    }
};

struct Workspace {
    char *device_input=nullptr,*device_output=nullptr,*host_input=nullptr,*host_output=nullptr;
    int capacity;
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t executable=nullptr;
    cudaGraphNode_t upload=nullptr,kernel=nullptr,readback=nullptr;
    bool started=false,selected=false,use_graph=false,shape_valid=false;
    Layout shape{1,0,1,false};
    // Instantiations,shape updates,completed graph launches,completed stream launches.
    unsigned long long counters[4]{};
    ~Workspace() {
        if (stream) cudaStreamSynchronize(stream);
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
        cudaFree(device_input); cudaFree(device_output);
        if (host_input) cudaFreeHost(host_input);
        if (host_output) cudaFreeHost(host_output);
    }
};

static size_t aligned8(size_t bytes) { return (bytes+7)&~size_t(7); }

// Only launch geometry/pointers/byte counts change. Every call uploads fresh data.
// Kernel parameter values are copied by the graph API before this function returns.
static cudaError_t configure_graph(Workspace* w,const Layout& q,bool create) {
    if (!create && w->shape_valid && q.same_shape(w->shape)) return cudaSuccess;
    double* rows=reinterpret_cast<double*>(w->device_input);
    int* meta=reinterpret_cast<int*>(w->device_input+q.meta_offset);
    double* pairs=reinterpret_cast<double*>(w->device_input+q.pairs_offset);
    double* dd=reinterpret_cast<double*>(w->device_input+q.dd_offset);
    double* hypotheses=reinterpret_cast<double*>(w->device_input+q.hypotheses_offset);
    double* cfg=reinterpret_cast<double*>(w->device_input+q.cfg_offset);
    double* values=reinterpret_cast<double*>(w->device_output);
    int* status=reinterpret_cast<int*>(w->device_output+q.values_bytes);
    double* diagnostic=q.diagnostic ? reinterpret_cast<double*>(w->device_output+q.diagnostic_offset) : nullptr;
    int m=q.m,nb=q.nb;
    void* args[]={&rows,&meta,&pairs,&dd,&hypotheses,&cfg,&m,&nb,&values,&status,&diagnostic};
    cudaKernelNodeParams params{};
    params.func=reinterpret_cast<void*>(likelihood_batch);
    params.gridDim=dim3(q.count); params.blockDim=dim3(128); params.kernelParams=args;
    cudaError_t error;
    w->shape_valid=false;
    if (create) {
        error=cudaGraphCreate(&w->graph,0);
        if (error != cudaSuccess) return error;
        error=cudaGraphAddMemcpyNode1D(&w->upload,w->graph,nullptr,0,
            w->device_input,w->host_input,q.input_bytes,cudaMemcpyHostToDevice);
        if (error != cudaSuccess) return error;
        error=cudaGraphAddKernelNode(&w->kernel,w->graph,&w->upload,1,&params);
        if (error != cudaSuccess) return error;
        error=cudaGraphAddMemcpyNode1D(&w->readback,w->graph,&w->kernel,1,
            w->host_output,w->device_output,q.output_bytes,cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) return error;
        error=cudaGraphInstantiate(&w->executable,w->graph,0);
        if (error != cudaSuccess) return error;
        ++w->counters[0];
    } else {
        error=cudaGraphExecMemcpyNodeSetParams1D(w->executable,w->upload,
            w->device_input,w->host_input,q.input_bytes,cudaMemcpyHostToDevice);
        if (error != cudaSuccess) return error;
        error=cudaGraphExecKernelNodeSetParams(w->executable,w->kernel,&params);
        if (error != cudaSuccess) return error;
        error=cudaGraphExecMemcpyNodeSetParams1D(w->executable,w->readback,
            w->host_output,w->device_output,q.output_bytes,cudaMemcpyDeviceToHost);
        if (error != cudaSuccess) return error;
        ++w->counters[1];
    }
    w->shape=q; w->shape_valid=true;
    return cudaSuccess;
}

extern "C" __declspec(dllexport) void* rbpf_create(int capacity) {
    if (capacity < 1 || capacity > 4096) return nullptr;
    Workspace* w=new(std::nothrow) Workspace;
    if (!w) return nullptr;
    w->capacity=capacity;
    const size_t input_bytes=MAX_ROWS*7*sizeof(double)+aligned8(MAX_ROWS*3*sizeof(int))+
        MAX_ROWS*2*sizeof(double)+size_t(capacity)*MAX_ROWS*sizeof(double)+size_t(capacity)*12*sizeof(double)+7*sizeof(double);
    const size_t output_bytes=aligned8(size_t(capacity)*(sizeof(double)+sizeof(int)))+size_t(capacity)*DIAG_SIZE*sizeof(double);
    if (cudaMalloc(&w->device_input,input_bytes) != cudaSuccess ||
        cudaMalloc(&w->device_output,output_bytes) != cudaSuccess ||
        cudaMallocHost(&w->host_input,input_bytes) != cudaSuccess ||
        cudaMallocHost(&w->host_output,output_bytes) != cudaSuccess ||
        cudaStreamCreateWithFlags(&w->stream,cudaStreamNonBlocking) != cudaSuccess) {
        delete w; return nullptr;
    }
    return w;
}

extern "C" __declspec(dllexport) void rbpf_destroy(void* pointer) { delete static_cast<Workspace*>(pointer); }

extern "C" __declspec(dllexport) int rbpf_graph_select(void* pointer,int enabled) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || w->started || w->selected || (enabled != 0 && enabled != 1)) return -1;
    w->selected=true; w->use_graph=enabled != 0;
    // Instantiate one bounded graph during workspace setup,without scoring.
    return enabled ? int(configure_graph(w,Layout(1,0,1,false),true)) : 0;
}

extern "C" __declspec(dllexport) int rbpf_graph_stats(void* pointer,unsigned long long* output) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || !output) return -1;
    for (int i=0;i<4;++i) output[i]=w->counters[i];
    return 0;
}

static int stream_failure(Workspace* w,cudaError_t error) {
    cudaStreamSynchronize(w->stream);
    return int(error);
}

extern "C" __declspec(dllexport) int rbpf_score(void* pointer, const double* rows, const int* meta,
    const double* pairs, const double* dd, const double* hypotheses, const double* cfg,
    int count, int m, int nb, int mode, double* out, int* status, double* diagnostic) {
    Workspace* w=static_cast<Workspace*>(pointer);
    if (!w || count < 1 || count > w->capacity || m < 0 || m > MAX_ROWS || nb < 1 || nb > MAX_ROWS || (mode != 0 && mode != 1)) return -1;
    w->started=true;
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
    const Layout q(count,m,nb,diagnostic != nullptr);
    std::memcpy(w->host_input,rows,q.rows_bytes);
    std::memcpy(w->host_input+q.meta_offset,meta,q.meta_bytes);
    std::memcpy(w->host_input+q.pairs_offset,pairs,q.pairs_bytes);
    std::memcpy(w->host_input+q.dd_offset,dd,q.dd_bytes);
    std::memcpy(w->host_input+q.hypotheses_offset,hypotheses,q.hypotheses_bytes);
    std::memcpy(w->host_input+q.cfg_offset,cfg,q.cfg_bytes);
    cudaError_t error;
    if (w->use_graph) {
        error=configure_graph(w,q,false);
        if (error != cudaSuccess) return stream_failure(w,error);
        error=cudaGraphLaunch(w->executable,w->stream);
        if (error != cudaSuccess) return stream_failure(w,error);
    } else {
        error=cudaMemcpyAsync(w->device_input,w->host_input,q.input_bytes,cudaMemcpyHostToDevice,w->stream);
        if (error != cudaSuccess) return stream_failure(w,error);
        likelihood_batch<<<count,128,0,w->stream>>>(
            reinterpret_cast<double*>(w->device_input),reinterpret_cast<int*>(w->device_input+q.meta_offset),
            reinterpret_cast<double*>(w->device_input+q.pairs_offset),reinterpret_cast<double*>(w->device_input+q.dd_offset),
            reinterpret_cast<double*>(w->device_input+q.hypotheses_offset),reinterpret_cast<double*>(w->device_input+q.cfg_offset),m,nb,
            reinterpret_cast<double*>(w->device_output),reinterpret_cast<int*>(w->device_output+q.values_bytes),
            diagnostic ? reinterpret_cast<double*>(w->device_output+q.diagnostic_offset) : nullptr);
        error=cudaGetLastError();
        if (error != cudaSuccess) return stream_failure(w,error);
        error=cudaMemcpyAsync(w->host_output,w->device_output,q.output_bytes,cudaMemcpyDeviceToHost,w->stream);
        if (error != cudaSuccess) return stream_failure(w,error);
    }
    // Complete every upload/kernel/readback before host reads,updates or destruction.
    error=cudaStreamSynchronize(w->stream);
    if (error != cudaSuccess) return stream_failure(w,error);
    ++w->counters[w->use_graph ? 2 : 3];
    std::memcpy(out,w->host_output,q.values_bytes);
    std::memcpy(status,w->host_output+q.values_bytes,q.status_bytes);
    if (diagnostic) std::memcpy(diagnostic,w->host_output+q.diagnostic_offset,size_t(count)*DIAG_SIZE*sizeof(double));
    return 0;
}
