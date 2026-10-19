// Per-score exact first-system sharing. No cross-call numerical cache.
struct InitialTemplate {
    double normal[MAX_D*MAX_D],lu[MAX_D*MAX_D],logdet;
    int pivot[MAX_D],valid;
};
struct InitialKey { uint64_t words[11];int eligible,dependents; };

__device__ bool initial_safe(const Scratch& s) {
    if (s.dimension!=11 || s.status) return false;
    for (int i=0;i<11;++i) {
        if (!(isfinite(s.rhs[i]) && fabs(s.rhs[i])<=0x1p500)) return false;
        for (int j=0;j<11;++j) {
            const double value=s.normal[i*MAX_D+j];
            if (!(isfinite(value) && fabs(value)<=0x1p500)) return false;
        }
    }
    return true;
}

__device__ void initial_rhs(Scratch& s,int i) {
    double total=0.0;
    for (int r=0;r<s.count;++r) total+=s.Aw[r*MAX_D+i]*s.bw[r];
    s.rhs[i]=total;
}

// Load the original LU bits,then execute the unchanged original solve tail.
template<int N> __device__ __forceinline__ void initial_solve_rhs(Scratch& s,int lane) {
    constexpr unsigned mask=0xffffffffu;
    double row[N];
    double b=lane<N ? s.rhs[lane] : 0.0;
    #pragma unroll
    for (int k=0;k<N;++k) row[k]=lane<N ? s.lu[lane*N+k] : 0.0;
    #pragma unroll
    for (int i=0;i<N;++i) {
        const int best=s.pivot[i];
        const double upper=__shfl_sync(mask,b,i);
        const double lower=__shfl_sync(mask,b,best);
        if (lane==i) b=lower;
        else if (lane==best) b=upper;
    }
    // Forward elimination by column preserves each row's ascending-k subtracts.
    #pragma unroll
    for (int k=0;k<N;++k) {
        const double solved=__shfl_sync(mask,b,k);
        if (lane>k && lane<N) b-=row[k]*solved;
    }
    // Backward substitution retains ascending-k within every descending-i row.
    #pragma unroll
    for (int i=N-1;i>=0;--i) {
        #pragma unroll
        for (int k=i+1;k<N;++k) {
            const double solved=__shfl_sync(mask,b,k);
            if (lane==i) b-=row[k]*solved;
        }
        if (lane==i) b/=row[i];
    }
    const unsigned bad=__ballot_sync(mask,lane<N && !isfinite(b));
    if (lane<N) s.solution[lane]=b;
    if (lane==0 && bad) s.status=3;
}

__global__ void build_initial_templates(const double* rows,const int* meta,const double* pairs,
    const double* dd,const double* hypotheses,const double* cfg,int m,int nb,
    const int* owners,InitialTemplate* templates) {
    const int h=blockIdx.x,lane=threadIdx.x;
    if (owners[h]!=h) return;
    __shared__ Scratch s;
    __shared__ int eligible;
    if (lane==0) {
        templates[h].valid=0;
        initialize(s,rows,meta,dd+h*nb,hypotheses+h*12,cfg,m);
    }
    __syncthreads();
    if (s.status || s.dimension!=11) return;
    for (int r=lane;r<s.count;r+=blockDim.x)
        prepare_row(s,r,rows,meta,pairs,dd+h*nb,hypotheses+h*12,cfg);
    __syncthreads();
    for (int k=lane;k<121;k+=blockDim.x) normal_entry(s,k);
    __syncthreads();
    if (lane==0) eligible=initial_safe(s);
    __syncthreads();
    if (!eligible) return;
    if (lane<32) solve_register<11>(s,lane);
    __syncthreads();
    if (s.status) return;
    for (int k=lane;k<121;k+=blockDim.x) {
        templates[h].normal[k]=s.normal[k];templates[h].lu[k]=s.lu[k];
    }
    for (int i=lane;i<11;i+=blockDim.x) templates[h].pivot[i]=s.pivot[i];
    if (lane==0) {templates[h].logdet=s.logdet_normal;templates[h].valid=1;}
}
