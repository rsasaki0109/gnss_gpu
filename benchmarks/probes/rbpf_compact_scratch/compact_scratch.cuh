// Only row-array capacities and C++ type signatures differ;numeric bodies stay ordered.
namespace compact {
template<int RowCapacity> struct ScratchRows {
    double A[RowCapacity * MAX_D], Aw[RowCapacity * MAX_D];
    double b[RowCapacity], bw[RowCapacity], sigma[RowCapacity], rowll[RowCapacity];
    double prior[MAX_D * MAX_D], normal[MAX_D * MAX_D], lu[MAX_D * MAX_D];
    double rhs[MAX_D], solution[MAX_D], temporary[MAX_D];
    int pivot[MAX_D], kept[RowCapacity], group[RowCapacity];
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
template<int RowCapacity>
__host__ __device__ void initialize(ScratchRows<RowCapacity>& s, const double* rows, const int* meta,
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

template<int RowCapacity>
__host__ __device__ void prepare_row(ScratchRows<RowCapacity>& s, int r, const double* rows, const int* meta,
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

template<int RowCapacity>
__host__ __device__ void normal_entry(ScratchRows<RowCapacity>& s, int index) {
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

template<int RowCapacity>
__host__ __device__ void solve_normal(ScratchRows<RowCapacity>& s) {
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

template<int RowCapacity>
__host__ __device__ double residual(const ScratchRows<RowCapacity>& s, int r) {
    double value = 0.0;
    for (int j = 0; j < s.dimension; ++j) value += s.A[r*MAX_D+j] * s.solution[j];
    return s.b[r] - value;
}

template<int RowCapacity>
__host__ __device__ void reweight_row(ScratchRows<RowCapacity>& s, int r, const double* cfg) {
    const double weight = fmin(1.0, cfg[2] / fmax(fabs(residual(s,r)), 1e-12));
    const double sw = sqrt(weight);
    s.bw[r] = s.b[r] * sw;
    for (int j = 0; j < s.dimension; ++j) s.Aw[r*MAX_D+j] = s.A[r*MAX_D+j] * sw;
}

template<int RowCapacity>
__host__ __device__ void likelihood_row(ScratchRows<RowCapacity>& s, int r, const int* meta, const double* cfg) {
    const double value = residual(s,r), sigma = s.sigma[r];
    const double outlier = fmax(meta[3*s.kept[r]] ? cfg[5] : cfg[6], sigma);
    const double base = log(1.0-cfg[4]) - 0.5*value*value - log(sigma);
    const double scaled = value*sigma/outlier;
    const double tail = log(cfg[4]) - 0.5*scaled*scaled - log(outlier);
    s.rowll[r] = fmax(base,tail) + log1p(exp(-fabs(base-tail))) - 0.5*LOG2PI;
}

template<int RowCapacity>
__host__ __device__ void finish(ScratchRows<RowCapacity>& s) {
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

template<int RowCapacity>
__host__ __device__ void write_result(const ScratchRows<RowCapacity>& s, double* value, int* status, double* diagnostic) {
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

// All32 lanes of warp zero participate;one row per lane,compile-time columns.
// Independent rows interleave,but each element keeps original operation order.
template<int N,int RowCapacity> __device__ __forceinline__ void solve_register(ScratchRows<RowCapacity>& s,int lane) {
    constexpr unsigned mask=0xffffffffu;
    double row[N];
    double b=lane<N ? s.rhs[lane] : 0.0;
    bool unsafe=!(isfinite(b) && fabs(b)<=0x1p500);
    #pragma unroll
    for (int k=0;k<N;++k) {
        row[k]=lane<N ? s.normal[lane*MAX_D+k] : 0.0;
        unsafe=unsafe || !(isfinite(row[k]) && fabs(row[k])<=0x1p500);
    }
    // Dispatch guard only;unsafe values use literal original solve,not pruning.
    if (__any_sync(mask,unsafe)) {
        if (lane==0) solve_normal(s);
        return;
    }
    int processed=0,sign=1,failed=0;
    double saved_diagonal=0.0;
    #pragma unroll
    for (int j=0;j<N;++j) {
        const double own=fabs(row[j]);
        double magnitude=__shfl_sync(mask,own,j);
        int best=j;
        #pragma unroll
        for (int i=j+1;i<N;++i) {
            const double candidate=__shfl_sync(mask,own,i);
            if (candidate>magnitude) {best=i;magnitude=candidate;}
        }
        if (!(magnitude>0.0) || !isfinite(magnitude)) {failed=1;break;}
        if (lane==0) s.pivot[j]=best;
        #pragma unroll
        for (int k=0;k<N;++k) {
            const double upper=__shfl_sync(mask,row[k],j);
            const double lower=__shfl_sync(mask,row[k],best);
            if (lane==j) row[k]=lower;
            else if (lane==best) row[k]=upper;
        }
        const double diagonal=__shfl_sync(mask,row[j],j);
        if (best!=j) sign=-sign;
        if (diagonal<0.0) sign=-sign;
        if (lane==j) saved_diagonal=diagonal;
        if (lane>j && lane<N) row[j]/=diagonal;
        #pragma unroll
        for (int k=j+1;k<N;++k) {
            const double upper=__shfl_sync(mask,row[k],j);
            if (lane>j && lane<N) row[k]-=row[j]*upper;
        }
        processed=j+1;
    }
    if (lane<N) {
        s.solution[lane]=b;
        #pragma unroll
        for (int k=0;k<N;++k) s.lu[lane*N+k]=row[k];
    }
    const double diagonal_log=lane<processed ? log(fabs(saved_diagonal)) : 0.0;
    double logdet=0.0;
    #pragma unroll
    for (int j=0;j<N;++j) if (j<processed) {
        const double term=__shfl_sync(mask,diagonal_log,j);
        if (lane==0) logdet+=term;
    }
    if (lane==0) {
        s.logdet_normal=logdet;
        if (failed || !isfinite(logdet) || sign<=0) s.status=1;
    }
    __syncwarp(mask);
    if (s.status) return;
    // Order every shared status read before lane zero can write status3.
    __syncwarp(mask);
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

}
static_assert(sizeof(Scratch)==30904);
static_assert(sizeof(compact::ScratchRows<60>)==16216);
static_assert(sizeof(compact::ScratchRows<64>)==17080);
static_assert(sizeof(compact::ScratchRows<128>)==30904);
