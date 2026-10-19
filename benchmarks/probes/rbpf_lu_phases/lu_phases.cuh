// Diagnostic boundaries deliberately add two warp barriers per tick.
// Both barriers remain in the marker-free perturbation control.
template<bool Measure> __device__ __forceinline__ void phase_tick(
    unsigned long long* stamps,int index,int lane) {
    __syncwarp(0xffffffffu);
    if constexpr (Measure) { if (lane==0) stamps[index]=clock64(); }
    __syncwarp(0xffffffffu);
}

// All32 lanes of warp zero participate;one row per lane,compile-time columns.
// Independent rows interleave,but each element keeps original operation order.
template<int N,bool Measure> __device__ __forceinline__ void solve_register_phases(Scratch& s,int lane,unsigned long long* stamps) {
    phase_tick<Measure>(stamps,0,lane);
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
        phase_tick<Measure>(stamps,1,lane);
        if constexpr (Measure) { if (lane==0) {stamps[8]=2;stamps[9]=0;} }
        if (lane==0) solve_normal(s);
        return;
    }
    phase_tick<Measure>(stamps,1,lane);
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
    phase_tick<Measure>(stamps,2,lane);
    if constexpr (Measure) { if (lane==0) {stamps[8]=1;stamps[9]=processed;} }
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
    phase_tick<Measure>(stamps,3,lane);
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
    phase_tick<Measure>(stamps,4,lane);
    // Forward elimination by column preserves each row's ascending-k subtracts.
    #pragma unroll
    for (int k=0;k<N;++k) {
        const double solved=__shfl_sync(mask,b,k);
        if (lane>k && lane<N) b-=row[k]*solved;
    }
    phase_tick<Measure>(stamps,5,lane);
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
    phase_tick<Measure>(stamps,6,lane);
    const unsigned bad=__ballot_sync(mask,lane<N && !isfinite(b));
    if (lane<N) s.solution[lane]=b;
    if (lane==0 && bad) s.status=3;
    phase_tick<Measure>(stamps,7,lane);
}
