// All32 lanes of warp zero participate;one row per lane,compile-time columns.
// Independent rows interleave,but each element keeps original operation order.
template<int N> __device__ __forceinline__ void solve_register(Scratch& s,int lane) {
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
    // Broadcast each immutable solved row once;cache bits in warp registers.
    // Descending-i rows still subtract in their original ascending-k order.
    double back[N];
    #pragma unroll
    for (int i=N-1;i>=0;--i) {
        #pragma unroll
        for (int k=i+1;k<N;++k) {
            if (lane==i) b-=row[k]*back[k];
        }
        if (lane==i) b/=row[i];
        // All32 lanes participate;back[k] is initialized before any k>i use.
        if (i>0) back[i]=__shfl_sync(mask,b,i);
    }
    const unsigned bad=__ballot_sync(mask,lane<N && !isfinite(b));
    if (lane<N) s.solution[lane]=b;
    if (lane==0 && bad) s.status=3;
}
