// All128 block threads participate;one shared LU element per active thread.
// Ordered per-element j updates;literal original serial forward/backward solve.
template<int N> __device__ __forceinline__ void solve_block(Scratch& s,int lane) {
    constexpr int elements=N*N;
    const int row=lane/N,col=lane%N;
    const double element=lane<elements ? s.normal[row*MAX_D+col] : 0.0;
    const double rhs=lane<N ? s.rhs[lane] : 0.0;
    const bool unsafe=!(isfinite(element) && fabs(element)<=0x1p500) ||
        !(isfinite(rhs) && fabs(rhs)<=0x1p500);
    // Uniform whole-block dispatch;no changed numeric threshold or approximation.
    if (__syncthreads_or(unsafe)) {
        if (lane==0) solve_normal(s);
        return;
    }
    if (lane<elements) s.lu[lane]=element;
    if (lane<N) s.solution[lane]=rhs;
    if (lane==0) s.logdet_normal=0.0;
    __syncthreads();
    int sign=1,processed=0;
    #pragma unroll
    for (int j=0;j<N;++j) {
        int failed=0;
        if (lane==0) {
            int best=j;
            double magnitude=fabs(s.lu[j*N+j]);
            #pragma unroll
            for (int i=j+1;i<N;++i) {
                const double candidate=fabs(s.lu[i*N+j]);
                if (candidate>magnitude) {best=i;magnitude=candidate;}
            }
            if (!(magnitude>0.0) || !isfinite(magnitude)) failed=1;
            else s.pivot[j]=best;
        }
        // Predicate is local;block vote/barrier broadcasts abort and orders pivot.
        if (__syncthreads_or(failed)) {
            if (lane==0) s.status=1;
            break;
        }
        const int best=s.pivot[j];
        if (best!=j && lane<N) {
            const double saved=s.lu[j*N+lane];
            s.lu[j*N+lane]=s.lu[best*N+lane];
            s.lu[best*N+lane]=saved;
        }
        __syncthreads();
        const double diagonal=s.lu[j*N+j];
        if (lane==0) {
            if (best!=j) sign=-sign;
            if (diagonal<0.0) sign=-sign;
        }
        if (lane>j && lane<N) s.lu[lane*N+j]/=diagonal;
        __syncthreads();
        if (lane<elements && row>j && col>j)
            s.lu[lane]-=s.lu[row*N+j]*s.lu[j*N+col];
        // Complete all source reads and cell writes before the next pivot/swap.
        __syncthreads();
        processed=j+1;
    }
    // Later row swaps leave earlier diagonals fixed;add logs in original j order.
    if (lane<processed) s.temporary[lane]=log(fabs(s.lu[lane*N+lane]));
    __syncthreads();
    if (lane==0) {
        for (int j=0;j<processed;++j) s.logdet_normal+=s.temporary[j];
        if (s.status || !isfinite(s.logdet_normal) || sign<=0) {s.status=1;return;}
        solve_lu(s.lu,N,s.pivot,s.solution);
        for (int i=0;i<N;++i) if (!isfinite(s.solution[i])) s.status=3;
    }
    // Caller block barrier orders the lane-zero solution/status writes.
}
