// All 32 lanes of warp zero participate. Keep pivot selection and each
// element's division/multiply/subtract order identical to decompose().
__device__ void solve_normal_warp(Scratch& s, int lane) {
    const int n = s.dimension;
    for (int k = lane; k < n*n; k += 32)
        s.lu[k] = s.normal[(k/n)*MAX_D+k%n];
    if (lane < n) s.solution[lane] = s.rhs[lane];
    if (lane == 0) s.logdet_normal = 0.0;
    __syncwarp();
    int sign = 1;
    int processed = 0;
    for (int j = 0; j < n; ++j) {
        if (lane == 0) {
            int best = j;
            double magnitude = fabs(s.lu[j*n+j]);
            for (int i = j+1; i < n; ++i) {
                const double candidate = fabs(s.lu[i*n+j]);
                if (candidate > magnitude) { best = i; magnitude = candidate; }
            }
            if (!(magnitude > 0.0) || !isfinite(magnitude)) s.status = 1;
            else s.pivot[j] = best;
        }
        __syncwarp();
        if (s.status) break;
        const int best = s.pivot[j];
        if (best != j && lane < n) {
            const double saved = s.lu[j*n+lane];
            s.lu[j*n+lane] = s.lu[best*n+lane];
            s.lu[best*n+lane] = saved;
        }
        __syncwarp();
        const double diagonal = s.lu[j*n+j];
        if (lane == 0) {
            if (best != j) sign = -sign;
            if (diagonal < 0.0) sign = -sign;
        }
        if (lane > j && lane < n) {
            s.lu[lane*n+j] /= diagonal;
            for (int k = j+1; k < n; ++k)
                s.lu[lane*n+k] -= s.lu[lane*n+j] * s.lu[j*n+k];
        }
        __syncwarp();
        processed = j+1;
    }
    // Later pivots never change an earlier diagonal. Compute each original
    // log independently, then add in the original column order. On failure,
    // retain the same partial logdet diagnostics as the serial implementation.
    if (lane < processed) s.temporary[lane] = log(fabs(s.lu[lane*n+lane]));
    __syncwarp();
    if (lane == 0) {
        for (int j = 0; j < processed; ++j) s.logdet_normal += s.temporary[j];
        if (s.status || !isfinite(s.logdet_normal) || sign <= 0) { s.status = 1; return; }
        solve_lu(s.lu, n, s.pivot, s.solution);
        for (int i = 0; i < n; ++i) if (!isfinite(s.solution[i])) s.status = 3;
    }
}
