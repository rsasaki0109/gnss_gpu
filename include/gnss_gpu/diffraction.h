#pragma once

namespace gnss_gpu {

// rx_ecef:[3], sat_ecef:[n_sat*3], edge_start/end/mid:[n_edge*3]
// Output arrays are allocated by caller.
// valid/excess/amplitude/fresnel_v/atten_db length: n_sat*n_edge
// point length: n_sat*n_edge*3
void compute_diffraction_candidates(
    const double* rx_ecef, const double* sat_ecef,
    const double* edge_start, const double* edge_end, const double* edge_mid,
    int n_sat, int n_edge,
    double max_edge_range_m, double max_ray_edge_distance_m,
    double max_excess_path_m, double wavelength_m,
    int* valid, double* excess, double* amplitude,
    double* fresnel_v, double* atten_db, double* point);

}  // namespace gnss_gpu
