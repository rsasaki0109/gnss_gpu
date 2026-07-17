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

// UTD (Kouyoumjian-Pathak) wedge diffraction candidate search.
//
// rx_ecef:[3], sat_ecef:[n_sat*3], edge_start/end/mid:[n_edge*3],
// face_dir_a/face_dir_b:[n_edge*3].
// wedge_n:[n_wedge_n] mirrors utd_diffraction._wedge_n_at on the host:
//   n_wedge_n == 0        -> every edge uses the default wedge angle 2.0
//   n_wedge_n == 1        -> that single value is broadcast to every edge
//   n_wedge_n == n_edge   -> per-edge values (out-of-range/non-finite/<=0
//                            entries fall back to 2.0)
// mode: 0 = absorbing, 1 = soft, 2 = hard (mirrors utd_coefficient's mode).
//
// Output arrays are allocated by caller.
// valid/excess/amplitude/beta0/phi/phi_p/wedge_n_out/atten_db length:
//   n_sat*n_edge
// point length: n_sat*n_edge*3
void compute_utd_diffraction_candidates(
    const double* rx_ecef, const double* sat_ecef,
    const double* edge_start, const double* edge_end, const double* edge_mid,
    const double* face_dir_a, const double* face_dir_b,
    const double* wedge_n, int n_wedge_n,
    int n_sat, int n_edge,
    double max_edge_range_m, double max_ray_edge_distance_m,
    double max_excess_path_m, double wavelength_m,
    int mode,
    int* valid, double* excess, double* amplitude,
    double* beta0_out, double* phi_out, double* phi_p_out,
    double* wedge_n_out, double* atten_db, double* point);

}  // namespace gnss_gpu
