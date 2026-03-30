#pragma once

namespace gnss_gpu {

struct Triangle {
  double v0[3];
  double v1[3];
  double v2[3];
};

struct RayResult {
  bool is_los;
  double path_delay;
};

// Batch LOS check: for each satellite, test if ray from rx is blocked by any triangle
// rx_ecef: [3] receiver ECEF position
// sat_ecef: [n_sat * 3] satellite ECEF positions
// triangles: [n_tri] building triangles
// is_los: [n_sat] output, true if line-of-sight is clear
void raytrace_los_check(const double* rx_ecef, const double* sat_ecef,
                        const Triangle* triangles, int* is_los,
                        int n_sat, int n_tri);

// First-order multipath reflection computation
// rx_ecef: [3] receiver ECEF position
// sat_ecef: [n_sat * 3] satellite ECEF positions
// triangles: [n_tri] building triangles
// reflection_points: [n_sat * 3] output, closest reflection point per satellite
// excess_delays: [n_sat] output, excess path delay in meters (0 if no reflection)
void raytrace_multipath(const double* rx_ecef, const double* sat_ecef,
                        const Triangle* triangles, double* reflection_points,
                        double* excess_delays, int n_sat, int n_tri);

}  // namespace gnss_gpu
