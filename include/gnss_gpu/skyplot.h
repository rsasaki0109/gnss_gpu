#pragma once

namespace gnss_gpu {

struct GridQuality {
    double pdop, hdop, vdop, gdop;
    int n_visible;
};

// Compute DOP values for a grid of receiver positions
// grid_ecef: [n_grid * 3] receiver ECEF positions (x,y,z interleaved)
// sat_ecef:  [n_sat * 3]  satellite ECEF positions
// pdop, hdop, vdop, gdop: [n_grid] output DOP values
// n_visible: [n_grid] number of visible satellites per grid point
// elevation_mask_rad: minimum elevation angle [rad]
void compute_grid_quality(
    const double* grid_ecef, const double* sat_ecef,
    double* pdop, double* hdop, double* vdop, double* gdop, int* n_visible,
    int n_grid, int n_sat, double elevation_mask_rad);

// Compute sky visibility mask using 3D building geometry
// grid_ecef:  [n_grid * 3] receiver ECEF positions
// triangles:  [n_tri * 9]  triangle vertices (v0x,v0y,v0z, v1x,v1y,v1z, v2x,v2y,v2z)
// sky_mask:   [n_grid * n_az * n_el] output visibility (0=blocked, 1=visible)
// n_az, n_el: azimuth and elevation bin counts
void compute_sky_visibility(
    const double* grid_ecef, const double* triangles,
    float* sky_mask, int n_grid, int n_tri, int n_az, int n_el);

}  // namespace gnss_gpu
