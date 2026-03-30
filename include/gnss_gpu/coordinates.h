#pragma once

namespace gnss_gpu {

// WGS84 constants
constexpr double WGS84_A = 6378137.0;            // semi-major axis [m]
constexpr double WGS84_F = 1.0 / 298.257223563;  // flattening
constexpr double WGS84_B = WGS84_A * (1.0 - WGS84_F);  // semi-minor axis
constexpr double WGS84_E2 = 2.0 * WGS84_F - WGS84_F * WGS84_F;  // first eccentricity squared

// Speed of light [m/s]
constexpr double C_LIGHT = 299792458.0;

// ECEF to LLA (batch, GPU)
void ecef_to_lla(const double* ecef_x, const double* ecef_y, const double* ecef_z,
                 double* lat, double* lon, double* alt,
                 int n);

// LLA to ECEF (batch, GPU)
void lla_to_ecef(const double* lat, const double* lon, const double* alt,
                 double* ecef_x, double* ecef_y, double* ecef_z,
                 int n);

// Compute satellite azimuth and elevation from receiver ECEF position
// sat_ecef: [n_sat * 3] (x,y,z interleaved)
// az, el: [n_sat] output in radians
void satellite_azel(double rx, double ry, double rz,
                    const double* sat_ecef, double* az, double* el,
                    int n_sat);

}  // namespace gnss_gpu
