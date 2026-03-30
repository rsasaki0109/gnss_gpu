#pragma once

namespace gnss_gpu {

// GPS broadcast ephemeris parameters (IS-GPS-200)
struct EphemerisParams {
    double sqrt_a;     // sqrt of semi-major axis [m^0.5]
    double e;          // eccentricity
    double i0;         // inclination at reference time [rad]
    double omega0;     // longitude of ascending node at reference [rad]
    double omega;      // argument of perigee [rad]
    double M0;         // mean anomaly at reference [rad]
    double delta_n;    // mean motion correction [rad/s]
    double omega_dot;  // rate of right ascension [rad/s]
    double idot;       // rate of inclination [rad/s]
    double cuc, cus;   // argument of latitude corrections [rad]
    double crc, crs;   // orbit radius corrections [m]
    double cic, cis;   // inclination corrections [rad]
    double toe;        // time of ephemeris (GPS seconds of week)
    double af0;        // clock bias [s]
    double af1;        // clock drift [s/s]
    double af2;        // clock drift rate [s/s^2]
    double toc;        // time of clock (GPS seconds of week)
    double tgd;        // group delay [s]
    int week;          // GPS week number
};

// Compute satellite position + clock correction for batch of satellites at given time
// params: [n_sat] ephemeris parameters
// gps_time: GPS seconds of week
// sat_pos: [n_sat * 3] output ECEF positions (x,y,z interleaved)
// sat_clk: [n_sat] output clock corrections [s]
void compute_satellite_position(
    const EphemerisParams* params, double gps_time,
    double* sat_pos, double* sat_clk,
    int n_sat);

// Batch: compute for multiple epochs
// params: [n_sat] ephemeris parameters (same for all epochs)
// gps_times: [n_epoch] GPS seconds of week
// sat_pos: [n_epoch * n_sat * 3] output ECEF positions
// sat_clk: [n_epoch * n_sat] output clock corrections [s]
void compute_satellite_position_batch(
    const EphemerisParams* params, const double* gps_times,
    double* sat_pos, double* sat_clk,
    int n_epoch, int n_sat);

}  // namespace gnss_gpu
