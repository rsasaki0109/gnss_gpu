#pragma once

namespace gnss_gpu {

// Saastamoinen tropospheric delay model
// Returns zenith delay in meters
// lat: receiver latitude [rad]
// alt: receiver altitude [m]
// el: satellite elevation [rad]
double tropo_saastamoinen(double lat, double alt, double el);

// Klobuchar ionospheric delay model (GPS broadcast)
// alpha, beta: 4 ionospheric parameters from GPS navigation message
// lat, lon: receiver geodetic position [rad]
// az, el: satellite azimuth and elevation [rad]
// gps_time: GPS time of week [s]
// Returns delay in meters (L1 frequency)
double iono_klobuchar(const double alpha[4], const double beta[4],
                      double lat, double lon, double az, double el,
                      double gps_time);

// GPU batch: apply tropospheric correction
void tropo_correction_batch(
    const double* rx_lla,        // [n_epoch, 3] (lat, lon, alt in rad, rad, m)
    const double* sat_el,        // [n_epoch, n_sat] elevation angles [rad]
    double* corrections,         // [n_epoch, n_sat] output delay corrections [m]
    int n_epoch, int n_sat);

// GPU batch: apply ionospheric correction
void iono_correction_batch(
    const double* rx_lla,        // [n_epoch, 3] (lat, lon, alt in rad, rad, m)
    const double* sat_az,        // [n_epoch, n_sat] azimuth angles [rad]
    const double* sat_el,        // [n_epoch, n_sat] elevation angles [rad]
    const double alpha[4], const double beta[4],
    const double* gps_times,    // [n_epoch]
    double* corrections,         // [n_epoch, n_sat] output delay corrections [m]
    int n_epoch, int n_sat);

}  // namespace gnss_gpu
