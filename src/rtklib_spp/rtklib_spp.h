/*------------------------------------------------------------------------------
 * rtklib_spp.h: C wrapper for RTKLIB pntpos SPP measurement export
 *
 * Provides a library interface equivalent to the export_spp_meas CLI tool,
 * returning structured measurement data instead of writing CSV to stdout.
 *
 * NOTE: CMake integration requires building with the rtklib_c static library.
 *       See CMakeLists.txt for the RTKLIB source list and compile definitions.
 *-----------------------------------------------------------------------------*/
#ifndef RTKLIB_SPP_H
#define RTKLIB_SPP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int gps_week;
    double gps_tow;
    int prn;
    char sat_id[8];
    double prange_m;
    double r_m;
    double iono_m;
    double trop_m;
    double sat_clk_m;
    double satx, saty, satz;
    double el_rad;
    double var_total;
} rtklib_spp_meas_t;

typedef struct {
    rtklib_spp_meas_t *meas;
    int n_meas;
    int capacity;
} rtklib_spp_result_t;

/* Run pntpos-based SPP and return per-satellite measurements.
 * Returns 0 on success, nonzero on error.
 * Caller must free result with rtklib_spp_free(). */
int rtklib_spp_export(const char *obs_file, const char *nav_file,
                      double el_mask_deg, rtklib_spp_result_t *result);

void rtklib_spp_free(rtklib_spp_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* RTKLIB_SPP_H */
