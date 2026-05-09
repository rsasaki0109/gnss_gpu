/*------------------------------------------------------------------------------
 * _rtklib_spp_bindings.cpp: pybind11 wrapper for rtklib_spp C library
 *
 * Exposes rtklib_spp_export() as a Python function returning a dict of numpy
 * arrays, replacing the subprocess+CSV workflow in gtsam_public_dataset.py.
 *-----------------------------------------------------------------------------*/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "rtklib_spp/rtklib_spp.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_rtklib_spp, m) {
    m.doc() = "RTKLIB pntpos SPP measurement export (pybind11 wrapper)";

    m.def("export_spp_meas",
        [](const std::string &obs_file, const std::string &nav_file,
           double el_mask_deg) -> py::dict
        {
            rtklib_spp_result_t result = {};
            int stat = rtklib_spp_export(obs_file.c_str(), nav_file.c_str(),
                                         el_mask_deg, &result);
            if (stat != 0) {
                rtklib_spp_free(&result);
                throw std::runtime_error("rtklib_spp_export failed");
            }

            const ssize_t n = result.n_meas;

            // Allocate plain C arrays, copy, then wrap in numpy
            std::vector<int>    v_gps_week(n);
            std::vector<double> v_gps_tow(n);
            std::vector<int>    v_prn(n);
            py::list            sat_id_list;
            std::vector<double> v_prange_m(n);
            std::vector<double> v_r_m(n);
            std::vector<double> v_iono_m(n);
            std::vector<double> v_trop_m(n);
            std::vector<double> v_sat_clk_m(n);
            std::vector<double> v_satx(n);
            std::vector<double> v_saty(n);
            std::vector<double> v_satz(n);
            std::vector<double> v_el_rad(n);
            std::vector<double> v_var_total(n);

            for (ssize_t i = 0; i < n; i++) {
                const rtklib_spp_meas_t &row = result.meas[i];
                v_gps_week[i]  = row.gps_week;
                v_gps_tow[i]   = row.gps_tow;
                v_prn[i]       = row.prn;
                sat_id_list.append(py::str(row.sat_id));
                v_prange_m[i]  = row.prange_m;
                v_r_m[i]       = row.r_m;
                v_iono_m[i]    = row.iono_m;
                v_trop_m[i]    = row.trop_m;
                v_sat_clk_m[i] = row.sat_clk_m;
                v_satx[i]      = row.satx;
                v_saty[i]      = row.saty;
                v_satz[i]      = row.satz;
                v_el_rad[i]    = row.el_rad;
                v_var_total[i] = row.var_total;
            }

            rtklib_spp_free(&result);

            // Helper: create numpy array from std::vector (copies data).
            // Use explicit shape+strides to avoid the zero-stride broadcast
            // bug in some pybind11/numpy combinations.
            auto make_np_double = [&](const std::vector<double> &v) {
                ssize_t sz = static_cast<ssize_t>(v.size());
                auto arr = py::array_t<double>(
                    {sz},                               // shape
                    {static_cast<ssize_t>(sizeof(double))}  // strides
                );
                std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(double));
                return arr;
            };
            auto make_np_int = [&](const std::vector<int> &v) {
                ssize_t sz = static_cast<ssize_t>(v.size());
                auto arr = py::array_t<int>(
                    {sz},
                    {static_cast<ssize_t>(sizeof(int))}
                );
                std::memcpy(arr.mutable_data(), v.data(), v.size() * sizeof(int));
                return arr;
            };

            py::dict out;
            out["gps_week"]  = make_np_int(v_gps_week);
            out["gps_tow"]   = make_np_double(v_gps_tow);
            out["prn"]       = make_np_int(v_prn);
            out["sat_id"]    = sat_id_list;
            out["prange_m"]  = make_np_double(v_prange_m);
            out["r_m"]       = make_np_double(v_r_m);
            out["iono_m"]    = make_np_double(v_iono_m);
            out["trop_m"]    = make_np_double(v_trop_m);
            out["sat_clk_m"] = make_np_double(v_sat_clk_m);
            out["satx"]      = make_np_double(v_satx);
            out["saty"]      = make_np_double(v_saty);
            out["satz"]      = make_np_double(v_satz);
            out["el_rad"]    = make_np_double(v_el_rad);
            out["var_total"] = make_np_double(v_var_total);
            return out;
        },
        py::arg("obs_file"),
        py::arg("nav_file"),
        py::arg("el_mask_deg") = 15.0,
        "Run RTKLIB pntpos SPP and return per-satellite measurements as dict "
        "of numpy arrays.  Keys: gps_week, gps_tow, prn, sat_id, prange_m, "
        "r_m, iono_m, trop_m, sat_clk_m, satx, saty, satz, el_rad, var_total."
    );
}
