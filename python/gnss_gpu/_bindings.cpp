#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/coordinates.h"
#include "gnss_gpu/fgo.h"
#include "gnss_gpu/positioning.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu, m) {
  m.doc() = "GPU-accelerated GNSS signal processing";

  // --- Coordinate transforms ---
  m.def("ecef_to_lla", [](py::array_t<double> x, py::array_t<double> y, py::array_t<double> z) {
    auto bx = x.request(), by = y.request(), bz = z.request();
    int n = bx.size;
    auto lat = py::array_t<double>(n);
    auto lon = py::array_t<double>(n);
    auto alt = py::array_t<double>(n);
    gnss_gpu::ecef_to_lla(static_cast<double*>(bx.ptr), static_cast<double*>(by.ptr),
                          static_cast<double*>(bz.ptr),
                          static_cast<double*>(lat.request().ptr),
                          static_cast<double*>(lon.request().ptr),
                          static_cast<double*>(alt.request().ptr), n);
    return py::make_tuple(lat, lon, alt);
  }, "Convert ECEF to LLA (radians)", py::arg("x"), py::arg("y"), py::arg("z"));

  m.def("lla_to_ecef", [](py::array_t<double> lat, py::array_t<double> lon, py::array_t<double> alt) {
    auto bla = lat.request(), blo = lon.request(), bal = alt.request();
    int n = bla.size;
    auto x = py::array_t<double>(n);
    auto y = py::array_t<double>(n);
    auto z = py::array_t<double>(n);
    gnss_gpu::lla_to_ecef(static_cast<double*>(bla.ptr), static_cast<double*>(blo.ptr),
                          static_cast<double*>(bal.ptr),
                          static_cast<double*>(x.request().ptr),
                          static_cast<double*>(y.request().ptr),
                          static_cast<double*>(z.request().ptr), n);
    return py::make_tuple(x, y, z);
  }, "Convert LLA (radians) to ECEF", py::arg("lat"), py::arg("lon"), py::arg("alt"));

  m.def("satellite_azel", [](double rx, double ry, double rz, py::array_t<double> sat_ecef) {
    auto buf = sat_ecef.request();
    int n_sat = buf.size / 3;  // accept flat or (N,3)
    auto az = py::array_t<double>(n_sat);
    auto el = py::array_t<double>(n_sat);
    gnss_gpu::satellite_azel(rx, ry, rz, static_cast<double*>(buf.ptr),
                             static_cast<double*>(az.request().ptr),
                             static_cast<double*>(el.request().ptr), n_sat);
    return py::make_tuple(az, el);
  }, "Compute satellite azimuth/elevation from receiver ECEF position",
     py::arg("rx"), py::arg("ry"), py::arg("rz"), py::arg("sat_ecef"));

  // --- Positioning ---
  m.def("wls_position", [](py::array_t<double> sat_ecef, py::array_t<double> pseudoranges,
                            py::array_t<double> weights, int max_iter, double tol) {
    auto bs = sat_ecef.request(), bp = pseudoranges.request(), bw = weights.request();
    int n_sat = bp.size;
    if (n_sat < 4)
      throw std::runtime_error("wls_position requires at least 4 satellites");
    auto result = py::array_t<double>({4}, {sizeof(double)});
    double* result_ptr = result.mutable_data();
    int iters = gnss_gpu::wls_position(static_cast<double*>(bs.ptr),
                                        static_cast<double*>(bp.ptr),
                                        static_cast<double*>(bw.ptr),
                                        result_ptr,
                                        n_sat, max_iter, tol);
    return py::make_tuple(result, iters);
  }, "Single-epoch WLS positioning",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("max_iter") = 10, py::arg("tol") = 1e-4);

  m.def("wls_batch", [](py::array_t<double> sat_ecef, py::array_t<double> pseudoranges,
                         py::array_t<double> weights, int max_iter, double tol) {
    auto bs = sat_ecef.request(), bp = pseudoranges.request();
    int n_epoch = bs.shape[0];
    int n_sat = bs.shape[1];
    if (n_sat < 4)
      throw std::runtime_error("wls_batch requires at least 4 satellites");
    auto results = py::array_t<double>({n_epoch, 4});
    auto iters = py::array_t<int>(n_epoch);
    gnss_gpu::wls_batch(static_cast<double*>(bs.ptr),
                         static_cast<double*>(bp.ptr),
                         static_cast<double*>(weights.request().ptr),
                         static_cast<double*>(results.request().ptr),
                         static_cast<int*>(iters.request().ptr),
                         n_epoch, n_sat, max_iter, tol);
    return py::make_tuple(results, iters);
  }, "Batch WLS positioning (GPU parallel)",
     py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
     py::arg("max_iter") = 10, py::arg("tol") = 1e-4);

  m.def(
      "fgo_gnss_lm",
      [](py::array_t<double> sat_ecef, py::array_t<double> pseudorange, py::array_t<double> weights,
         py::array_t<double> state_io, double motion_sigma_m, int max_iter, double tol, double huber_k,
         int enable_line_search, py::object sys_kind_py, int n_clock) {
        auto bs = sat_ecef.request(), bp = pseudorange.request(), bw = weights.request();
        auto bst = state_io.request();
        if (bs.ndim != 3 || bp.ndim != 2 || bw.ndim != 2 || bst.ndim != 2)
          throw std::runtime_error(
              "fgo_gnss_lm: expected sat_ecef [T,S,3], pr/weights [T,S], state [T,3+n_clock]");
        int n_epoch = static_cast<int>(bs.shape[0]);
        int n_sat = static_cast<int>(bs.shape[1]);
        const int ss = 3 + n_clock;
        if (bp.shape[0] != n_epoch || bw.shape[0] != n_epoch ||
            static_cast<int>(bp.shape[1]) != n_sat || static_cast<int>(bw.shape[1]) != n_sat)
          throw std::runtime_error("fgo_gnss_lm: pseudorange/weights shape mismatch");
        if (static_cast<int>(bst.shape[0]) != n_epoch || static_cast<int>(bst.shape[1]) != ss)
          throw std::runtime_error("fgo_gnss_lm: state_io must be [T, 3+n_clock]");
        if (bst.readonly) throw std::runtime_error("fgo_gnss_lm: state_io must be writable");
        if (n_clock < 1 || n_clock > 4)
          throw std::runtime_error("fgo_gnss_lm: n_clock must be in 1..4");

        const std::int32_t* sk_ptr = nullptr;
        if (!sys_kind_py.is_none()) {
          auto sk_arr = py::cast<py::array_t<std::int32_t>>(sys_kind_py);
          auto skr = sk_arr.request();
          if (skr.ndim != 2 || skr.shape[0] != n_epoch || skr.shape[1] != n_sat)
            throw std::runtime_error("fgo_gnss_lm: sys_kind must be int32 [T, S]");
          sk_ptr = static_cast<const std::int32_t*>(skr.ptr);
        } else if (n_clock > 1) {
          throw std::runtime_error("fgo_gnss_lm: sys_kind is required when n_clock > 1");
        }

        double mse = 0.0;
        int iters = gnss_gpu::fgo_gnss_lm(
            static_cast<double*>(bs.ptr), static_cast<double*>(bp.ptr), static_cast<double*>(bw.ptr),
            sk_ptr, n_clock, static_cast<double*>(bst.ptr), n_epoch, n_sat, motion_sigma_m, max_iter,
            tol, huber_k, enable_line_search, &mse);
        return py::make_tuple(iters, mse);
      },
      "GPU FGO: PseudorangeFactor_XC-style clocks (h=[1,0..] or [1,1,0..]) + optional RW motion. "
      "GN + host Cholesky; huber_k>0 enables IRLS Huber on z=|sqrt(w)*res|; "
      "enable_line_search uses backtracking on the GN step.",
      py::arg("sat_ecef"), py::arg("pseudorange"), py::arg("weights"), py::arg("state_io"),
      py::arg("motion_sigma_m") = 0.0, py::arg("max_iter") = 25, py::arg("tol") = 1e-3,
      py::arg("huber_k") = 0.0, py::arg("enable_line_search") = 1, py::arg("sys_kind") = py::none(),
      py::arg("n_clock") = 1);
}
