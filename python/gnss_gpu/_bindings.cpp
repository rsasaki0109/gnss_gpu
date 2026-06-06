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
         int enable_line_search, py::object sys_kind_py, int n_clock,
         py::object motion_displacement_py,
         py::object tdcp_meas_py, py::object tdcp_weights_py, double tdcp_sigma_m,
         double tdcp_huber_k) {
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
        if (n_clock < 1 || n_clock > 7)
          throw std::runtime_error("fgo_gnss_lm: n_clock must be in 1..7");

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

        const double* md_ptr = nullptr;
        if (!motion_displacement_py.is_none()) {
          auto md_arr = py::cast<py::array_t<double>>(motion_displacement_py);
          auto mdr = md_arr.request();
          if (mdr.size != n_epoch * 3)
            throw std::runtime_error("motion_displacement must have n_epoch*3 elements");
          md_ptr = static_cast<const double*>(mdr.ptr);
        }

        const double* tdcp_meas_ptr = nullptr;
        if (!tdcp_meas_py.is_none()) {
          auto tm_arr = py::cast<py::array_t<double>>(tdcp_meas_py);
          auto tmr = tm_arr.request();
          if (tmr.size != (n_epoch - 1) * n_sat)
            throw std::runtime_error("fgo_gnss_lm: tdcp_meas must have (T-1)*S elements");
          tdcp_meas_ptr = static_cast<const double*>(tmr.ptr);
        }

        const double* tdcp_weights_ptr = nullptr;
        if (!tdcp_weights_py.is_none()) {
          auto tw_arr = py::cast<py::array_t<double>>(tdcp_weights_py);
          auto twr = tw_arr.request();
          if (twr.size != (n_epoch - 1) * n_sat)
            throw std::runtime_error("fgo_gnss_lm: tdcp_weights must have (T-1)*S elements");
          tdcp_weights_ptr = static_cast<const double*>(twr.ptr);
        }

        double mse = 0.0;
        int iters = gnss_gpu::fgo_gnss_lm(
            static_cast<double*>(bs.ptr), static_cast<double*>(bp.ptr), static_cast<double*>(bw.ptr),
            sk_ptr, n_clock, static_cast<double*>(bst.ptr), n_epoch, n_sat, motion_sigma_m, max_iter,
            tol, huber_k, enable_line_search, &mse, md_ptr,
            tdcp_meas_ptr, tdcp_weights_ptr, tdcp_sigma_m, tdcp_huber_k);
        return py::make_tuple(iters, mse);
      },
      "GPU FGO: PseudorangeFactor_XC-style clocks (h=[1,0..] or [1,1,0..]) + optional RW motion + TDCP. "
      "GN + host Cholesky; huber_k>0 enables IRLS Huber on z=|sqrt(w)*res|; "
      "enable_line_search uses backtracking on the GN step.",
      py::arg("sat_ecef"), py::arg("pseudorange"), py::arg("weights"), py::arg("state_io"),
      py::arg("motion_sigma_m") = 0.0, py::arg("max_iter") = 25, py::arg("tol") = 1e-3,
      py::arg("huber_k") = 0.0, py::arg("enable_line_search") = 1, py::arg("sys_kind") = py::none(),
      py::arg("n_clock") = 1,
      py::arg("motion_displacement") = py::none(),
      py::arg("tdcp_meas") = py::none(), py::arg("tdcp_weights") = py::none(),
      py::arg("tdcp_sigma_m") = 0.0,
      py::arg("tdcp_huber_k") = 0.0);

  // --- Extended FGO with velocity state + Doppler ---
  m.def(
      "fgo_gnss_lm_vd",
      [](py::array_t<double> sat_ecef, py::array_t<double> pseudorange, py::array_t<double> weights,
         py::array_t<double> state_io, double motion_sigma_m, double clock_drift_sigma_m,
         bool clock_use_average_drift,
         double stop_velocity_sigma_mps, double stop_position_sigma_m,
         int max_iter, double tol, double huber_k,
         int enable_line_search, py::object sys_kind_py, int n_clock,
         py::object sat_vel_py, py::object doppler_py, py::object doppler_weights_py,
         py::object dt_py, py::object stop_mask_py,
         py::object tdcp_meas_py, py::object tdcp_weights_py, double tdcp_sigma_m, bool tdcp_use_drift,
         double relative_height_sigma_m, py::object enu_up_ecef_py, py::object rel_height_i_py,
         py::object rel_height_j_py,
         py::object imu_delta_p_py, py::object imu_delta_v_py, py::object imu_delta_angle_py,
         py::object imu_delta_t_py,
         py::object imu_delta_p_bias_accel_jac_py, py::object imu_delta_v_bias_accel_jac_py,
         py::object imu_delta_p_bias_gyro_jac_py, py::object imu_delta_v_bias_gyro_jac_py,
         py::object imu_delta_angle_bias_gyro_jac_py,
         double imu_position_sigma_m, double imu_velocity_sigma_mps, double imu_attitude_sigma_rad,
         py::object imu_position_weights_py, py::object imu_velocity_weights_py, py::object imu_attitude_weights_py,
         py::object imu_preintegration_information_py,
         bool imu_factor_use_next_bias,
         py::object sat_clock_drift_py,
         py::object absolute_height_ref_ecef_py, double absolute_height_sigma_m,
         double imu_accel_bias_prior_sigma_mps2, double imu_accel_bias_between_sigma_mps2,
         py::object imu_accel_bias_between_weights_py,
         double imu_gyro_bias_prior_sigma_radps, double imu_gyro_bias_between_sigma_radps,
         py::object imu_gyro_bias_between_weights_py,
         double doppler_huber_k, double tdcp_huber_k,
	         py::object tdcp_linearization_ref_ecef_py,
	         double stop_velocity_huber_k, double stop_position_huber_k,
	         double relative_height_huber_k, double absolute_height_huber_k,
         py::object imu_gravity_py,
         py::object pr_linearization_ref_ecef_py, py::object pr_linearization_los_ecef_py,
         py::object doppler_linearization_ref_vel_py, py::object doppler_linearization_los_ecef_py,
         double stop_attitude_sigma_rad, double lm_damping) {
        auto bs = sat_ecef.request(), bp = pseudorange.request(), bw = weights.request();
        auto bst = state_io.request();
        if (bs.ndim != 3 || bp.ndim != 2 || bw.ndim != 2 || bst.ndim != 2)
          throw std::runtime_error(
              "fgo_gnss_lm_vd: expected sat_ecef [T,S,3], pr/weights [T,S], state [T,7+n_clock]");
        int n_epoch = static_cast<int>(bs.shape[0]);
        int n_sat = static_cast<int>(bs.shape[1]);
        const int base_ss = 7 + n_clock;
        const int ss = static_cast<int>(bst.shape[1]);
        const bool has_accel_bias_state = ss == base_ss + 3;
        const bool has_imu_bias_state = ss == base_ss + 6;
        const bool has_attitude_imu_bias_state = ss == base_ss + 9;
        const bool has_split_pose_attitude_imu_bias_state = ss == base_ss + 12;
        if (bp.shape[0] != n_epoch || bw.shape[0] != n_epoch ||
            static_cast<int>(bp.shape[1]) != n_sat || static_cast<int>(bw.shape[1]) != n_sat)
          throw std::runtime_error("fgo_gnss_lm_vd: pseudorange/weights shape mismatch");
        if (static_cast<int>(bst.shape[0]) != n_epoch ||
            (ss != base_ss && !has_accel_bias_state && !has_imu_bias_state &&
             !has_attitude_imu_bias_state && !has_split_pose_attitude_imu_bias_state))
          throw std::runtime_error("fgo_gnss_lm_vd: state_io must be [T, 7+n_clock], [T, 10+n_clock], [T, 13+n_clock], [T, 16+n_clock], or [T, 19+n_clock]");
        if (bst.readonly) throw std::runtime_error("fgo_gnss_lm_vd: state_io must be writable");
        if (n_clock < 1 || n_clock > 7)
          throw std::runtime_error("fgo_gnss_lm_vd: n_clock must be in 1..7");

        const std::int32_t* sk_ptr = nullptr;
        if (!sys_kind_py.is_none()) {
          auto sk_arr = py::cast<py::array_t<std::int32_t>>(sys_kind_py);
          auto skr = sk_arr.request();
          if (skr.ndim != 2 || skr.shape[0] != n_epoch || skr.shape[1] != n_sat)
            throw std::runtime_error("fgo_gnss_lm_vd: sys_kind must be int32 [T, S]");
          sk_ptr = static_cast<const std::int32_t*>(skr.ptr);
        } else if (n_clock > 1) {
          throw std::runtime_error("fgo_gnss_lm_vd: sys_kind is required when n_clock > 1");
        }

        const double* sv_ptr = nullptr;
        if (!sat_vel_py.is_none()) {
          auto sv_arr = py::cast<py::array_t<double>>(sat_vel_py);
          auto svr = sv_arr.request();
          if (svr.size != n_epoch * n_sat * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: sat_vel must have T*S*3 elements");
          sv_ptr = static_cast<const double*>(svr.ptr);
        }

        const double* dop_ptr = nullptr;
        if (!doppler_py.is_none()) {
          auto dop_arr = py::cast<py::array_t<double>>(doppler_py);
          auto dopr = dop_arr.request();
          if (dopr.size != n_epoch * n_sat)
            throw std::runtime_error("fgo_gnss_lm_vd: doppler must have T*S elements");
          dop_ptr = static_cast<const double*>(dopr.ptr);
        }

        const double* dw_ptr = nullptr;
        if (!doppler_weights_py.is_none()) {
          auto dw_arr = py::cast<py::array_t<double>>(doppler_weights_py);
          auto dwr = dw_arr.request();
          if (dwr.size != n_epoch * n_sat)
            throw std::runtime_error("fgo_gnss_lm_vd: doppler_weights must have T*S elements");
          dw_ptr = static_cast<const double*>(dwr.ptr);
        }

        const double* dt_ptr = nullptr;
        if (!dt_py.is_none()) {
          auto dt_arr = py::cast<py::array_t<double>>(dt_py);
          auto dtr = dt_arr.request();
          if (dtr.size != n_epoch)
            throw std::runtime_error("fgo_gnss_lm_vd: dt must have T elements");
          dt_ptr = static_cast<const double*>(dtr.ptr);
        }

        const std::uint8_t* stop_mask_ptr = nullptr;
        if (!stop_mask_py.is_none()) {
          auto stop_mask_arr = py::cast<py::array_t<std::uint8_t>>(stop_mask_py);
          auto smr = stop_mask_arr.request();
          if (smr.size != n_epoch)
            throw std::runtime_error("fgo_gnss_lm_vd: stop_mask must have T elements");
          stop_mask_ptr = static_cast<const std::uint8_t*>(smr.ptr);
        }

        const double* tdcp_meas_ptr = nullptr;
        if (!tdcp_meas_py.is_none()) {
          auto tm_arr = py::cast<py::array_t<double>>(tdcp_meas_py);
          auto tmr = tm_arr.request();
          if (tmr.size != (n_epoch - 1) * n_sat)
            throw std::runtime_error("fgo_gnss_lm_vd: tdcp_meas must have (T-1)*S elements");
          tdcp_meas_ptr = static_cast<const double*>(tmr.ptr);
        }

        const double* tdcp_weights_ptr = nullptr;
        if (!tdcp_weights_py.is_none()) {
          auto tw_arr = py::cast<py::array_t<double>>(tdcp_weights_py);
          auto twr = tw_arr.request();
          if (twr.size != (n_epoch - 1) * n_sat)
            throw std::runtime_error("fgo_gnss_lm_vd: tdcp_weights must have (T-1)*S elements");
          tdcp_weights_ptr = static_cast<const double*>(twr.ptr);
        }

        const double* enu_up_ptr = nullptr;
        double enu_buf[3] = {0.0, 0.0, 0.0};
        if (!enu_up_ecef_py.is_none()) {
          auto up_arr = py::cast<py::array_t<double>>(enu_up_ecef_py);
          auto upr = up_arr.request();
          if (upr.size != 3)
            throw std::runtime_error("fgo_gnss_lm_vd: enu_up_ecef must have 3 elements");
          enu_buf[0] = static_cast<const double*>(upr.ptr)[0];
          enu_buf[1] = static_cast<const double*>(upr.ptr)[1];
          enu_buf[2] = static_cast<const double*>(upr.ptr)[2];
          enu_up_ptr = enu_buf;
        }

        const std::int32_t* rel_i_ptr = nullptr;
        const std::int32_t* rel_j_ptr = nullptr;
        int n_rel_edges = 0;
        if (!rel_height_i_py.is_none() || !rel_height_j_py.is_none()) {
          if (rel_height_i_py.is_none() || rel_height_j_py.is_none())
            throw std::runtime_error("fgo_gnss_lm_vd: rel_height_edge_i and rel_height_edge_j must be both set or both none");
          auto ri_arr = py::cast<py::array_t<std::int32_t>>(rel_height_i_py);
          auto rj_arr = py::cast<py::array_t<std::int32_t>>(rel_height_j_py);
          auto rir = ri_arr.request();
          auto rjr = rj_arr.request();
          if (rir.size != rjr.size)
            throw std::runtime_error("fgo_gnss_lm_vd: rel_height edge arrays must match length");
          n_rel_edges = static_cast<int>(rir.size);
          rel_i_ptr = static_cast<const std::int32_t*>(rir.ptr);
          rel_j_ptr = static_cast<const std::int32_t*>(rjr.ptr);
        }

        const double* imu_dp_ptr = nullptr;
        if (!imu_delta_p_py.is_none()) {
          auto dp_arr = py::cast<py::array_t<double>>(imu_delta_p_py);
          auto dpr = dp_arr.request();
          if (dpr.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_p must have (T-1)*3 elements");
          imu_dp_ptr = static_cast<const double*>(dpr.ptr);
        }

        const double* imu_dv_ptr = nullptr;
        if (!imu_delta_v_py.is_none()) {
          auto dv_arr = py::cast<py::array_t<double>>(imu_delta_v_py);
          auto dvr = dv_arr.request();
          if (dvr.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_v must have (T-1)*3 elements");
          imu_dv_ptr = static_cast<const double*>(dvr.ptr);
        }

        const double* imu_da_ptr = nullptr;
        if (!imu_delta_angle_py.is_none()) {
          auto da_arr = py::cast<py::array_t<double>>(imu_delta_angle_py);
          auto dar = da_arr.request();
          if (dar.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_angle must have (T-1)*3 elements");
          imu_da_ptr = static_cast<const double*>(dar.ptr);
        }

        const double* imu_dt_ptr = nullptr;
        if (!imu_delta_t_py.is_none()) {
          auto dt_arr = py::cast<py::array_t<double>>(imu_delta_t_py);
          auto dtr = dt_arr.request();
          if (dtr.size != (n_epoch - 1))
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_t must have T-1 elements");
          imu_dt_ptr = static_cast<const double*>(dtr.ptr);
        }

        const double* imu_dp_ba_jac_ptr = nullptr;
        if (!imu_delta_p_bias_accel_jac_py.is_none()) {
          auto jac_arr = py::cast<py::array_t<double>>(imu_delta_p_bias_accel_jac_py);
          auto jacr = jac_arr.request();
          if (jacr.size != (n_epoch - 1) * 9)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_p_bias_accel_jac must have (T-1)*3*3 elements");
          imu_dp_ba_jac_ptr = static_cast<const double*>(jacr.ptr);
        }

        const double* imu_dv_ba_jac_ptr = nullptr;
        if (!imu_delta_v_bias_accel_jac_py.is_none()) {
          auto jac_arr = py::cast<py::array_t<double>>(imu_delta_v_bias_accel_jac_py);
          auto jacr = jac_arr.request();
          if (jacr.size != (n_epoch - 1) * 9)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_v_bias_accel_jac must have (T-1)*3*3 elements");
          imu_dv_ba_jac_ptr = static_cast<const double*>(jacr.ptr);
        }

        const double* imu_dp_bg_jac_ptr = nullptr;
        if (!imu_delta_p_bias_gyro_jac_py.is_none()) {
          auto jac_arr = py::cast<py::array_t<double>>(imu_delta_p_bias_gyro_jac_py);
          auto jacr = jac_arr.request();
          if (jacr.size != (n_epoch - 1) * 9)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_p_bias_gyro_jac must have (T-1)*3*3 elements");
          imu_dp_bg_jac_ptr = static_cast<const double*>(jacr.ptr);
        }

        const double* imu_dv_bg_jac_ptr = nullptr;
        if (!imu_delta_v_bias_gyro_jac_py.is_none()) {
          auto jac_arr = py::cast<py::array_t<double>>(imu_delta_v_bias_gyro_jac_py);
          auto jacr = jac_arr.request();
          if (jacr.size != (n_epoch - 1) * 9)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_v_bias_gyro_jac must have (T-1)*3*3 elements");
          imu_dv_bg_jac_ptr = static_cast<const double*>(jacr.ptr);
        }

        const double* imu_da_bg_jac_ptr = nullptr;
        if (!imu_delta_angle_bias_gyro_jac_py.is_none()) {
          auto jac_arr = py::cast<py::array_t<double>>(imu_delta_angle_bias_gyro_jac_py);
          auto jacr = jac_arr.request();
          if (jacr.size != (n_epoch - 1) * 9)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_delta_angle_bias_gyro_jac must have (T-1)*3*3 elements");
          imu_da_bg_jac_ptr = static_cast<const double*>(jacr.ptr);
        }

        const double* imu_pos_w_ptr = nullptr;
        if (!imu_position_weights_py.is_none()) {
          auto pw_arr = py::cast<py::array_t<double>>(imu_position_weights_py);
          auto pwr = pw_arr.request();
          if (pwr.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_position_weights must have (T-1)*3 elements");
          imu_pos_w_ptr = static_cast<const double*>(pwr.ptr);
        }

        const double* imu_vel_w_ptr = nullptr;
        if (!imu_velocity_weights_py.is_none()) {
          auto vw_arr = py::cast<py::array_t<double>>(imu_velocity_weights_py);
          auto vwr = vw_arr.request();
          if (vwr.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_velocity_weights must have (T-1)*3 elements");
          imu_vel_w_ptr = static_cast<const double*>(vwr.ptr);
        }

        const double* imu_att_w_ptr = nullptr;
        if (!imu_attitude_weights_py.is_none()) {
          auto aw_arr = py::cast<py::array_t<double>>(imu_attitude_weights_py);
          auto awr = aw_arr.request();
          if (awr.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_attitude_weights must have (T-1)*3 elements");
          imu_att_w_ptr = static_cast<const double*>(awr.ptr);
        }

        const double* imu_pva_info_ptr = nullptr;
        if (!imu_preintegration_information_py.is_none()) {
          auto info_arr = py::cast<py::array_t<double>>(imu_preintegration_information_py);
          auto infor = info_arr.request();
          if (infor.size != (n_epoch - 1) * 81)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_preintegration_information must have (T-1)*9*9 elements");
          imu_pva_info_ptr = static_cast<const double*>(infor.ptr);
        }

        const double* scd_ptr = nullptr;
        if (!sat_clock_drift_py.is_none()) {
          auto scd_arr = py::cast<py::array_t<double>>(sat_clock_drift_py);
          auto scdr = scd_arr.request();
          if (scdr.size != n_epoch * n_sat)
            throw std::runtime_error("fgo_gnss_lm_vd: sat_clock_drift must have T*S elements");
          scd_ptr = static_cast<const double*>(scdr.ptr);
        }

        const double* abs_height_ref_ptr = nullptr;
        if (!absolute_height_ref_ecef_py.is_none()) {
          auto abs_arr = py::cast<py::array_t<double>>(absolute_height_ref_ecef_py);
          auto absr = abs_arr.request();
          if (absr.size != n_epoch * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: absolute_height_ref_ecef must have T*3 elements");
          abs_height_ref_ptr = static_cast<const double*>(absr.ptr);
        }

        const double* imu_accel_bias_between_w_ptr = nullptr;
        if (!imu_accel_bias_between_weights_py.is_none()) {
          auto bw_arr = py::cast<py::array_t<double>>(imu_accel_bias_between_weights_py);
          auto bwr = bw_arr.request();
          if (bwr.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_accel_bias_between_weights must have (T-1)*3 elements");
          imu_accel_bias_between_w_ptr = static_cast<const double*>(bwr.ptr);
        }

        const double* imu_gyro_bias_between_w_ptr = nullptr;
        if (!imu_gyro_bias_between_weights_py.is_none()) {
          auto bw_arr = py::cast<py::array_t<double>>(imu_gyro_bias_between_weights_py);
          auto bwr = bw_arr.request();
          if (bwr.size != (n_epoch - 1) * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: imu_gyro_bias_between_weights must have (T-1)*3 elements");
          imu_gyro_bias_between_w_ptr = static_cast<const double*>(bwr.ptr);
        }

	        const double* tdcp_ref_ptr = nullptr;
	        if (!tdcp_linearization_ref_ecef_py.is_none()) {
	          auto ref_arr = py::cast<py::array_t<double>>(tdcp_linearization_ref_ecef_py);
          auto refr = ref_arr.request();
          if (refr.size != n_epoch * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: tdcp_linearization_ref_ecef must have T*3 elements");
	          tdcp_ref_ptr = static_cast<const double*>(refr.ptr);
	        }

	        const double* imu_gravity_ptr = nullptr;
	        if (!imu_gravity_py.is_none()) {
	          if (!has_attitude_imu_bias_state && !has_split_pose_attitude_imu_bias_state)
	            throw std::runtime_error("fgo_gnss_lm_vd: imu_gravity requires [T, 16+n_clock] or [T, 19+n_clock] attitude/bias state");
	          auto gravity_arr = py::cast<py::array_t<double>>(imu_gravity_py);
	          auto gravityr = gravity_arr.request();
	          if (gravityr.size != (n_epoch - 1) * 3)
	            throw std::runtime_error("fgo_gnss_lm_vd: imu_gravity must have (T-1)*3 elements");
	          imu_gravity_ptr = static_cast<const double*>(gravityr.ptr);
	        }

        if (pr_linearization_ref_ecef_py.is_none() != pr_linearization_los_ecef_py.is_none())
          throw std::runtime_error("fgo_gnss_lm_vd: pr_linearization_ref_ecef and pr_linearization_los_ecef must be both set or both none");
        const double* pr_ref_ptr = nullptr;
        if (!pr_linearization_ref_ecef_py.is_none()) {
          auto ref_arr = py::cast<py::array_t<double>>(pr_linearization_ref_ecef_py);
          auto refr = ref_arr.request();
          if (refr.size != n_epoch * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: pr_linearization_ref_ecef must have T*3 elements");
          pr_ref_ptr = static_cast<const double*>(refr.ptr);
        }
        const double* pr_los_ptr = nullptr;
        if (!pr_linearization_los_ecef_py.is_none()) {
          auto los_arr = py::cast<py::array_t<double>>(pr_linearization_los_ecef_py);
          auto losr = los_arr.request();
          if (losr.size != n_epoch * n_sat * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: pr_linearization_los_ecef must have T*S*3 elements");
          pr_los_ptr = static_cast<const double*>(losr.ptr);
        }

        if (doppler_linearization_ref_vel_py.is_none() != doppler_linearization_los_ecef_py.is_none())
          throw std::runtime_error("fgo_gnss_lm_vd: doppler_linearization_ref_vel and doppler_linearization_los_ecef must be both set or both none");
        const double* doppler_ref_vel_ptr = nullptr;
        if (!doppler_linearization_ref_vel_py.is_none()) {
          auto ref_arr = py::cast<py::array_t<double>>(doppler_linearization_ref_vel_py);
          auto refr = ref_arr.request();
          if (refr.size != n_epoch * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: doppler_linearization_ref_vel must have T*3 elements");
          doppler_ref_vel_ptr = static_cast<const double*>(refr.ptr);
        }
        const double* doppler_los_ptr = nullptr;
        if (!doppler_linearization_los_ecef_py.is_none()) {
          auto los_arr = py::cast<py::array_t<double>>(doppler_linearization_los_ecef_py);
          auto losr = los_arr.request();
          if (losr.size != n_epoch * n_sat * 3)
            throw std::runtime_error("fgo_gnss_lm_vd: doppler_linearization_los_ecef must have T*S*3 elements");
          doppler_los_ptr = static_cast<const double*>(losr.ptr);
        }

	        double mse = 0.0;
	        int iters = gnss_gpu::fgo_gnss_lm_vd(
            static_cast<double*>(bs.ptr), static_cast<double*>(bp.ptr), static_cast<double*>(bw.ptr),
            sk_ptr, n_clock, static_cast<double*>(bst.ptr), n_epoch, n_sat,
            motion_sigma_m, clock_drift_sigma_m, clock_use_average_drift,
            stop_velocity_sigma_mps, stop_position_sigma_m, max_iter,
            tol, huber_k, enable_line_search, &mse,
            sv_ptr, dop_ptr, dw_ptr, dt_ptr, stop_mask_ptr,
            tdcp_meas_ptr, tdcp_weights_ptr, tdcp_sigma_m, tdcp_use_drift,
            relative_height_sigma_m, enu_up_ptr, n_rel_edges, rel_i_ptr, rel_j_ptr,
            imu_dp_ptr, imu_dv_ptr, imu_da_ptr, imu_dt_ptr,
            imu_dp_ba_jac_ptr, imu_dv_ba_jac_ptr,
            imu_dp_bg_jac_ptr, imu_dv_bg_jac_ptr, imu_da_bg_jac_ptr,
            imu_position_sigma_m, imu_velocity_sigma_mps,
            imu_attitude_sigma_rad,
            imu_pos_w_ptr, imu_vel_w_ptr, imu_att_w_ptr, imu_pva_info_ptr,
            imu_factor_use_next_bias, scd_ptr, abs_height_ref_ptr, absolute_height_sigma_m,
            ss, imu_accel_bias_prior_sigma_mps2, imu_accel_bias_between_sigma_mps2,
            imu_accel_bias_between_w_ptr,
	            imu_gyro_bias_prior_sigma_radps, imu_gyro_bias_between_sigma_radps,
	            imu_gyro_bias_between_w_ptr,
	            doppler_huber_k, tdcp_huber_k, tdcp_ref_ptr,
            stop_velocity_huber_k, stop_position_huber_k,
            relative_height_huber_k, absolute_height_huber_k, imu_gravity_ptr,
            pr_ref_ptr, pr_los_ptr, doppler_ref_vel_ptr, doppler_los_ptr,
            stop_attitude_sigma_rad, lm_damping);
        return py::make_tuple(iters, mse);
      },
      "GPU FGO with velocity state + Doppler factor + optional TDCP. "
      "State: [x,y,z,vx,vy,vz,clk...,drift] per epoch; optionally append [bax,bay,baz], [bax,bay,baz,bgx,bgy,bgz], or [attx,atty,attz,bax,bay,baz,bgx,bgy,bgz]. "
      "Motion factor couples position/velocity: x_{t+1}-x_t = (v_t+v_{t+1})*dt/2. "
      "Clock drift factor: clk_{t+1} = clk_t + drift_t*dt. "
      "Optional stop factors add v_t=0 priors and x_t=x_t+1 hold factors on stop epochs. "
      "Optional IMU priors constrain x/v deltas between adjacent epochs. "
      "Optional height priors constrain ENU-up absolute or loop-closure height. "
      "Doppler factor constrains velocity and drift. "
      "TDCP factor provides cm-level inter-epoch constraints from carrier phase.",
      py::arg("sat_ecef"), py::arg("pseudorange"), py::arg("weights"), py::arg("state_io"),
      py::arg("motion_sigma_m") = 0.0, py::arg("clock_drift_sigma_m") = 0.0,
      py::arg("clock_use_average_drift") = false,
      py::arg("stop_velocity_sigma_mps") = 0.0, py::arg("stop_position_sigma_m") = 0.0,
      py::arg("max_iter") = 25, py::arg("tol") = 1e-3,
      py::arg("huber_k") = 0.0, py::arg("enable_line_search") = 1,
      py::arg("sys_kind") = py::none(), py::arg("n_clock") = 1,
      py::arg("sat_vel") = py::none(), py::arg("doppler") = py::none(),
      py::arg("doppler_weights") = py::none(), py::arg("dt") = py::none(),
      py::arg("stop_mask") = py::none(),
      py::arg("tdcp_meas") = py::none(), py::arg("tdcp_weights") = py::none(),
      py::arg("tdcp_sigma_m") = 0.0, py::arg("tdcp_use_drift") = false,
      py::arg("relative_height_sigma_m") = 0.0, py::arg("enu_up_ecef") = py::none(),
      py::arg("rel_height_edge_i") = py::none(), py::arg("rel_height_edge_j") = py::none(),
      py::arg("imu_delta_p") = py::none(), py::arg("imu_delta_v") = py::none(),
      py::arg("imu_delta_angle") = py::none(),
      py::arg("imu_delta_t") = py::none(),
      py::arg("imu_delta_p_bias_accel_jac") = py::none(),
      py::arg("imu_delta_v_bias_accel_jac") = py::none(),
      py::arg("imu_delta_p_bias_gyro_jac") = py::none(),
      py::arg("imu_delta_v_bias_gyro_jac") = py::none(),
      py::arg("imu_delta_angle_bias_gyro_jac") = py::none(),
      py::arg("imu_position_sigma_m") = 0.0, py::arg("imu_velocity_sigma_mps") = 0.0,
      py::arg("imu_attitude_sigma_rad") = 0.0,
      py::arg("imu_position_weights") = py::none(),
      py::arg("imu_velocity_weights") = py::none(),
      py::arg("imu_attitude_weights") = py::none(),
      py::arg("imu_preintegration_information") = py::none(),
      py::arg("imu_factor_use_next_bias") = false,
      py::arg("sat_clock_drift") = py::none(),
      py::arg("absolute_height_ref_ecef") = py::none(),
      py::arg("absolute_height_sigma_m") = 0.0,
      py::arg("imu_accel_bias_prior_sigma_mps2") = 0.0,
      py::arg("imu_accel_bias_between_sigma_mps2") = 0.0,
      py::arg("imu_accel_bias_between_weights") = py::none(),
      py::arg("imu_gyro_bias_prior_sigma_radps") = 0.0,
      py::arg("imu_gyro_bias_between_sigma_radps") = 0.0,
      py::arg("imu_gyro_bias_between_weights") = py::none(),
      py::arg("doppler_huber_k") = 0.0,
      py::arg("tdcp_huber_k") = 0.0,
	      py::arg("tdcp_linearization_ref_ecef") = py::none(),
	      py::arg("stop_velocity_huber_k") = 0.0,
	      py::arg("stop_position_huber_k") = 0.0,
	      py::arg("relative_height_huber_k") = 0.0,
	      py::arg("absolute_height_huber_k") = 0.0,
	      py::arg("imu_gravity") = py::none(),
      py::arg("pr_linearization_ref_ecef") = py::none(),
      py::arg("pr_linearization_los_ecef") = py::none(),
      py::arg("doppler_linearization_ref_vel") = py::none(),
      py::arg("doppler_linearization_los_ecef") = py::none(),
      py::arg("stop_attitude_sigma_rad") = 0.0,
      py::arg("lm_damping") = 0.0);
}
