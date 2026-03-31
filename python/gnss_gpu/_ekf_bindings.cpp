#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "gnss_gpu/ekf.h"
#include <cstring>

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_ekf, m) {
    m.doc() = "GPU-accelerated Extended Kalman Filter for GNSS positioning";

    // Expose EKFConfig
    py::class_<gnss_gpu::EKFConfig>(m, "EKFConfig")
        .def(py::init([](double sigma_pos, double sigma_vel,
                         double sigma_clk, double sigma_drift, double sigma_pr) {
            gnss_gpu::EKFConfig cfg;
            cfg.sigma_pos = sigma_pos;
            cfg.sigma_vel = sigma_vel;
            cfg.sigma_clk = sigma_clk;
            cfg.sigma_drift = sigma_drift;
            cfg.sigma_pr = sigma_pr;
            return cfg;
        }), py::arg("sigma_pos") = 1.0, py::arg("sigma_vel") = 0.1,
            py::arg("sigma_clk") = 100.0, py::arg("sigma_drift") = 10.0,
            py::arg("sigma_pr") = 5.0)
        .def_readwrite("sigma_pos", &gnss_gpu::EKFConfig::sigma_pos)
        .def_readwrite("sigma_vel", &gnss_gpu::EKFConfig::sigma_vel)
        .def_readwrite("sigma_clk", &gnss_gpu::EKFConfig::sigma_clk)
        .def_readwrite("sigma_drift", &gnss_gpu::EKFConfig::sigma_drift)
        .def_readwrite("sigma_pr", &gnss_gpu::EKFConfig::sigma_pr);

    // Expose EKFState as opaque handle with numpy accessors
    py::class_<gnss_gpu::EKFState>(m, "EKFState")
        .def(py::init<>())
        .def("get_state", [](const gnss_gpu::EKFState& s) {
            auto arr = py::array_t<double>(std::vector<ssize_t>{8});
            auto buf = arr.request();
            memcpy(buf.ptr, s.x, 8 * sizeof(double));
            return arr;
        })
        .def("get_covariance", [](const gnss_gpu::EKFState& s) {
            auto arr = py::array_t<double>(std::vector<ssize_t>{8, 8});
            auto buf = arr.request();
            memcpy(buf.ptr, s.P, 64 * sizeof(double));
            return arr;
        });

    // ekf_initialize
    m.def("ekf_initialize", [](py::array_t<double> initial_pos,
                                double initial_cb, double sigma_pos, double sigma_cb) {
        auto bp = initial_pos.request();
        if (bp.size < 3) throw std::runtime_error("initial_pos must have at least 3 elements");
        gnss_gpu::EKFState state;
        gnss_gpu::ekf_initialize(&state, static_cast<double*>(bp.ptr),
                                  initial_cb, sigma_pos, sigma_cb);
        return state;
    }, "Initialize EKF state from position",
       py::arg("initial_pos"), py::arg("initial_cb") = 0.0,
       py::arg("sigma_pos") = 100.0, py::arg("sigma_cb") = 1000.0);

    // ekf_predict — operate on numpy arrays to avoid struct copy issues
    m.def("ekf_predict", [](py::array_t<double> state_x, py::array_t<double> state_P,
                             double dt, const gnss_gpu::EKFConfig& config) {
        auto bx = state_x.request();
        auto bP = state_P.request();
        if (bx.size < 8) throw std::runtime_error("state_x must have 8 elements");
        if (bP.size < 64) throw std::runtime_error("state_P must have 64 elements");
        gnss_gpu::EKFState state;
        memcpy(state.x, bx.ptr, 8 * sizeof(double));
        memcpy(state.P, bP.ptr, 64 * sizeof(double));
        gnss_gpu::ekf_predict(&state, dt, config);
        memcpy(state_x.mutable_data(), state.x, 8 * sizeof(double));
        memcpy(state_P.mutable_data(), state.P, 64 * sizeof(double));
    }, "EKF predict step (modifies state_x and state_P in-place)",
       py::arg("state_x"), py::arg("state_P"), py::arg("dt"), py::arg("config"));

    // ekf_update — operate on numpy arrays to avoid struct copy issues
    m.def("ekf_update", [](py::array_t<double> state_x, py::array_t<double> state_P,
                            py::array_t<double> sat_ecef,
                            py::array_t<double> pseudoranges,
                            py::array_t<double> weights) {
        auto bx = state_x.request();
        auto bP = state_P.request();
        auto bs = sat_ecef.request();
        auto bp = pseudoranges.request();
        auto bw = weights.request();
        if (bx.size < 8) throw std::runtime_error("state_x must have 8 elements");
        if (bP.size < 64) throw std::runtime_error("state_P must have 64 elements");
        // sat_ecef: accept (N,3) or (N*3,) flat
        int n_sat = static_cast<int>(bp.size);
        gnss_gpu::EKFState state;
        memcpy(state.x, bx.ptr, 8 * sizeof(double));
        memcpy(state.P, bP.ptr, 64 * sizeof(double));
        gnss_gpu::ekf_update(&state, static_cast<double*>(bs.ptr),
                              static_cast<double*>(bp.ptr),
                              static_cast<double*>(bw.ptr), n_sat);
        memcpy(state_x.mutable_data(), state.x, 8 * sizeof(double));
        memcpy(state_P.mutable_data(), state.P, 64 * sizeof(double));
    }, "EKF update step with pseudorange measurements (modifies state_x and state_P in-place)",
       py::arg("state_x"), py::arg("state_P"), py::arg("sat_ecef"),
       py::arg("pseudoranges"), py::arg("weights"));

    // ekf_batch
    m.def("ekf_batch", [](py::array_t<double> states_x, py::array_t<double> states_P,
                           py::array_t<double> sat_ecef, py::array_t<double> pseudoranges,
                           py::array_t<double> weights, double dt,
                           const gnss_gpu::EKFConfig& config) {
        auto bx = states_x.request();
        auto bP = states_P.request();
        {
            auto bs = sat_ecef.request();
            // sat_ecef: accept (N,3) or (N*3,) flat
        }
        int n_instances = bx.shape[0];
        int n_sat = pseudoranges.request().shape[0];

        // Pack into EKFState array
        std::vector<gnss_gpu::EKFState> states(n_instances);
        double* xptr = static_cast<double*>(bx.ptr);
        double* pptr = static_cast<double*>(bP.ptr);
        for (int i = 0; i < n_instances; i++) {
            memcpy(states[i].x, xptr + i * 8, 8 * sizeof(double));
            memcpy(states[i].P, pptr + i * 64, 64 * sizeof(double));
        }

        gnss_gpu::ekf_batch(states.data(),
                              static_cast<double*>(sat_ecef.request().ptr),
                              static_cast<double*>(pseudoranges.request().ptr),
                              static_cast<double*>(weights.request().ptr),
                              dt, config, n_instances, n_sat);

        // Unpack results
        auto out_x = py::array_t<double>({n_instances, 8});
        auto out_P = py::array_t<double>({n_instances, 8, 8});
        double* ox = out_x.mutable_data();
        double* oP = out_P.mutable_data();
        for (int i = 0; i < n_instances; i++) {
            memcpy(ox + i * 8, states[i].x, 8 * sizeof(double));
            memcpy(oP + i * 64, states[i].P, 64 * sizeof(double));
        }
        return py::make_tuple(out_x, out_P);
    }, "Batch EKF predict+update on GPU",
       py::arg("states_x"), py::arg("states_P"),
       py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights"),
       py::arg("dt"), py::arg("config"));
}
