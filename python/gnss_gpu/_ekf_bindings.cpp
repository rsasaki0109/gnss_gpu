#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/ekf.h"
#include "gnss_gpu/pybind_validation.h"
#include <cstring>

namespace py = pybind11;

namespace {

using gnss_gpu::pybind_validation::DoubleArray;
namespace pv = gnss_gpu::pybind_validation;

void validate_ekf_config(const gnss_gpu::EKFConfig& config) {
    pv::validate_positive_finite(config.sigma_pos, "sigma_pos");
    pv::validate_positive_finite(config.sigma_vel, "sigma_vel");
    pv::validate_positive_finite(config.sigma_clk, "sigma_clk");
    pv::validate_positive_finite(config.sigma_drift, "sigma_drift");
    pv::validate_positive_finite(config.sigma_pr, "sigma_pr");
}

void validate_initial_pos(const py::buffer_info& buf) {
    if (buf.ndim != 1 || buf.size != 3) {
        throw std::runtime_error("initial_pos must have shape (3,)");
    }
    pv::ensure_finite_doubles(buf, "initial_pos must be finite");
}

void validate_state_x(const py::buffer_info& buf) {
    if (buf.ndim != 1 || buf.size != 8) {
        throw std::runtime_error("state_x must have shape (8,)");
    }
    pv::ensure_finite_doubles(buf, "state_x must be finite");
}

void validate_state_P(const py::buffer_info& buf) {
    const bool flat_ok = (buf.ndim == 1 && buf.size == 64);
    const bool matrix_ok =
        (buf.ndim == 2 && buf.shape[0] == 8 && buf.shape[1] == 8);
    if (!flat_ok && !matrix_ok) {
        throw std::runtime_error("state_P must have shape (8, 8) or 64 flat elements");
    }
    pv::ensure_finite_doubles(buf, "state_P must be finite");
}

void validate_sat_ecef(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim == 2) {
        if (buf.shape[0] != n_sat || buf.shape[1] != 3) {
            throw std::runtime_error("sat_ecef shape must match n_sat satellites");
        }
    } else if (buf.ndim == 1) {
        if (buf.size != static_cast<py::ssize_t>(n_sat) * 3) {
            throw std::runtime_error("sat_ecef shape must match n_sat satellites");
        }
    } else {
        throw std::runtime_error("sat_ecef must be 1D or 2D");
    }
    pv::ensure_finite_doubles(buf, "sat_ecef must be finite");
}

void validate_pseudoranges(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim != 1 || buf.size != n_sat) {
        throw std::runtime_error("pseudoranges length must match n_sat");
    }
    pv::ensure_finite_doubles(buf, "pseudoranges must be finite");
}

void validate_weights(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim != 1 || buf.size != n_sat) {
        throw std::runtime_error("weights length must match n_sat");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(ptr[i])) {
            throw std::runtime_error("weights must be finite");
        }
        if (ptr[i] < 0.0) {
            throw std::runtime_error("weights must be non-negative");
        }
    }
}

int validate_batch_states_x(const py::buffer_info& buf) {
    if (buf.ndim != 2 || buf.shape[1] != 8) {
        throw std::runtime_error("states_x must have shape (n_instances, 8)");
    }
    const int n_instances = static_cast<int>(buf.shape[0]);
    if (n_instances < 1) {
        throw std::runtime_error("n_instances must be >= 1");
    }
    pv::ensure_finite_doubles(buf, "states_x must be finite");
    return n_instances;
}

void validate_batch_states_P(const py::buffer_info& buf, int n_instances) {
    if (buf.ndim != 3 || buf.shape[0] != n_instances ||
        buf.shape[1] != 8 || buf.shape[2] != 8) {
        throw std::runtime_error("states_P must have shape (n_instances, 8, 8)");
    }
    pv::ensure_finite_doubles(buf, "states_P must be finite");
}

}  // namespace

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
            validate_ekf_config(cfg);
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
            auto arr = py::array_t<double>(std::vector<py::ssize_t>{8});
            auto buf = arr.request();
            memcpy(buf.ptr, s.x, 8 * sizeof(double));
            return arr;
        })
        .def("get_covariance", [](const gnss_gpu::EKFState& s) {
            auto arr = py::array_t<double>(std::vector<py::ssize_t>{8, 8});
            auto buf = arr.request();
            memcpy(buf.ptr, s.P, 64 * sizeof(double));
            return arr;
        });

    // ekf_initialize
    m.def("ekf_initialize", [](DoubleArray initial_pos,
                                double initial_cb, double sigma_pos, double sigma_cb) {
        auto bp = initial_pos.request();
        validate_initial_pos(bp);
        pv::validate_positive_finite(sigma_pos, "sigma_pos");
        pv::validate_positive_finite(sigma_cb, "sigma_cb");
        gnss_gpu::EKFState state;
        gnss_gpu::ekf_initialize(&state, static_cast<double*>(bp.ptr),
                                  initial_cb, sigma_pos, sigma_cb);
        return state;
    }, "Initialize EKF state from position",
       py::arg("initial_pos"), py::arg("initial_cb") = 0.0,
       py::arg("sigma_pos") = 100.0, py::arg("sigma_cb") = 1000.0);

    // ekf_predict — operate on numpy arrays to avoid struct copy issues
    m.def("ekf_predict", [](DoubleArray state_x, DoubleArray state_P,
                             double dt, const gnss_gpu::EKFConfig& config) {
        auto bx = state_x.request();
        auto bP = state_P.request();
        validate_state_x(bx);
        validate_state_P(bP);
        pv::validate_dt_positive(dt);
        validate_ekf_config(config);
        gnss_gpu::EKFState state;
        memcpy(state.x, bx.ptr, 8 * sizeof(double));
        memcpy(state.P, bP.ptr, 64 * sizeof(double));
        gnss_gpu::ekf_predict(&state, dt, config);
        memcpy(state_x.mutable_data(), state.x, 8 * sizeof(double));
        memcpy(state_P.mutable_data(), state.P, 64 * sizeof(double));
    }, "EKF predict step (modifies state_x and state_P in-place)",
       py::arg("state_x"), py::arg("state_P"), py::arg("dt"), py::arg("config"));

    // ekf_update — operate on numpy arrays to avoid struct copy issues
    m.def("ekf_update", [](DoubleArray state_x, DoubleArray state_P,
                            DoubleArray sat_ecef,
                            DoubleArray pseudoranges,
                            DoubleArray weights) {
        auto bx = state_x.request();
        auto bP = state_P.request();
        auto bp = pseudoranges.request();
        validate_state_x(bx);
        validate_state_P(bP);
        const int n_sat = static_cast<int>(bp.size);
        pv::validate_n_sat(n_sat);
        validate_sat_ecef(sat_ecef.request(), n_sat);
        validate_pseudoranges(bp, n_sat);
        validate_weights(weights.request(), n_sat);
        gnss_gpu::EKFState state;
        memcpy(state.x, bx.ptr, 8 * sizeof(double));
        memcpy(state.P, bP.ptr, 64 * sizeof(double));
        gnss_gpu::ekf_update(&state, static_cast<double*>(sat_ecef.request().ptr),
                              static_cast<double*>(bp.ptr),
                              static_cast<double*>(weights.request().ptr), n_sat);
        memcpy(state_x.mutable_data(), state.x, 8 * sizeof(double));
        memcpy(state_P.mutable_data(), state.P, 64 * sizeof(double));
    }, "EKF update step with pseudorange measurements (modifies state_x and state_P in-place)",
       py::arg("state_x"), py::arg("state_P"), py::arg("sat_ecef"),
       py::arg("pseudoranges"), py::arg("weights"));

    // ekf_batch
    m.def("ekf_batch", [](DoubleArray states_x, DoubleArray states_P,
                           DoubleArray sat_ecef, DoubleArray pseudoranges,
                           DoubleArray weights, double dt,
                           const gnss_gpu::EKFConfig& config) {
        auto bx = states_x.request();
        auto bP = states_P.request();
        const int n_instances = validate_batch_states_x(bx);
        validate_batch_states_P(bP, n_instances);
        pv::validate_dt_positive(dt);
        validate_ekf_config(config);

        auto bpr = pseudoranges.request();
        if (bpr.ndim != 1) {
            throw std::runtime_error("pseudoranges must be a 1D array");
        }
        const int n_sat = static_cast<int>(bpr.size);
        pv::validate_n_sat(n_sat);
        validate_sat_ecef(sat_ecef.request(), n_sat);
        validate_pseudoranges(bpr, n_sat);
        validate_weights(weights.request(), n_sat);

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
                              static_cast<double*>(bpr.ptr),
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
