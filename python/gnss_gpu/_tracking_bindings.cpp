#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gnss_gpu/tracking.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void validate_config(const gnss_gpu::TrackingConfig& config) {
  if (!std::isfinite(config.sampling_freq) || config.sampling_freq <= 0.0) {
    throw std::runtime_error("sampling_freq must be positive and finite");
  }
  if (!std::isfinite(config.intermediate_freq)) {
    throw std::runtime_error("intermediate_freq must be finite");
  }
  if (!std::isfinite(config.integration_time) || config.integration_time <= 0.0) {
    throw std::runtime_error("integration_time must be positive and finite");
  }
  if (!std::isfinite(config.dll_bandwidth) || config.dll_bandwidth < 0.0) {
    throw std::runtime_error("dll_bandwidth must be finite and non-negative");
  }
  if (!std::isfinite(config.pll_bandwidth) || config.pll_bandwidth < 0.0) {
    throw std::runtime_error("pll_bandwidth must be finite and non-negative");
  }
  if (!std::isfinite(config.correlator_spacing) || config.correlator_spacing <= 0.0) {
    throw std::runtime_error("correlator_spacing must be positive and finite");
  }
}

void ensure_finite_float_values(const py::buffer_info& buf, const char* message) {
  const auto* ptr = static_cast<const float*>(buf.ptr);
  for (py::ssize_t i = 0; i < buf.size; ++i) {
    if (!std::isfinite(ptr[i])) {
      throw std::runtime_error(message);
    }
  }
}

void ensure_finite_double_values(const py::buffer_info& buf, const char* message) {
  const auto* ptr = static_cast<const double*>(buf.ptr);
  for (py::ssize_t i = 0; i < buf.size; ++i) {
    if (!std::isfinite(ptr[i])) {
      throw std::runtime_error(message);
    }
  }
}

void validate_n_channels(int n_channels, const char* function_name) {
  if (n_channels < 1) {
    throw std::runtime_error(std::string(function_name) + ": n_channels must be >= 1");
  }
}

void validate_channel_state(const gnss_gpu::ChannelState& ch, const char* function_name) {
  if (ch.prn < 1 || ch.prn > 32) {
    throw std::runtime_error(std::string(function_name) + ": channel PRN must be in [1, 32]");
  }
  if (!std::isfinite(ch.code_phase)) {
    throw std::runtime_error(std::string(function_name) + ": channel code_phase must be finite");
  }
  if (!std::isfinite(ch.code_freq) || ch.code_freq <= 0.0) {
    throw std::runtime_error(std::string(function_name) + ": channel code_freq must be positive and finite");
  }
  if (!std::isfinite(ch.carrier_phase)) {
    throw std::runtime_error(std::string(function_name) + ": channel carrier_phase must be finite");
  }
  if (!std::isfinite(ch.carrier_freq)) {
    throw std::runtime_error(std::string(function_name) + ": channel carrier_freq must be finite");
  }
  if (!std::isfinite(ch.cn0)) {
    throw std::runtime_error(std::string(function_name) + ": channel cn0 must be finite");
  }
  if (!std::isfinite(ch.dll_integrator) || !std::isfinite(ch.pll_integrator)) {
    throw std::runtime_error(std::string(function_name) + ": channel integrators must be finite");
  }
}

std::vector<gnss_gpu::ChannelState> validate_channels(py::list channels_list,
                                                       int n_channels,
                                                       const char* function_name) {
  validate_n_channels(n_channels, function_name);
  if (channels_list.size() != static_cast<size_t>(n_channels)) {
    throw std::runtime_error(std::string(function_name) + ": channels length must match n_channels");
  }

  std::vector<gnss_gpu::ChannelState> channels(n_channels);
  for (int i = 0; i < n_channels; i++) {
    auto ch = channels_list[i].cast<gnss_gpu::ChannelState*>();
    channels[i] = *ch;
    validate_channel_state(channels[i], function_name);
  }
  return channels;
}

int validate_signal(const py::buffer_info& buf, int n_samples,
                    const char* function_name) {
  if (buf.ndim != 1) {
    throw std::runtime_error(std::string(function_name) + ": signal must be a 1D array");
  }
  if (n_samples < 1) {
    throw std::runtime_error(std::string(function_name) + ": n_samples must be >= 1");
  }
  if (buf.size < n_samples) {
    throw std::runtime_error(std::string(function_name) + ": signal length must be >= n_samples");
  }
  ensure_finite_float_values(buf, "signal must be finite");
  return n_samples;
}

void validate_double_size(const py::buffer_info& buf, py::ssize_t expected_size,
                          const char* message) {
  if (buf.size != expected_size) {
    throw std::runtime_error(message);
  }
}

void write_channels_back(py::list channels_list,
                         const std::vector<gnss_gpu::ChannelState>& channels) {
  for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(channels.size()); i++) {
    auto ch = channels_list[i].cast<gnss_gpu::ChannelState*>();
    *ch = channels[static_cast<size_t>(i)];
  }
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_tracking, m) {
  m.doc() = "GPU-accelerated GNSS tracking loops";

  // --- TrackingConfig ---
  py::class_<gnss_gpu::TrackingConfig>(m, "TrackingConfig")
    .def(py::init([]() {
      gnss_gpu::TrackingConfig cfg;
      cfg.sampling_freq = 4.092e6;
      cfg.intermediate_freq = 4.092e6;
      cfg.integration_time = 1e-3;
      cfg.dll_bandwidth = 2.0;
      cfg.pll_bandwidth = 15.0;
      cfg.correlator_spacing = 0.5;
      return cfg;
    }))
    .def(py::init([](double sampling_freq, double intermediate_freq,
                     double integration_time, double dll_bandwidth,
                     double pll_bandwidth, double correlator_spacing) {
      gnss_gpu::TrackingConfig cfg;
      cfg.sampling_freq = sampling_freq;
      cfg.intermediate_freq = intermediate_freq;
      cfg.integration_time = integration_time;
      cfg.dll_bandwidth = dll_bandwidth;
      cfg.pll_bandwidth = pll_bandwidth;
      cfg.correlator_spacing = correlator_spacing;
      validate_config(cfg);
      return cfg;
    }), py::arg("sampling_freq") = 4.092e6,
        py::arg("intermediate_freq") = 4.092e6,
        py::arg("integration_time") = 1e-3,
        py::arg("dll_bandwidth") = 2.0,
        py::arg("pll_bandwidth") = 15.0,
        py::arg("correlator_spacing") = 0.5)
    .def_readwrite("sampling_freq", &gnss_gpu::TrackingConfig::sampling_freq)
    .def_readwrite("intermediate_freq", &gnss_gpu::TrackingConfig::intermediate_freq)
    .def_readwrite("integration_time", &gnss_gpu::TrackingConfig::integration_time)
    .def_readwrite("dll_bandwidth", &gnss_gpu::TrackingConfig::dll_bandwidth)
    .def_readwrite("pll_bandwidth", &gnss_gpu::TrackingConfig::pll_bandwidth)
    .def_readwrite("correlator_spacing", &gnss_gpu::TrackingConfig::correlator_spacing);

  // --- ChannelState ---
  py::class_<gnss_gpu::ChannelState>(m, "ChannelState")
    .def(py::init<>())
    .def_readwrite("code_phase", &gnss_gpu::ChannelState::code_phase)
    .def_readwrite("code_freq", &gnss_gpu::ChannelState::code_freq)
    .def_readwrite("carrier_phase", &gnss_gpu::ChannelState::carrier_phase)
    .def_readwrite("carrier_freq", &gnss_gpu::ChannelState::carrier_freq)
    .def_readwrite("cn0", &gnss_gpu::ChannelState::cn0)
    .def_readwrite("dll_integrator", &gnss_gpu::ChannelState::dll_integrator)
    .def_readwrite("pll_integrator", &gnss_gpu::ChannelState::pll_integrator)
    .def_readwrite("prn", &gnss_gpu::ChannelState::prn)
    .def_readwrite("locked", &gnss_gpu::ChannelState::locked);

  // --- batch_correlate ---
  m.def("batch_correlate", [](FloatArray signal,
                               py::list channels_list,
                               int n_channels, int n_samples,
                               const gnss_gpu::TrackingConfig& config) {
    auto sig_buf = signal.request();
    validate_config(config);
    validate_signal(sig_buf, n_samples, "batch_correlate");
    std::vector<gnss_gpu::ChannelState> channels =
      validate_channels(channels_list, n_channels, "batch_correlate");

    auto correlations = py::array_t<double>({n_channels * 6});
    gnss_gpu::batch_correlate(
      static_cast<float*>(sig_buf.ptr),
      channels.data(),
      static_cast<double*>(correlations.request().ptr),
      n_channels, n_samples, config);

    return correlations;
  }, "Batch correlate signal with all channels",
     py::arg("signal"), py::arg("channels"),
     py::arg("n_channels"), py::arg("n_samples"),
     py::arg("config"));

  // --- scalar_tracking_update ---
  m.def("scalar_tracking_update", [](py::list channels_list,
                                      DoubleArray correlations,
                                      int n_channels,
                                      const gnss_gpu::TrackingConfig& config) {
    validate_config(config);
    std::vector<gnss_gpu::ChannelState> channels =
      validate_channels(channels_list, n_channels, "scalar_tracking_update");
    auto corr_buf = correlations.request();
    validate_double_size(
      corr_buf, static_cast<py::ssize_t>(n_channels) * 6,
      "scalar_tracking_update: correlations must have n_channels * 6 values");
    ensure_finite_double_values(corr_buf, "correlations must be finite");

    gnss_gpu::scalar_tracking_update(
      channels.data(),
      static_cast<double*>(corr_buf.ptr),
      n_channels, config);

    // Write back updated states
    write_channels_back(channels_list, channels);
  }, "Scalar tracking loop update",
     py::arg("channels"), py::arg("correlations"),
     py::arg("n_channels"), py::arg("config"));

  // --- vector_tracking_update ---
  m.def("vector_tracking_update", [](py::list channels_list,
                                      DoubleArray correlations,
                                      DoubleArray sat_ecef,
                                      DoubleArray sat_vel,
                                      DoubleArray nav_state,
                                      DoubleArray nav_cov,
                                      int n_channels,
                                      const gnss_gpu::TrackingConfig& config,
                                      double dt) {
    validate_config(config);
    if (!std::isfinite(dt) || dt <= 0.0) {
      throw std::runtime_error("vector_tracking_update: dt must be positive and finite");
    }
    if (n_channels > 32) {
      throw std::runtime_error("vector_tracking_update supports at most 32 channels");
    }
    std::vector<gnss_gpu::ChannelState> channels =
      validate_channels(channels_list, n_channels, "vector_tracking_update");
    auto corr_buf = correlations.request();
    auto sat_buf = sat_ecef.request();
    auto vel_buf = sat_vel.request();
    auto state_buf = nav_state.request();
    auto cov_buf = nav_cov.request();
    validate_double_size(
      corr_buf, static_cast<py::ssize_t>(n_channels) * 6,
      "vector_tracking_update: correlations must have n_channels * 6 values");
    validate_double_size(
      sat_buf, static_cast<py::ssize_t>(n_channels) * 3,
      "vector_tracking_update: sat_ecef must have n_channels * 3 values");
    validate_double_size(
      vel_buf, static_cast<py::ssize_t>(n_channels) * 3,
      "vector_tracking_update: sat_vel must have n_channels * 3 values");
    if (state_buf.size < 8) {
      throw std::runtime_error("vector_tracking_update: nav_state must have at least 8 elements");
    }
    if (cov_buf.size < 64) {
      throw std::runtime_error("vector_tracking_update: nav_cov must have at least 64 elements");
    }
    ensure_finite_double_values(corr_buf, "correlations must be finite");
    ensure_finite_double_values(sat_buf, "sat_ecef must be finite");
    ensure_finite_double_values(vel_buf, "sat_vel must be finite");
    ensure_finite_double_values(state_buf, "nav_state must be finite");
    ensure_finite_double_values(cov_buf, "nav_cov must be finite");

    gnss_gpu::vector_tracking_update(
      channels.data(),
      static_cast<double*>(corr_buf.ptr),
      static_cast<double*>(sat_buf.ptr),
      static_cast<double*>(vel_buf.ptr),
      static_cast<double*>(state_buf.ptr),
      static_cast<double*>(cov_buf.ptr),
      n_channels, config, dt);

    write_channels_back(channels_list, channels);
  }, "Vector tracking loop update (EKF-based)",
     py::arg("channels"), py::arg("correlations"),
     py::arg("sat_ecef"), py::arg("sat_vel"),
     py::arg("nav_state"), py::arg("nav_cov"),
     py::arg("n_channels"), py::arg("config"), py::arg("dt"));

  // --- cn0_nwpr ---
  m.def("cn0_nwpr", [](DoubleArray correlations_hist,
                        int n_channels, int n_hist, double T) {
    validate_n_channels(n_channels, "cn0_nwpr");
    if (n_hist < 2) {
      throw std::runtime_error("cn0_nwpr: n_hist must be >= 2");
    }
    if (!std::isfinite(T) || T <= 0.0) {
      throw std::runtime_error("cn0_nwpr: T must be positive and finite");
    }
    auto hist_buf = correlations_hist.request();
    validate_double_size(
      hist_buf, static_cast<py::ssize_t>(n_channels) * n_hist * 6,
      "cn0_nwpr: correlations_hist must have n_channels * n_hist * 6 values");
    ensure_finite_double_values(hist_buf, "correlations_hist must be finite");

    auto cn0 = py::array_t<double>(std::vector<py::ssize_t>{n_channels});
    gnss_gpu::cn0_nwpr(
      static_cast<double*>(hist_buf.ptr),
      static_cast<double*>(cn0.request().ptr),
      n_channels, n_hist, T);
    return cn0;
  }, "Estimate CN0 using Narrow-Wideband Power Ratio",
     py::arg("correlations_hist"), py::arg("n_channels"),
     py::arg("n_hist"), py::arg("T"));
}
