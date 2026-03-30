#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gnss_gpu/tracking.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_tracking, m) {
  m.doc() = "GPU-accelerated GNSS tracking loops";

  // --- TrackingConfig ---
  py::class_<gnss_gpu::TrackingConfig>(m, "TrackingConfig")
    .def(py::init<>())
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
  m.def("batch_correlate", [](py::array_t<float> signal,
                               py::list channels_list,
                               int n_channels, int n_samples,
                               const gnss_gpu::TrackingConfig& config) {
    auto sig_buf = signal.request();
    // Convert Python channel list to C++ array
    std::vector<gnss_gpu::ChannelState> channels(n_channels);
    for (int i = 0; i < n_channels; i++) {
      auto ch = channels_list[i].cast<gnss_gpu::ChannelState*>();
      channels[i] = *ch;
    }

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
                                      py::array_t<double> correlations,
                                      int n_channels,
                                      const gnss_gpu::TrackingConfig& config) {
    std::vector<gnss_gpu::ChannelState> channels(n_channels);
    for (int i = 0; i < n_channels; i++) {
      channels[i] = *channels_list[i].cast<gnss_gpu::ChannelState*>();
    }

    gnss_gpu::scalar_tracking_update(
      channels.data(),
      static_cast<double*>(correlations.request().ptr),
      n_channels, config);

    // Write back updated states
    for (int i = 0; i < n_channels; i++) {
      auto ch = channels_list[i].cast<gnss_gpu::ChannelState*>();
      *ch = channels[i];
    }
  }, "Scalar tracking loop update",
     py::arg("channels"), py::arg("correlations"),
     py::arg("n_channels"), py::arg("config"));

  // --- vector_tracking_update ---
  m.def("vector_tracking_update", [](py::list channels_list,
                                      py::array_t<double> correlations,
                                      py::array_t<double> sat_ecef,
                                      py::array_t<double> sat_vel,
                                      py::array_t<double> nav_state,
                                      py::array_t<double> nav_cov,
                                      int n_channels,
                                      const gnss_gpu::TrackingConfig& config,
                                      double dt) {
    std::vector<gnss_gpu::ChannelState> channels(n_channels);
    for (int i = 0; i < n_channels; i++) {
      channels[i] = *channels_list[i].cast<gnss_gpu::ChannelState*>();
    }

    gnss_gpu::vector_tracking_update(
      channels.data(),
      static_cast<double*>(correlations.request().ptr),
      static_cast<double*>(sat_ecef.request().ptr),
      static_cast<double*>(sat_vel.request().ptr),
      static_cast<double*>(nav_state.request().ptr),
      static_cast<double*>(nav_cov.request().ptr),
      n_channels, config, dt);

    for (int i = 0; i < n_channels; i++) {
      auto ch = channels_list[i].cast<gnss_gpu::ChannelState*>();
      *ch = channels[i];
    }
  }, "Vector tracking loop update (EKF-based)",
     py::arg("channels"), py::arg("correlations"),
     py::arg("sat_ecef"), py::arg("sat_vel"),
     py::arg("nav_state"), py::arg("nav_cov"),
     py::arg("n_channels"), py::arg("config"), py::arg("dt"));

  // --- cn0_nwpr ---
  m.def("cn0_nwpr", [](py::array_t<double> correlations_hist,
                        int n_channels, int n_hist, double T) {
    auto cn0 = py::array_t<double>(std::vector<ssize_t>{n_channels});
    gnss_gpu::cn0_nwpr(
      static_cast<double*>(correlations_hist.request().ptr),
      static_cast<double*>(cn0.request().ptr),
      n_channels, n_hist, T);
    return cn0;
  }, "Estimate CN0 using Narrow-Wideband Power Ratio",
     py::arg("correlations_hist"), py::arg("n_channels"),
     py::arg("n_hist"), py::arg("T"));
}
