#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include "gnss_gpu/pybind_validation.h"
#include "gnss_gpu/signal_sim.h"

namespace py = pybind11;

namespace {

namespace pv = gnss_gpu::pybind_validation;

void validate_channel_dict(const py::dict& d, size_t index) {
  static const char* required[] = {
      "prn", "code_phase", "carrier_phase", "doppler_hz", "amplitude", "nav_bit",
  };
  for (const char* key : required) {
    if (!d.contains(key)) {
      throw std::runtime_error(
          std::string("channels[") + std::to_string(index) +
          "] missing required key '" + key + "'");
    }
  }

  const int prn = d["prn"].cast<int>();
  if (prn < 1) {
    throw std::runtime_error(
        std::string("channels[") + std::to_string(index) + "].prn must be >= 1");
  }

  pv::validate_finite(d["code_phase"].cast<double>(),
                      (std::string("channels[") + std::to_string(index) +
                       "].code_phase").c_str());
  pv::validate_finite(d["carrier_phase"].cast<double>(),
                      (std::string("channels[") + std::to_string(index) +
                       "].carrier_phase").c_str());
  pv::validate_finite(d["doppler_hz"].cast<double>(),
                      (std::string("channels[") + std::to_string(index) +
                       "].doppler_hz").c_str());
  pv::validate_finite(d["amplitude"].cast<double>(),
                      (std::string("channels[") + std::to_string(index) +
                       "].amplitude").c_str());

  if (d.contains("nav_bit_rate")) {
    pv::validate_positive_finite(
        d["nav_bit_rate"].cast<double>(),
        (std::string("channels[") + std::to_string(index) + "].nav_bit_rate").c_str());
  }
  if (d.contains("system")) {
    const int system = d["system"].cast<int>();
    if (system < 0) {
      throw std::runtime_error(
          std::string("channels[") + std::to_string(index) +
          "].system must be non-negative");
    }
  }
}

void validate_channels_list(const py::list& channels_list) {
  for (size_t i = 0; i < channels_list.size(); ++i) {
    if (!py::isinstance<py::dict>(channels_list[i])) {
      throw std::runtime_error(
          std::string("channels[") + std::to_string(i) + "] must be a dict");
    }
    validate_channel_dict(channels_list[i].cast<py::dict>(), i);
  }
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_signal_sim, m) {
    m.doc() = "GPU-accelerated GNSS signal simulation";

    m.def("init_ca_codes", &gnss_gpu::init_ca_codes,
          "Initialize C/A codes in GPU constant memory");

    m.def("generate_signal", [](double sampling_freq, double intermediate_freq,
                                 py::list channels_list, int n_samples,
                                 double noise_floor_db,
                                 std::uint64_t noise_seed) {
        pv::validate_positive_finite(sampling_freq, "sampling_freq");
        pv::validate_finite(intermediate_freq, "intermediate_freq");
        pv::validate_finite(noise_floor_db, "noise_floor_db");
        if (n_samples < 1) {
            throw std::runtime_error("n_samples must be >= 1");
        }
        validate_channels_list(channels_list);

        size_t n_ch = channels_list.size();
        std::vector<gnss_gpu::SatChannel> channels(n_ch);
        // Keep nav_bits numpy arrays alive until simulate_epoch returns
        std::vector<py::array_t<int>> nav_bits_refs(n_ch);

        for (size_t i = 0; i < n_ch; ++i) {
            py::dict d = channels_list[i].cast<py::dict>();
            auto& ch = channels[i];
            ch.prn = d["prn"].cast<int>();
            if (d.contains("system")) ch.system = d["system"].cast<int>();
            ch.code_phase = d["code_phase"].cast<double>();
            ch.carrier_phase = d["carrier_phase"].cast<double>();
            ch.doppler_hz = d["doppler_hz"].cast<double>();
            ch.amplitude = d["amplitude"].cast<float>();
            ch.nav_bit = d["nav_bit"].cast<int>();
            if (d.contains("nav_bit_rate")) ch.nav_bit_rate = d["nav_bit_rate"].cast<double>();
            if (d.contains("nav_bits") && !d["nav_bits"].is_none()) {
                nav_bits_refs[i] = d["nav_bits"].cast<py::array_t<int>>();
                ch.nav_bits = static_cast<const int*>(nav_bits_refs[i].data());
                ch.nav_bit_count = (int)nav_bits_refs[i].size();
            }
        }

        gnss_gpu::SignalSimConfig config{};
        config.sampling_freq = sampling_freq;
        config.intermediate_freq = intermediate_freq;
        config.noise_floor_db = noise_floor_db;
        config.noise_seed = noise_seed;

        py::array_t<float> out({2 * n_samples});
        gnss_gpu::simulate_epoch(config, channels.data(),
                                  (int)channels.size(),
                                  static_cast<float*>(out.mutable_data()),
                                  n_samples);
        return out;
    }, "Generate composite GNSS IQ signal",
       py::arg("sampling_freq"), py::arg("intermediate_freq"),
       py::arg("channels"), py::arg("n_samples"),
       py::arg("noise_floor_db"),
       py::arg("noise_seed") = 0);

    m.def("generate_single", [](int prn, double code_phase, double carrier_phase,
                                 double doppler_hz, float amplitude, int nav_bit,
                                 double sampling_freq, double intermediate_freq,
                                 int n_samples, int system) {
        if (prn < 1) {
            throw std::runtime_error("prn must be >= 1");
        }
        pv::validate_finite(code_phase, "code_phase");
        pv::validate_finite(carrier_phase, "carrier_phase");
        pv::validate_finite(doppler_hz, "doppler_hz");
        pv::validate_finite(amplitude, "amplitude");
        pv::validate_positive_finite(sampling_freq, "sampling_freq");
        pv::validate_finite(intermediate_freq, "intermediate_freq");
        if (n_samples < 1) {
            throw std::runtime_error("n_samples must be >= 1");
        }
        if (system < 0) {
            throw std::runtime_error("system must be non-negative");
        }

        gnss_gpu::SatChannel ch{};
        ch.prn = prn;
        ch.system = system;
        ch.code_phase = code_phase;
        ch.carrier_phase = carrier_phase;
        ch.doppler_hz = doppler_hz;
        ch.amplitude = amplitude;
        ch.nav_bit = nav_bit;

        py::array_t<float> out({2 * n_samples});
        gnss_gpu::generate_single_channel(ch,
                                           static_cast<float*>(out.mutable_data()),
                                           n_samples, sampling_freq, intermediate_freq);
        return out;
    }, "Generate single-channel GNSS IQ signal",
       py::arg("prn"), py::arg("code_phase"), py::arg("carrier_phase"),
       py::arg("doppler_hz"), py::arg("amplitude"), py::arg("nav_bit"),
       py::arg("sampling_freq"), py::arg("intermediate_freq"),
       py::arg("n_samples"), py::arg("system") = 0);
}
