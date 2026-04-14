#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstdint>
#include <vector>
#include "gnss_gpu/signal_sim.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_signal_sim, m) {
    m.doc() = "GPU-accelerated GNSS signal simulation";

    m.def("init_ca_codes", &gnss_gpu::init_ca_codes,
          "Initialize C/A codes in GPU constant memory");

    m.def("generate_signal", [](double sampling_freq, double intermediate_freq,
                                 py::list channels_list, int n_samples,
                                 double noise_floor_db,
                                 std::uint64_t noise_seed) {
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
