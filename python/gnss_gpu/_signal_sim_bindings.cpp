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
        std::vector<gnss_gpu::SatChannel> channels;
        channels.reserve(channels_list.size());
        for (auto item : channels_list) {
            py::dict d = item.cast<py::dict>();
            gnss_gpu::SatChannel ch{};
            ch.prn = d["prn"].cast<int>();
            ch.code_phase = d["code_phase"].cast<double>();
            ch.carrier_phase = d["carrier_phase"].cast<double>();
            ch.doppler_hz = d["doppler_hz"].cast<double>();
            ch.amplitude = d["amplitude"].cast<float>();
            ch.nav_bit = d["nav_bit"].cast<int>();
            channels.push_back(ch);
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
                                 int n_samples) {
        gnss_gpu::SatChannel ch{};
        ch.prn = prn;
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
       py::arg("n_samples"));
}
