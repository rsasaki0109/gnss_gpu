#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gnss_gpu/acquisition.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

void validate_prn(int prn, const char* name) {
    if (prn < 1 || prn > 32) {
        throw std::runtime_error(std::string(name) + " must be in [1, 32]");
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

void ensure_valid_prn_values(const py::buffer_info& buf, const char* message) {
    const auto* ptr = static_cast<const int*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (ptr[i] < 1 || ptr[i] > 32) {
            throw std::runtime_error(message);
        }
    }
}

void validate_acquisition_options(double sampling_freq, double intermediate_freq,
                                  double doppler_range, double doppler_step,
                                  float threshold) {
    if (!std::isfinite(sampling_freq) || sampling_freq <= 0.0) {
        throw std::runtime_error("sampling_freq must be positive and finite");
    }
    if (!std::isfinite(intermediate_freq)) {
        throw std::runtime_error("intermediate_freq must be finite");
    }
    if (!std::isfinite(doppler_range) || doppler_range < 0.0) {
        throw std::runtime_error("doppler_range must be finite and non-negative");
    }
    if (!std::isfinite(doppler_step) || doppler_step <= 0.0) {
        throw std::runtime_error("doppler_step must be positive and finite");
    }
    if (!std::isfinite(threshold) || threshold < 0.0f) {
        throw std::runtime_error("threshold must be finite and non-negative");
    }
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_acq, m) {
    m.doc() = "GPU-accelerated GNSS signal acquisition";

    m.def("generate_ca_code", [](int prn) {
        validate_prn(prn, "prn");
        int code[1023];
        gnss_gpu::generate_ca_code(prn, code);
        py::list result;
        for (int i = 0; i < 1023; i++) {
            result.append(code[i]);
        }
        return result;
    }, "Generate GPS C/A code (1023 chips, +1/-1)",
       py::arg("prn"));

    m.def("acquire_parallel", [](FloatArray signal,
                                  double sampling_freq, double intermediate_freq,
                                  IntArray prn_list,
                                  double doppler_range, double doppler_step,
                                  float threshold) {
        auto sig_buf = signal.request();
        auto prn_buf = prn_list.request();
        if (sig_buf.ndim != 1)
            throw std::runtime_error("signal must be a 1D array");
        if (prn_buf.ndim != 1)
            throw std::runtime_error("prn_list must be a 1D array");
        if (sig_buf.size < 1)
            throw std::runtime_error("signal must contain at least one sample");
        if (prn_buf.size < 1)
            throw std::runtime_error("prn_list must contain at least one PRN");
        validate_acquisition_options(
            sampling_freq, intermediate_freq, doppler_range, doppler_step, threshold);
        ensure_finite_float_values(sig_buf, "signal must be finite");
        ensure_valid_prn_values(prn_buf, "prn_list values must be in [1, 32]");

        int n_samples = static_cast<int>(sig_buf.size);
        int n_prn = static_cast<int>(prn_buf.size);

        std::vector<gnss_gpu::AcquisitionResult> results(n_prn);

        gnss_gpu::acquire_parallel(
            static_cast<float*>(sig_buf.ptr), n_samples,
            sampling_freq, intermediate_freq,
            static_cast<int*>(prn_buf.ptr), n_prn,
            doppler_range, doppler_step, threshold,
            results.data());

        py::list out;
        for (int i = 0; i < n_prn; i++) {
            py::dict d;
            d["prn"] = results[i].prn;
            d["acquired"] = results[i].acquired;
            d["code_phase"] = results[i].code_phase;
            d["doppler_hz"] = results[i].doppler_hz;
            d["snr"] = results[i].snr;
            out.append(d);
        }
        return out;
    }, "Run parallel acquisition over PRN list",
       py::arg("signal"), py::arg("sampling_freq"),
       py::arg("intermediate_freq"), py::arg("prn_list"),
       py::arg("doppler_range"), py::arg("doppler_step"),
       py::arg("threshold"));
}
