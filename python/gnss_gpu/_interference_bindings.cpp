#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gnss_gpu/interference.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

void ensure_finite_values(const py::buffer_info& buf, const char* message) {
    const auto* ptr = static_cast<const float*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(ptr[i])) {
            throw std::runtime_error(message);
        }
    }
}

void validate_fft_size(int fft_size) {
    if (fft_size < 2) {
        throw std::runtime_error("fft_size must be >= 2");
    }
}

void validate_sampling_freq(double sampling_freq) {
    if (!std::isfinite(sampling_freq) || sampling_freq <= 0.0) {
        throw std::runtime_error("sampling_freq must be positive and finite");
    }
}

void validate_threshold(float threshold_db) {
    if (!std::isfinite(threshold_db)) {
        throw std::runtime_error("threshold_db must be finite");
    }
}

void validate_stft_options(int fft_size, int hop_size, double sampling_freq) {
    validate_fft_size(fft_size);
    if (hop_size < 1) {
        throw std::runtime_error("hop_size must be >= 1");
    }
    validate_sampling_freq(sampling_freq);
}

int validate_signal_1d(const py::buffer_info& buf, int fft_size,
                       const char* array_name) {
    if (buf.ndim != 1) {
        throw std::runtime_error(std::string(array_name) + " must be a 1D array");
    }
    if (buf.size < fft_size) {
        throw std::runtime_error(std::string(array_name) + " length must be >= fft_size");
    }
    return static_cast<int>(buf.size);
}

int validate_spectrogram(const py::buffer_info& buf, int fft_size,
                         const char* function_name) {
    validate_fft_size(fft_size);
    if (buf.ndim != 2) {
        throw std::runtime_error(
            std::string(function_name) + ": spectrogram must be 2D array (n_frames, n_bins)");
    }
    if (buf.shape[0] < 1) {
        throw std::runtime_error(std::string(function_name) + ": n_frames must be >= 1");
    }
    int n_bins = fft_size / 2 + 1;
    if (buf.shape[1] != n_bins) {
        throw std::runtime_error(
            std::string(function_name) + ": spectrogram must have fft_size // 2 + 1 bins");
    }
    ensure_finite_values(buf, "spectrogram must be finite");
    return static_cast<int>(buf.shape[0]);
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_interference, m) {
    m.doc() = "GPU-accelerated GNSS interference detection and excision";

    py::enum_<gnss_gpu::InterferenceType>(m, "InterferenceType")
        .value("NONE", gnss_gpu::InterferenceType::NONE)
        .value("CW", gnss_gpu::InterferenceType::CW)
        .value("NARROWBAND", gnss_gpu::InterferenceType::NARROWBAND)
        .value("PULSED", gnss_gpu::InterferenceType::PULSED)
        .value("CHIRP", gnss_gpu::InterferenceType::CHIRP);

    m.def("compute_stft", [](FloatArray signal, int fft_size, int hop_size,
                              double sampling_freq) {
        auto buf = signal.request();
        validate_stft_options(fft_size, hop_size, sampling_freq);
        int n_samples = validate_signal_1d(buf, fft_size, "signal");
        ensure_finite_values(buf, "signal must be finite");
        int n_frames = (n_samples - fft_size) / hop_size + 1;
        int n_bins = fft_size / 2 + 1;

        auto spectrogram = py::array_t<float>({n_frames, n_bins});

        gnss_gpu::compute_stft(static_cast<float*>(buf.ptr),
                               static_cast<float*>(spectrogram.request().ptr),
                               n_samples, fft_size, hop_size, sampling_freq);

        return spectrogram;
    }, "Compute STFT power spectrogram (dB)",
       py::arg("signal"), py::arg("fft_size") = 1024,
       py::arg("hop_size") = 256, py::arg("sampling_freq") = 1.0);

    m.def("detect_interference", [](FloatArray spectrogram, int fft_size,
                                     double sampling_freq, float threshold_db,
                                     int max_detections) {
        auto buf = spectrogram.request();
        validate_sampling_freq(sampling_freq);
        validate_threshold(threshold_db);
        if (max_detections < 1) {
            throw std::runtime_error("max_detections must be >= 1");
        }
        int n_frames = validate_spectrogram(buf, fft_size, "detect_interference");

        std::vector<gnss_gpu::InterferenceDetection> dets(max_detections);
        int n_det = gnss_gpu::detect_interference(
            static_cast<float*>(buf.ptr), dets.data(), max_detections,
            n_frames, fft_size, sampling_freq, threshold_db);

        py::list result;
        for (int i = 0; i < n_det; i++) {
            py::dict d;
            d["type"] = static_cast<int>(dets[i].type);
            d["type_name"] = [](gnss_gpu::InterferenceType t) -> std::string {
                switch (t) {
                    case gnss_gpu::InterferenceType::NONE: return "NONE";
                    case gnss_gpu::InterferenceType::CW: return "CW";
                    case gnss_gpu::InterferenceType::NARROWBAND: return "NARROWBAND";
                    case gnss_gpu::InterferenceType::PULSED: return "PULSED";
                    case gnss_gpu::InterferenceType::CHIRP: return "CHIRP";
                }
                return "UNKNOWN";
            }(dets[i].type);
            d["center_freq_hz"] = dets[i].center_freq_hz;
            d["bandwidth_hz"] = dets[i].bandwidth_hz;
            d["power_db"] = dets[i].power_db;
            d["start_frame"] = dets[i].start_frame;
            d["end_frame"] = dets[i].end_frame;
            result.append(d);
        }
        return result;
    }, "Detect interference in spectrogram",
       py::arg("spectrogram"), py::arg("fft_size") = 1024,
       py::arg("sampling_freq") = 1.0, py::arg("threshold_db") = 15.0f,
       py::arg("max_detections") = 32);

    m.def("excise_interference", [](FloatArray input, FloatArray spectrogram,
                                     int fft_size, int hop_size, float threshold_db) {
        auto buf_in = input.request();
        auto buf_spec = spectrogram.request();
        validate_stft_options(fft_size, hop_size, 1.0);
        validate_threshold(threshold_db);
        int n_samples = validate_signal_1d(buf_in, fft_size, "input");
        ensure_finite_values(buf_in, "input must be finite");
        int n_frames = validate_spectrogram(buf_spec, fft_size, "excise_interference");
        int expected_frames = (n_samples - fft_size) / hop_size + 1;
        if (n_frames != expected_frames) {
            throw std::runtime_error(
                "excise_interference: spectrogram n_frames must match input, fft_size, and hop_size");
        }

        auto output = py::array_t<float>(std::vector<py::ssize_t>{n_samples});

        gnss_gpu::excise_interference(static_cast<float*>(buf_in.ptr),
                                      static_cast<float*>(output.request().ptr),
                                      static_cast<float*>(buf_spec.ptr),
                                      n_samples, fft_size, hop_size, threshold_db);

        return output;
    }, "Excise interference from signal",
       py::arg("input"), py::arg("spectrogram"),
       py::arg("fft_size") = 1024, py::arg("hop_size") = 256,
       py::arg("threshold_db") = 15.0f);
}
