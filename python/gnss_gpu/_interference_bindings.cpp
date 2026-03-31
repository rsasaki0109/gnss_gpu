#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gnss_gpu/interference.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_interference, m) {
    m.doc() = "GPU-accelerated GNSS interference detection and excision";

    py::enum_<gnss_gpu::InterferenceType>(m, "InterferenceType")
        .value("NONE", gnss_gpu::InterferenceType::NONE)
        .value("CW", gnss_gpu::InterferenceType::CW)
        .value("NARROWBAND", gnss_gpu::InterferenceType::NARROWBAND)
        .value("PULSED", gnss_gpu::InterferenceType::PULSED)
        .value("CHIRP", gnss_gpu::InterferenceType::CHIRP);

    m.def("compute_stft", [](py::array_t<float> signal, int fft_size, int hop_size,
                              double sampling_freq) {
        auto buf = signal.request();
        int n_samples = buf.size;
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

    m.def("detect_interference", [](py::array_t<float> spectrogram, int fft_size,
                                     double sampling_freq, float threshold_db,
                                     int max_detections) {
        auto buf = spectrogram.request();
        if (buf.ndim != 2)
          throw std::runtime_error("spectrogram must be 2D array (n_frames, n_bins)");
        int n_frames = buf.shape[0];

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

    m.def("excise_interference", [](py::array_t<float> input, py::array_t<float> spectrogram,
                                     int fft_size, int hop_size, float threshold_db) {
        auto buf_in = input.request();
        int n_samples = buf_in.size;

        auto output = py::array_t<float>(std::vector<ssize_t>{n_samples});

        gnss_gpu::excise_interference(static_cast<float*>(buf_in.ptr),
                                      static_cast<float*>(output.request().ptr),
                                      static_cast<float*>(spectrogram.request().ptr),
                                      n_samples, fft_size, hop_size, threshold_db);

        return output;
    }, "Excise interference from signal",
       py::arg("input"), py::arg("spectrogram"),
       py::arg("fft_size") = 1024, py::arg("hop_size") = 256,
       py::arg("threshold_db") = 15.0f);
}
