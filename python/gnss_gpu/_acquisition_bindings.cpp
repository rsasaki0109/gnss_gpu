#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "gnss_gpu/acquisition.h"

namespace py = pybind11;

PYBIND11_MODULE(_gnss_gpu_acq, m) {
    m.doc() = "GPU-accelerated GNSS signal acquisition";

    m.def("generate_ca_code", [](int prn) {
        int code[1023];
        gnss_gpu::generate_ca_code(prn, code);
        py::list result;
        for (int i = 0; i < 1023; i++) {
            result.append(code[i]);
        }
        return result;
    }, "Generate GPS C/A code (1023 chips, +1/-1)",
       py::arg("prn"));

    m.def("acquire_parallel", [](py::array_t<float> signal,
                                  double sampling_freq, double intermediate_freq,
                                  py::array_t<int> prn_list,
                                  double doppler_range, double doppler_step,
                                  float threshold) {
        auto sig_buf = signal.request();
        auto prn_buf = prn_list.request();
        if (sig_buf.ndim != 1)
            throw std::runtime_error("signal must be a 1D array");
        if (prn_buf.ndim != 1)
            throw std::runtime_error("prn_list must be a 1D array");
        int n_samples = sig_buf.size;
        int n_prn = prn_buf.size;

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
