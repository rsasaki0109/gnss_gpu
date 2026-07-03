#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "gnss_gpu/ephemeris.h"
#include "gnss_gpu/pybind_validation.h"

namespace py = pybind11;

namespace {

using gnss_gpu::pybind_validation::DoubleArray;
namespace pv = gnss_gpu::pybind_validation;

void validate_params_flat(const py::buffer_info& buf, int n_sat) {
  pv::validate_n_sat(n_sat);
  if (buf.ndim != 1) {
    throw std::runtime_error("params_flat must be a 1D array");
  }
  if (buf.size == 0) {
    throw std::runtime_error("params_flat must be non-empty");
  }
  const py::ssize_t min_bytes =
      static_cast<py::ssize_t>(n_sat) *
      static_cast<py::ssize_t>(sizeof(gnss_gpu::EphemerisParams));
  if (buf.size * static_cast<py::ssize_t>(sizeof(double)) < min_bytes) {
    throw std::runtime_error("params_flat is too small for n_sat EphemerisParams structs");
  }
  pv::ensure_finite_doubles(buf, "params_flat must be finite");
}

void validate_gps_times(const py::buffer_info& buf) {
  if (buf.ndim != 1) {
    throw std::runtime_error("gps_times must be a 1D array");
  }
  if (buf.size == 0) {
    throw std::runtime_error("gps_times must be non-empty");
  }
  pv::ensure_finite_doubles(buf, "gps_times must be finite");
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_ephemeris, m) {
    m.doc() = "GPU-accelerated broadcast ephemeris computation";

    // Expose EphemerisParams as a Python class for direct construction
    py::class_<gnss_gpu::EphemerisParams>(m, "EphemerisParams")
        .def(py::init<>())
        .def_readwrite("sqrt_a", &gnss_gpu::EphemerisParams::sqrt_a)
        .def_readwrite("e", &gnss_gpu::EphemerisParams::e)
        .def_readwrite("i0", &gnss_gpu::EphemerisParams::i0)
        .def_readwrite("omega0", &gnss_gpu::EphemerisParams::omega0)
        .def_readwrite("omega", &gnss_gpu::EphemerisParams::omega)
        .def_readwrite("M0", &gnss_gpu::EphemerisParams::M0)
        .def_readwrite("delta_n", &gnss_gpu::EphemerisParams::delta_n)
        .def_readwrite("omega_dot", &gnss_gpu::EphemerisParams::omega_dot)
        .def_readwrite("idot", &gnss_gpu::EphemerisParams::idot)
        .def_readwrite("cuc", &gnss_gpu::EphemerisParams::cuc)
        .def_readwrite("cus", &gnss_gpu::EphemerisParams::cus)
        .def_readwrite("crc", &gnss_gpu::EphemerisParams::crc)
        .def_readwrite("crs", &gnss_gpu::EphemerisParams::crs)
        .def_readwrite("cic", &gnss_gpu::EphemerisParams::cic)
        .def_readwrite("cis", &gnss_gpu::EphemerisParams::cis)
        .def_readwrite("toe", &gnss_gpu::EphemerisParams::toe)
        .def_readwrite("af0", &gnss_gpu::EphemerisParams::af0)
        .def_readwrite("af1", &gnss_gpu::EphemerisParams::af1)
        .def_readwrite("af2", &gnss_gpu::EphemerisParams::af2)
        .def_readwrite("toc", &gnss_gpu::EphemerisParams::toc)
        .def_readwrite("tgd", &gnss_gpu::EphemerisParams::tgd)
        .def_readwrite("week", &gnss_gpu::EphemerisParams::week);

    // Single-epoch satellite position computation
    m.def("compute_satellite_position",
        [](DoubleArray params_flat, double gps_time, int n_sat) {
            auto buf = params_flat.request();
            validate_params_flat(buf, n_sat);
            pv::validate_finite(gps_time, "gps_time");

            const gnss_gpu::EphemerisParams* params =
                reinterpret_cast<const gnss_gpu::EphemerisParams*>(buf.ptr);

            auto sat_pos = py::array_t<double>({n_sat, 3});
            auto sat_clk = py::array_t<double>(std::vector<py::ssize_t>{n_sat});

            gnss_gpu::compute_satellite_position(
                params, gps_time,
                static_cast<double*>(sat_pos.request().ptr),
                static_cast<double*>(sat_clk.request().ptr),
                n_sat);

            return py::make_tuple(sat_pos, sat_clk);
        },
        "Compute satellite ECEF positions and clock corrections at given GPS time",
        py::arg("params_flat"), py::arg("gps_time"), py::arg("n_sat"));

    // Batch satellite position computation
    m.def("compute_satellite_position_batch",
        [](DoubleArray params_flat, DoubleArray gps_times, int n_sat) {
            auto buf_params = params_flat.request();
            auto buf_times = gps_times.request();
            validate_params_flat(buf_params, n_sat);
            validate_gps_times(buf_times);
            int n_epoch = static_cast<int>(buf_times.size);

            const gnss_gpu::EphemerisParams* params =
                reinterpret_cast<const gnss_gpu::EphemerisParams*>(buf_params.ptr);

            auto sat_pos = py::array_t<double>({n_epoch, n_sat, 3});
            auto sat_clk = py::array_t<double>({n_epoch, n_sat});

            gnss_gpu::compute_satellite_position_batch(
                params, static_cast<const double*>(buf_times.ptr),
                static_cast<double*>(sat_pos.request().ptr),
                static_cast<double*>(sat_clk.request().ptr),
                n_epoch, n_sat);

            return py::make_tuple(sat_pos, sat_clk);
        },
        "Batch compute satellite positions for multiple epochs",
        py::arg("params_flat"), py::arg("gps_times"), py::arg("n_sat"));

    // Provide the size of EphemerisParams for numpy dtype construction
    m.attr("EPHEMERIS_PARAMS_SIZE") = py::int_(sizeof(gnss_gpu::EphemerisParams));
    m.attr("EPHEMERIS_PARAMS_N_DOUBLES") = py::int_(
        (sizeof(gnss_gpu::EphemerisParams) - sizeof(int)) / sizeof(double));
}
