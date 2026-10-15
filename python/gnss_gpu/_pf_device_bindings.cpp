#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <string>
#include "gnss_gpu/pf_device.h"

namespace py = pybind11;

namespace {

void validate_positive_finite(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::runtime_error(std::string(name) + " must be positive and finite");
    }
}

void validate_finite(double value, const char* name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(name) + " must be finite");
    }
}

void validate_n_particles(int n_particles) {
    if (n_particles < 1) {
        throw std::runtime_error("n_particles must be >= 1");
    }
}

void validate_n_sat(int n_sat) {
    if (n_sat < 1) {
        throw std::runtime_error("n_sat must be >= 1");
    }
}

void validate_device_state(const gnss_gpu::PFDeviceState* state, const char* function_name) {
    if (state == nullptr) {
        throw std::runtime_error(std::string(function_name) + ": state must not be null");
    }
    if (!state->allocated) {
        throw std::runtime_error(std::string(function_name) + ": state is not allocated");
    }
}

void validate_dt(double dt) {
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::runtime_error("dt must be non-negative and finite");
    }
}

void ensure_finite_doubles(const py::buffer_info& buf, const char* message) {
    const auto* ptr = static_cast<const double*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(ptr[i])) {
            throw std::runtime_error(message);
        }
    }
}

void validate_sat_ecef(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim == 2) {
        if (buf.shape[0] != n_sat || buf.shape[1] != 3) {
            throw std::runtime_error("sat_ecef shape must match n_sat satellites");
        }
    } else if (buf.ndim == 1) {
        if (buf.size != static_cast<py::ssize_t>(n_sat) * 3) {
            throw std::runtime_error("sat_ecef shape must match n_sat satellites");
        }
    } else {
        throw std::runtime_error("sat_ecef must be 1D or 2D");
    }
    ensure_finite_doubles(buf, "sat_ecef must be finite");
}

void validate_pseudoranges(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim != 1 || buf.size != n_sat) {
        throw std::runtime_error("pseudoranges length must match n_sat");
    }
    ensure_finite_doubles(buf, "pseudoranges must be finite");
}

void validate_weights_sat(const py::buffer_info& buf, int n_sat) {
    if (buf.ndim != 1 || buf.size != n_sat) {
        throw std::runtime_error("weights_sat length must match n_sat");
    }
    const auto* ptr = static_cast<const double*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(ptr[i])) {
            throw std::runtime_error("weights_sat must be finite");
        }
        if (ptr[i] < 0.0) {
            throw std::runtime_error("weights_sat must be non-negative");
        }
    }
}

}  // namespace

// Custom destructor for PFDeviceState managed by pybind11.
// When pybind11 deletes the Python wrapper, this frees GPU resources
// and then deletes the C++ object, preventing double-free.
struct PFDeviceStateDeleter {
    void operator()(gnss_gpu::PFDeviceState* state) {
        if (state) {
            gnss_gpu::pf_device_destroy(state);
        }
    }
};

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
static bool cdf_cpu_has_avx2() {
#if defined(_MSC_VER)
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7) return false;
    __cpuidex(info, 1, 0);
    const int avx_osxsave = (1 << 28) | (1 << 27);
    if ((info[2] & avx_osxsave) != avx_osxsave) return false;
    if ((_xgetbv(0) & 6) != 6) return false;
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
#else
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#endif
}

// Keep AVX2 instructions inside a separately dispatched function. Do not
// raise the translation unit's baseline ISA or require AVX2 on every host.
#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((target("avx2"), noinline))
#endif
static bool finite_monotone_cdf_avx2(const double* input, py::ssize_t size) {
    if (!std::isfinite(input[0])) return false;
    const __m256i exponent = _mm256_set1_epi64x(0x7ff0000000000000LL);
    py::ssize_t i = 1;
    for (; i + 7 < size; i += 8) {
        const __m256d a = _mm256_loadu_pd(input + i);
        const __m256d b = _mm256_loadu_pd(input + i + 4);
        const __m256i bad_a = _mm256_cmpeq_epi64(
            _mm256_and_si256(_mm256_castpd_si256(a), exponent), exponent);
        const __m256i bad_b = _mm256_cmpeq_epi64(
            _mm256_and_si256(_mm256_castpd_si256(b), exponent), exponent);
        if (_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_or_si256(bad_a, bad_b)))) return false;
        const __m256d before_a = _mm256_loadu_pd(input + i - 1);
        const __m256d before_b = _mm256_loadu_pd(input + i + 3);
        if (_mm256_movemask_pd(_mm256_or_pd(_mm256_cmp_pd(a, before_a, _CMP_LT_OQ),
                                          _mm256_cmp_pd(b, before_b, _CMP_LT_OQ)))) return false;
    }
    for (; i < size; ++i) {
        if (!std::isfinite(input[i]) || input[i] < input[i - 1]) return false;
    }
    return true;
}
#endif
#endif

// Validate every element before any state/scratch mutation. x86-64 has SSE2
// as its baseline ISA; other targets and small inputs retain the scalar path.
static bool finite_monotone_cdf(const double* input, py::ssize_t size) {
#if defined(_M_X64) || defined(__x86_64__)
    if (size >= 65536) {
#if defined(_MSC_VER) || defined(__GNUC__) || defined(__clang__)
        static const bool has_avx2 = cdf_cpu_has_avx2();
        if (has_avx2) return finite_monotone_cdf_avx2(input, size);
#endif
        if (!std::isfinite(input[0])) return false;
        const __m128i exponent = _mm_set1_epi64x(0x7ff0000000000000LL);
        py::ssize_t i = 1;
        for (; i + 3 < size; i += 4) {
            const __m128d a = _mm_loadu_pd(input + i);
            const __m128d b = _mm_loadu_pd(input + i + 2);
            // Integer exponent tests reject Inf and both NaN kinds without
            // performing floating-point comparisons on signaling NaNs.
            const __m128i bad_a = _mm_cmpeq_epi32(
                _mm_and_si128(_mm_castpd_si128(a), exponent), exponent);
            const __m128i bad_b = _mm_cmpeq_epi32(
                _mm_and_si128(_mm_castpd_si128(b), exponent), exponent);
            if (_mm_movemask_pd(_mm_castsi128_pd(_mm_or_si128(bad_a, bad_b)))) return false;
            const __m128d before_a = _mm_loadu_pd(input + i - 1);
            const __m128d before_b = _mm_loadu_pd(input + i + 1);
            if (_mm_movemask_pd(_mm_or_pd(_mm_cmplt_pd(a, before_a),
                                         _mm_cmplt_pd(b, before_b)))) return false;
        }
        for (; i < size; ++i) {
            if (!std::isfinite(input[i]) || input[i] < input[i - 1]) return false;
        }
        return true;
    }
#endif
    for (py::ssize_t i = 0; i < size; ++i) {
        if (!std::isfinite(input[i]) || (i > 0 && input[i] < input[i - 1])) return false;
    }
    return true;
}

PYBIND11_MODULE(_gnss_gpu_pf_device, m) {
    m.doc() = "GPU-accelerated Particle Filter with persistent device memory";

    // Expose PFDeviceState as opaque Python type with custom destructor.
    // The custom holder ensures pf_device_destroy is called exactly once
    // (either explicitly or when Python GC collects the object).
    py::class_<gnss_gpu::PFDeviceState, std::unique_ptr<gnss_gpu::PFDeviceState, PFDeviceStateDeleter>>(m, "PFDeviceState",
        "Opaque handle to device-resident particle state. "
        "Memory lives on GPU between calls.")
        .def_readonly("n_particles", &gnss_gpu::PFDeviceState::n_particles)
        .def_readonly("allocated", &gnss_gpu::PFDeviceState::allocated);

    m.def("pf_device_create", [](int n_particles) {
        validate_n_particles(n_particles);
        return std::unique_ptr<gnss_gpu::PFDeviceState, PFDeviceStateDeleter>(
            gnss_gpu::pf_device_create(n_particles), PFDeviceStateDeleter());
    }, "Allocate persistent GPU memory for particle state",
       py::arg("n_particles"));

    m.def("pf_device_destroy", [](gnss_gpu::PFDeviceState* state) {
        // Only free GPU resources; pybind11 still owns the pointer via unique_ptr.
        // Mark as deallocated so the destructor won't double-free GPU memory.
        if (state && state->allocated) {
            gnss_gpu::pf_device_destroy_resources(state);
        }
    }, "Free all GPU memory for particle state",
       py::arg("state"));

    m.def("pf_device_initialize", [](gnss_gpu::PFDeviceState* state,
                                     double init_x, double init_y, double init_z, double init_cb,
                                     double spread_pos, double spread_cb,
                                     unsigned long long seed,
                                     double init_vx, double init_vy, double init_vz,
                                     double spread_vel,
                                     double init_vel_sigma) {
        validate_device_state(state, "pf_device_initialize");
        validate_finite(init_x, "init_x");
        validate_finite(init_y, "init_y");
        validate_finite(init_z, "init_z");
        validate_finite(init_cb, "init_cb");
        validate_positive_finite(spread_pos, "spread_pos");
        validate_positive_finite(spread_cb, "spread_cb");
        validate_finite(init_vx, "init_vx");
        validate_finite(init_vy, "init_vy");
        validate_finite(init_vz, "init_vz");
        validate_finite(spread_vel, "spread_vel");
        validate_finite(init_vel_sigma, "init_vel_sigma");
        gnss_gpu::pf_device_initialize(state, init_x, init_y, init_z, init_cb,
                                       spread_pos, spread_cb, seed,
                                       init_vx, init_vy, init_vz, spread_vel,
                                       init_vel_sigma);
    }, "Initialize particles on device (no H2D copy)",
       py::arg("state"),
       py::arg("init_x"), py::arg("init_y"), py::arg("init_z"), py::arg("init_cb"),
       py::arg("spread_pos"), py::arg("spread_cb"), py::arg("seed"),
       py::arg("init_vx") = 0.0, py::arg("init_vy") = 0.0, py::arg("init_vz") = 0.0,
       py::arg("spread_vel") = 0.0,
       py::arg("init_vel_sigma") = 0.0);

    m.def("pf_device_predict", [](gnss_gpu::PFDeviceState* state,
                                  double vx, double vy, double vz,
                                  double dt, double sigma_pos, double sigma_cb,
                                  unsigned long long seed, int step,
                                  double sigma_vel,
                                  double velocity_guide_alpha,
                                  bool velocity_kf,
                                  double velocity_process_noise) {
        validate_device_state(state, "pf_device_predict");
        validate_finite(vx, "vx");
        validate_finite(vy, "vy");
        validate_finite(vz, "vz");
        validate_dt(dt);
        validate_positive_finite(sigma_pos, "sigma_pos");
        validate_positive_finite(sigma_cb, "sigma_cb");
        validate_finite(sigma_vel, "sigma_vel");
        validate_finite(velocity_guide_alpha, "velocity_guide_alpha");
        validate_finite(velocity_process_noise, "velocity_process_noise");
        gnss_gpu::pf_device_predict(
            state, vx, vy, vz, dt, sigma_pos, sigma_cb, seed, step,
            sigma_vel, velocity_guide_alpha, velocity_kf,
            velocity_process_noise);
    }, "Predict step - operates entirely on device memory",
       py::arg("state"),
       py::arg("vx"), py::arg("vy"), py::arg("vz"),
       py::arg("dt"), py::arg("sigma_pos"), py::arg("sigma_cb"),
       py::arg("seed"), py::arg("step"),
       py::arg("sigma_vel") = 0.0,
       py::arg("velocity_guide_alpha") = 1.0,
       py::arg("velocity_kf") = false,
       py::arg("velocity_process_noise") = 0.0);

    m.def("pf_device_weight", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> pseudoranges,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double sigma_pr, double nu,
                                 double per_particle_nlos_threshold_m,
                                 bool per_particle_huber,
                                 double per_particle_huber_k) {
        validate_device_state(state, "pf_device_weight");
        validate_n_sat(n_sat);
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_pr = pseudoranges.request();
        py::buffer_info b_w = weights_sat.request();
        validate_sat_ecef(b_sat, n_sat);
        validate_pseudoranges(b_pr, n_sat);
        validate_weights_sat(b_w, n_sat);
        validate_positive_finite(sigma_pr, "sigma_pr");
        validate_finite(nu, "nu");
        validate_finite(per_particle_nlos_threshold_m, "per_particle_nlos_threshold_m");
        validate_finite(per_particle_huber_k, "per_particle_huber_k");
        gnss_gpu::pf_device_weight(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_pr.ptr),
            static_cast<double*>(b_w.ptr),
            n_sat, sigma_pr, nu, per_particle_nlos_threshold_m,
            per_particle_huber, per_particle_huber_k);
    }, "Weight update with optional robust Student's t or Huber likelihood",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights_sat"),
       py::arg("n_sat"), py::arg("sigma_pr"), py::arg("nu") = 0.0,
       py::arg("per_particle_nlos_threshold_m") = 0.0,
       py::arg("per_particle_huber") = false,
       py::arg("per_particle_huber_k") = 1.5);

    m.def("pf_device_weight_dd_pseudorange", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef_k,
                                 py::array_t<double> ref_ecef,
                                 py::array_t<double> dd_pseudorange,
                                 py::array_t<double> base_range_k,
                                 py::array_t<double> base_range_ref,
                                 py::array_t<double> weights_dd,
                                 int n_dd, double sigma_pr,
                                 double per_particle_nlos_threshold_m,
                                 bool per_particle_huber,
                                 double per_particle_huber_k) {
        py::buffer_info b_sk = sat_ecef_k.request();
        py::buffer_info b_ref = ref_ecef.request();
        py::buffer_info b_dd = dd_pseudorange.request();
        py::buffer_info b_brk = base_range_k.request();
        py::buffer_info b_brr = base_range_ref.request();
        py::buffer_info b_w = weights_dd.request();
        gnss_gpu::pf_device_weight_dd_pseudorange(state,
            static_cast<double*>(b_sk.ptr),
            static_cast<double*>(b_ref.ptr),
            static_cast<double*>(b_dd.ptr),
            static_cast<double*>(b_brk.ptr),
            static_cast<double*>(b_brr.ptr),
            static_cast<double*>(b_w.ptr),
            n_dd, sigma_pr, per_particle_nlos_threshold_m,
            per_particle_huber, per_particle_huber_k);
    }, "Weight update using DD pseudorange likelihood (no clock bias needed)",
       py::arg("state"),
       py::arg("sat_ecef_k"), py::arg("ref_ecef"),
       py::arg("dd_pseudorange"), py::arg("base_range_k"),
       py::arg("base_range_ref"), py::arg("weights_dd"),
       py::arg("n_dd"), py::arg("sigma_pr"),
       py::arg("per_particle_nlos_threshold_m") = 0.0,
       py::arg("per_particle_huber") = false,
       py::arg("per_particle_huber_k") = 1.5);

    m.def("pf_device_weight_gmm", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> pseudoranges,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double sigma_pr,
                                 double w_los, double mu_nlos, double sigma_nlos) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_pr = pseudoranges.request();
        py::buffer_info b_w = weights_sat.request();
        gnss_gpu::pf_device_weight_gmm(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_pr.ptr),
            static_cast<double*>(b_w.ptr),
            n_sat, sigma_pr, w_los, mu_nlos, sigma_nlos);
    }, "Weight update using GMM likelihood (LOS + NLOS mixture)",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("pseudoranges"), py::arg("weights_sat"),
       py::arg("n_sat"), py::arg("sigma_pr"),
       py::arg("w_los") = 0.7, py::arg("mu_nlos") = 15.0, py::arg("sigma_nlos") = 30.0);

    m.def("pf_device_weight_carrier_afv", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> carrier_phase,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double wavelength, double sigma_cycles) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_cp = carrier_phase.request();
        py::buffer_info b_w = weights_sat.request();
        gnss_gpu::pf_device_weight_carrier_afv(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_cp.ptr),
            static_cast<double*>(b_w.ptr),
            n_sat, wavelength, sigma_cycles);
    }, "Weight update using carrier phase AFV likelihood (no ambiguity resolution needed)",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("carrier_phase"), py::arg("weights_sat"),
       py::arg("n_sat"), py::arg("wavelength") = 0.190293673, py::arg("sigma_cycles") = 0.05);

    m.def("pf_device_weight_dd_carrier_afv", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef_k,
                                 py::array_t<double> ref_ecef,
                                 py::array_t<double> dd_carrier,
                                 py::array_t<double> base_range_k,
                                 py::array_t<double> base_range_ref,
                                 py::array_t<double> weights_dd,
                                 py::array_t<double> wavelengths_m,
                                 int n_dd, double sigma_cycles,
                                 double per_particle_nlos_threshold_cycles,
                                 bool per_particle_huber,
                                 double per_particle_huber_k) {
        py::buffer_info b_sk = sat_ecef_k.request();
        py::buffer_info b_ref = ref_ecef.request();
        py::buffer_info b_dd = dd_carrier.request();
        py::buffer_info b_brk = base_range_k.request();
        py::buffer_info b_brr = base_range_ref.request();
        py::buffer_info b_w = weights_dd.request();
        py::buffer_info b_wl = wavelengths_m.request();
        gnss_gpu::pf_device_weight_dd_carrier_afv(state,
            static_cast<double*>(b_sk.ptr),
            static_cast<double*>(b_ref.ptr),
            static_cast<double*>(b_dd.ptr),
            static_cast<double*>(b_brk.ptr),
            static_cast<double*>(b_brr.ptr),
            static_cast<double*>(b_w.ptr),
            static_cast<double*>(b_wl.ptr),
            n_dd, sigma_cycles, per_particle_nlos_threshold_cycles,
            per_particle_huber, per_particle_huber_k);
    }, "Weight update using DD carrier phase AFV (no clock bias needed)",
       py::arg("state"),
       py::arg("sat_ecef_k"), py::arg("ref_ecef"),
       py::arg("dd_carrier"), py::arg("base_range_k"),
       py::arg("base_range_ref"), py::arg("weights_dd"),
       py::arg("wavelengths_m"),
       py::arg("n_dd"), py::arg("sigma_cycles") = 0.05,
       py::arg("per_particle_nlos_threshold_cycles") = 0.0,
       py::arg("per_particle_huber") = false,
       py::arg("per_particle_huber_k") = 1.5);

    m.def("pf_device_weight_dd_joint", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef_k,
                                 py::array_t<double> ref_ecef,
                                 py::array_t<double> dd_pseudorange,
                                 py::array_t<double> dd_carrier,
                                 py::array_t<double> base_range_k,
                                 py::array_t<double> base_range_ref,
                                 py::array_t<double> weights_dd,
                                 py::array_t<double> wavelengths_m,
                                 int n_dd, double sigma_pr, double sigma_cycles,
                                 double per_particle_nlos_threshold_m,
                                 double per_particle_nlos_threshold_cycles,
                                 bool per_particle_huber,
                                 double per_particle_huber_dd_pr_k,
                                 double per_particle_huber_dd_carrier_k) {
        gnss_gpu::pf_device_weight_dd_joint(
            state,
            static_cast<double*>(sat_ecef_k.request().ptr),
            static_cast<double*>(ref_ecef.request().ptr),
            static_cast<double*>(dd_pseudorange.request().ptr),
            static_cast<double*>(dd_carrier.request().ptr),
            static_cast<double*>(base_range_k.request().ptr),
            static_cast<double*>(base_range_ref.request().ptr),
            static_cast<double*>(weights_dd.request().ptr),
            static_cast<double*>(wavelengths_m.request().ptr),
            n_dd, sigma_pr, sigma_cycles,
            per_particle_nlos_threshold_m,
            per_particle_nlos_threshold_cycles,
            per_particle_huber,
            per_particle_huber_dd_pr_k,
            per_particle_huber_dd_carrier_k);
    }, "Joint DD pseudorange and carrier AFV update with one observation transfer",
       py::arg("state"), py::arg("sat_ecef_k"), py::arg("ref_ecef"),
       py::arg("dd_pseudorange"), py::arg("dd_carrier"),
       py::arg("base_range_k"), py::arg("base_range_ref"),
       py::arg("weights_dd"), py::arg("wavelengths_m"), py::arg("n_dd"),
       py::arg("sigma_pr") = 0.75, py::arg("sigma_cycles") = 0.05,
       py::arg("per_particle_nlos_threshold_m") = 0.0,
       py::arg("per_particle_nlos_threshold_cycles") = 0.0,
       py::arg("per_particle_huber") = false,
       py::arg("per_particle_huber_dd_pr_k") = 1.5,
       py::arg("per_particle_huber_dd_carrier_k") = 1.5);

    m.def("pf_device_weight_doppler", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> sat_vel,
                                 py::array_t<double> doppler_hz,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double wavelength_m,
                                 double sigma_mps,
                                 double velocity_update_gain,
                                 double max_velocity_update_mps) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_sat_vel = sat_vel.request();
        py::buffer_info b_doppler = doppler_hz.request();
        py::buffer_info b_weights = weights_sat.request();
        gnss_gpu::pf_device_weight_doppler(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_sat_vel.ptr),
            static_cast<double*>(b_doppler.ptr),
            static_cast<double*>(b_weights.ptr),
            n_sat, wavelength_m, sigma_mps,
            velocity_update_gain, max_velocity_update_mps);
    }, "Velocity-domain update using Doppler observations (range-rate = -doppler * wavelength)",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("sat_vel"),
       py::arg("doppler_hz"), py::arg("weights_sat"),
       py::arg("n_sat"),
       py::arg("wavelength_m") = 0.19029367279836488,
       py::arg("sigma_mps") = 0.5,
       py::arg("velocity_update_gain") = 0.25,
       py::arg("max_velocity_update_mps") = 10.0);

    m.def("pf_device_doppler_kf_update", [](gnss_gpu::PFDeviceState* state,
                                 py::array_t<double> sat_ecef,
                                 py::array_t<double> sat_vel,
                                 py::array_t<double> doppler_hz,
                                 py::array_t<double> weights_sat,
                                 int n_sat, double wavelength_m,
                                 double sigma_mps) {
        py::buffer_info b_sat = sat_ecef.request();
        py::buffer_info b_sat_vel = sat_vel.request();
        py::buffer_info b_doppler = doppler_hz.request();
        py::buffer_info b_weights = weights_sat.request();
        gnss_gpu::pf_device_doppler_kf_update(state,
            static_cast<double*>(b_sat.ptr),
            static_cast<double*>(b_sat_vel.ptr),
            static_cast<double*>(b_doppler.ptr),
            static_cast<double*>(b_weights.ptr),
            n_sat, wavelength_m, sigma_mps);
    }, "Proper RBPF Doppler KF update over per-particle velocity Gaussian",
       py::arg("state"),
       py::arg("sat_ecef"), py::arg("sat_vel"),
       py::arg("doppler_hz"), py::arg("weights_sat"),
       py::arg("n_sat"),
       py::arg("wavelength_m") = 0.19029367279836488,
       py::arg("sigma_mps") = 0.5);

    m.def("pf_device_position_update", [](gnss_gpu::PFDeviceState* state,
                                         double ref_x, double ref_y, double ref_z,
                                         double sigma_pos) {
        gnss_gpu::pf_device_position_update(state, ref_x, ref_y, ref_z, sigma_pos);
    }, "Apply position-domain soft constraint from external estimate",
       py::arg("state"),
       py::arg("ref_x"), py::arg("ref_y"), py::arg("ref_z"),
       py::arg("sigma_pos"));

    m.def("pf_device_shift_clock_bias", [](gnss_gpu::PFDeviceState* state,
                                           double shift) {
        gnss_gpu::pf_device_shift_clock_bias(state, shift);
    }, "Shift all particles' clock bias by a constant offset",
       py::arg("state"), py::arg("shift"));

    m.def("pf_device_shift_position", [](gnss_gpu::PFDeviceState* state,
                                         double dx, double dy, double dz) {
        gnss_gpu::pf_device_shift_position(state, dx, dy, dz);
    }, "Shift all particle positions by a constant ECEF delta, preserving weights",
       py::arg("state"), py::arg("dx"), py::arg("dy"), py::arg("dz"));

    m.def("pf_device_reset_position", [](gnss_gpu::PFDeviceState* state,
                                         double ref_x, double ref_y, double ref_z,
                                         double sigma_pos,
                                         unsigned long long seed, int step) {
        gnss_gpu::pf_device_reset_position(state, ref_x, ref_y, ref_z,
                                           sigma_pos, seed, step);
    }, "Redraw particle positions around a reference and reset weights",
       py::arg("state"), py::arg("ref_x"), py::arg("ref_y"), py::arg("ref_z"),
       py::arg("sigma_pos"), py::arg("seed"), py::arg("step"));

    m.def("pf_device_ess", [](const gnss_gpu::PFDeviceState* state) {
        return gnss_gpu::pf_device_ess(state);
    }, "Compute ESS on device, return scalar to host",
       py::arg("state"));

    m.def("pf_device_position_spread", [](const gnss_gpu::PFDeviceState* state,
                                          double center_x, double center_y, double center_z) {
        return gnss_gpu::pf_device_position_spread(state, center_x, center_y, center_z);
    }, "Compute weighted RMS position spread around a reference point",
       py::arg("state"), py::arg("center_x"), py::arg("center_y"), py::arg("center_z"));

    m.def("pf_device_resample_systematic", [](gnss_gpu::PFDeviceState* state,
                                              unsigned long long seed) {
        gnss_gpu::pf_device_resample_systematic(state, seed);
    }, "Systematic resampling - operates entirely on device",
       py::arg("state"), py::arg("seed"));

    m.def("pf_device_resample_megopolis", [](gnss_gpu::PFDeviceState* state,
                                             int n_iterations,
                                             unsigned long long seed) {
        gnss_gpu::pf_device_resample_megopolis(state, n_iterations, seed);
    }, "Megopolis resampling - operates entirely on device",
       py::arg("state"), py::arg("n_iterations"), py::arg("seed"));

    m.def("pf_device_resample_megopolis_coalesced", [](gnss_gpu::PFDeviceState* state,
                                                       int n_iterations,
                                                       unsigned long long seed) {
        if (n_iterations < 0) {
            throw std::runtime_error("n_iterations must be >= 0");
        }
        gnss_gpu::pf_device_resample_megopolis_coalesced(state, n_iterations, seed);
    }, "Megopolis resampling with ancestor-weight acceptance and coalesced proposals",
       py::arg("state"), py::arg("n_iterations"), py::arg("seed"));

    m.def("pf_device_estimate", [](const gnss_gpu::PFDeviceState* state) {
        double result[4];
        gnss_gpu::pf_device_estimate(state, result);
        return py::array_t<double>({4}, {sizeof(double)}, result);
    }, "Compute weighted mean on device, return [4] to host",
       py::arg("state"));

    m.def("pf_device_position_mixture_components", [](const gnss_gpu::PFDeviceState* state,
        py::array_t<double, py::array::c_style | py::array::forcecast> refs,
        py::array_t<double, py::array::c_style | py::array::forcecast> log_mix,
        double sigma_squared) -> py::object {
        validate_device_state(state, "pf_device_position_mixture_components");
        auto r = refs.request(), w = log_mix.request();
        if (r.ndim != 2 || r.shape[1] != 3 || r.shape[0] < 2 || r.shape[0] > 4) {
            throw std::runtime_error("refs shape must be (2..4, 3)");
        }
        if (w.ndim != 1 || w.size != r.shape[0]) {
            throw std::runtime_error("log_mix length must match refs");
        }
        ensure_finite_doubles(r, "refs must be finite");
        const double* weights = static_cast<const double*>(w.ptr);
        for (py::ssize_t i = 0; i < w.size; ++i) {
            if (std::isnan(weights[i]) || weights[i] == INFINITY) {
                throw std::runtime_error("log_mix must be finite or negative infinity");
            }
        }
        validate_positive_finite(sigma_squared, "sigma_squared");
        py::array_t<double> output({state->n_particles, static_cast<int>(r.shape[0])});
        bool finite = gnss_gpu::pf_device_position_mixture_components(state,
            static_cast<const double*>(r.ptr), weights, static_cast<int>(r.shape[0]),
            sigma_squared, output.mutable_data());
        if (!finite) return py::none();
        return output;
    }, "Exact FP64 mixture components, or None for a nonfinite-coordinate CPU fallback",
       py::arg("state"), py::arg("refs"), py::arg("log_mix"), py::arg("sigma_squared"));

    m.def("pf_device_get_particles", [](const gnss_gpu::PFDeviceState* state) {
        int N = state->n_particles;
        auto output = py::array_t<double>({N, 4});
        gnss_gpu::pf_device_get_particles(state,
            static_cast<double*>(output.request().ptr));
        return output;
    }, "Copy particles to host for visualization (only when needed)",
       py::arg("state"));

    m.def("pf_device_get_particle_states", [](const gnss_gpu::PFDeviceState* state) {
        int N = state->n_particles;
        auto output = py::array_t<double>({N, 16});
        gnss_gpu::pf_device_get_particle_states(state,
            static_cast<double*>(output.request().ptr));
        return output;
    }, "Copy full particle states [x,y,z,cb,mu_vx,mu_vy,mu_vz,Sigma_v(3x3)] to host",
       py::arg("state"));

    m.def("pf_device_gather_from", [](gnss_gpu::PFDeviceState* target,
            const gnss_gpu::PFDeviceState* source,
            py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> ancestors) {
        validate_device_state(target, "pf_device_gather_from target");
        validate_device_state(source, "pf_device_gather_from source");
        auto r = ancestors.request();
        if (r.ndim != 1 || r.size != target->n_particles) {
            throw std::runtime_error("pf_device_gather_from: ancestors length must match target");
        }
        const auto* input = static_cast<const std::int64_t*>(r.ptr);
        std::vector<int> indices((size_t)r.size);
        for (py::ssize_t i = 0; i < r.size; ++i) {
            if (input[i] < 0 || input[i] >= source->n_particles) {
                throw std::runtime_error("pf_device_gather_from: ancestor outside source");
            }
            indices[(size_t)i] = static_cast<int>(input[i]);
        }
        return gnss_gpu::pf_device_gather_from(target, source, indices.data());
    }, "Gather full source states into a distinct resident target; reset weights; sync",
       py::arg("target"), py::arg("source"), py::arg("ancestors"));

    m.def("pf_device_gather_from_cdf", [](gnss_gpu::PFDeviceState* target,
            const gnss_gpu::PFDeviceState* source,
            py::array_t<double, py::array::c_style | py::array::forcecast> cdf) {
        validate_device_state(target, "pf_device_gather_from_cdf target");
        validate_device_state(source, "pf_device_gather_from_cdf source");
        auto r = cdf.request();
        if (r.ndim != 1 || r.size != source->n_particles) {
            throw std::runtime_error("pf_device_gather_from_cdf: CDF length must match source");
        }
        const auto* input = static_cast<const double*>(r.ptr);
        // Decline unsupported CDFs before any state or scratch mutation.
        if (!finite_monotone_cdf(input, r.size)) return false;
        return gnss_gpu::pf_device_gather_from_cdf(target, source, input);
    }, "Gather using exact left search on a finite monotone CPU CDF; reset weights; sync",
       py::arg("target"), py::arg("source"), py::arg("cdf"));

    m.attr("_uniform_cdf_checked_gather") = m.attr("pf_device_gather_from_cdf");
    m.def("pf_device_gather_from_uniform_cdf", [](gnss_gpu::PFDeviceState* target,
            const gnss_gpu::PFDeviceState* source, py::object cdf_output) {
        validate_device_state(target, "pf_device_gather_from_uniform_cdf target");
        validate_device_state(source, "pf_device_gather_from_uniform_cdf source");
        double* output = nullptr;
        py::array owner;
        if (!cdf_output.is_none()) {
            if (!py::isinstance<py::array>(cdf_output)) {
                throw std::runtime_error("CDF output must be a writable contiguous float64 array");
            }
            owner = py::cast<py::array>(cdf_output);
            auto r = owner.request();
            if (!owner.dtype().is(py::dtype::of<double>()) || !owner.writeable() ||
                !(owner.flags() & py::array::c_style) || r.ndim != 1 || r.size != source->n_particles) {
                throw std::runtime_error("CDF output must be a writable contiguous float64 array matching source");
            }
            output = static_cast<double*>(r.ptr);
        }
        return gnss_gpu::pf_device_gather_from_uniform_cdf(target, source, output);
    }, "Gather from the exact sequential uniform FP64 CDF; optional full CDF output; sync",
       py::arg("target"), py::arg("source"), py::arg("cdf_output") = py::none());

    m.def("pf_device_set_particle_states", [](gnss_gpu::PFDeviceState* state,
                                              py::array_t<double, py::array::c_style | py::array::forcecast> input) {
        int N = state->n_particles;
        auto r = input.request();
        if (r.ndim != 2 || r.shape[0] != N || r.shape[1] != 16) {
            throw std::runtime_error("pf_device_set_particle_states: input shape must be (n_particles, 16)");
        }
        gnss_gpu::pf_device_set_particle_states(state, static_cast<const double*>(r.ptr));
    }, "Upload full particle states [x,y,z,cb,mu_vx,mu_vy,mu_vz,Sigma_v(3x3)] and reset weights",
       py::arg("state"), py::arg("input"));

    m.def("pf_device_set_velocity_covariance", [](gnss_gpu::PFDeviceState* state,
            py::array_t<double, py::array::c_style | py::array::forcecast> covariance) {
        validate_device_state(state, "pf_device_set_velocity_covariance");
        auto r = covariance.request();
        if (r.ndim != 2 || r.shape[0] != 3 || r.shape[1] != 3) {
            throw std::runtime_error("pf_device_set_velocity_covariance: covariance shape must be (3, 3)");
        }
        ensure_finite_doubles(r, "pf_device_set_velocity_covariance: covariance must be finite");
        gnss_gpu::pf_device_set_velocity_covariance(state, static_cast<const double*>(r.ptr));
    }, "Broadcast velocity covariance on device without changing other state or weights",
       py::arg("state"), py::arg("covariance"));

    m.def("pf_device_get_log_weights", [](const gnss_gpu::PFDeviceState* state,
                                         py::array_t<double> out) {
        int N = state->n_particles;
        auto r = out.request();
        if (r.size != N) {
            throw std::runtime_error("pf_device_get_log_weights: output size must match n_particles");
        }
        gnss_gpu::pf_device_get_log_weights(state, static_cast<double*>(r.ptr));
    }, "Copy log-weights to host (D2H, synchronizes stream)",
       py::arg("state"), py::arg("out"));

    m.attr("_uniform_scan_getter") = m.attr("pf_device_get_log_weights");
    m.attr("_uniform_scan_cdf") = m.attr("pf_device_gather_from_uniform_cdf");
    m.attr("_uniform_scan_state_type") = m.attr("PFDeviceState");
    m.def("pf_device_try_gather_uniform_weights", [](gnss_gpu::PFDeviceState* target,
                                                    const gnss_gpu::PFDeviceState* source) {
        return gnss_gpu::pf_device_try_gather_uniform_weights(target, source);
    }, "Try exact full GPU finite/equal-weight validation and D199 gather; false leaves visible state unchanged",
       py::arg("target"), py::arg("source"));

    m.def("pf_device_set_log_weights", [](gnss_gpu::PFDeviceState* state,
                                          py::array_t<double, py::array::c_style | py::array::forcecast> input) {
        int N = state->n_particles;
        auto r = input.request();
        if (r.size != N) {
            throw std::runtime_error("pf_device_set_log_weights: input size must match n_particles");
        }
        gnss_gpu::pf_device_set_log_weights(state, static_cast<const double*>(r.ptr));
    }, "Upload log-weights from host (H2D, sync)",
       py::arg("state"), py::arg("input"));

    // Keep the original array setter identity even when callers replace it.
    m.attr("_uniform_weights_array_setter") = m.attr("pf_device_set_log_weights");
    m.def("pf_device_fill_log_weights", [](gnss_gpu::PFDeviceState* state, double value) {
        validate_device_state(state, "pf_device_fill_log_weights");
        gnss_gpu::pf_device_fill_log_weights(state, value);
    }, "Broadcast exact FP64 log-weight bits; invalidate ESS; synchronize stream",
       py::arg("state"), py::arg("value"));

    m.def("pf_device_get_resample_ancestors", [](const gnss_gpu::PFDeviceState* state,
                                                  py::array_t<int> out) {
        int N = state->n_particles;
        auto r = out.request();
        if (r.size != N) {
            throw std::runtime_error(
                "pf_device_get_resample_ancestors: output size must match n_particles");
        }
        gnss_gpu::pf_device_get_resample_ancestors(state, static_cast<int*>(r.ptr));
    }, "After systematic resample: copy ancestor indices out[j]=source (D2H, sync)",
       py::arg("state"), py::arg("out"));

    m.def("pf_device_sync", [](gnss_gpu::PFDeviceState* state) {
        gnss_gpu::pf_device_sync(state);
    }, "Synchronize CUDA stream - wait for all pending operations to complete",
       py::arg("state"));
}
