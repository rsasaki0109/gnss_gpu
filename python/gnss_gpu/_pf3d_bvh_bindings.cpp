#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "gnss_gpu/pf_3d_bvh.h"
#include <vector>

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void ensure_finite_doubles(const py::buffer_info& buf, const char* message) {
    const auto* ptr = static_cast<const double*>(buf.ptr);
    for (py::ssize_t i = 0; i < buf.size; ++i) {
        if (!std::isfinite(ptr[i])) {
            throw std::runtime_error(message);
        }
    }
}

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

void validate_particle_arrays(const py::buffer_info& px, const py::buffer_info& py_arr,
                              const py::buffer_info& pz, const py::buffer_info& pcb,
                              int n_particles) {
    if (px.ndim != 1 || px.size != n_particles) {
        throw std::runtime_error("px length must match n_particles");
    }
    if (py_arr.ndim != 1 || py_arr.size != n_particles) {
        throw std::runtime_error("py length must match n_particles");
    }
    if (pz.ndim != 1 || pz.size != n_particles) {
        throw std::runtime_error("pz length must match n_particles");
    }
    if (pcb.ndim != 1 || pcb.size != n_particles) {
        throw std::runtime_error("pcb length must match n_particles");
    }
}

void validate_log_weights(const py::buffer_info& buf, int n_particles) {
    if (buf.ndim != 1 || buf.size != n_particles) {
        throw std::runtime_error("log_weights length must match n_particles");
    }
    ensure_finite_doubles(buf, "log_weights must be finite");
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

void validate_triangles(const py::buffer_info& buf) {
    if (buf.ndim != 3 || buf.shape[1] != 3 || buf.shape[2] != 3) {
        throw std::runtime_error("sorted_tris must have shape (n_tri, 3, 3)");
    }
    if (buf.shape[0] > 0) {
        ensure_finite_doubles(buf, "sorted_tris must be finite");
    }
}

void validate_nodes_flat(const py::buffer_info& buf, int n_tri) {
    if (buf.ndim != 2 || buf.shape[1] != 10) {
        throw std::runtime_error("nodes_flat must have shape (n_nodes, 10)");
    }
    const int n_nodes = static_cast<int>(buf.shape[0]);
    if (n_tri > 0 && n_nodes < 1) {
        throw std::runtime_error(
            "nodes_flat must contain at least one node when sorted_tris is non-empty");
    }
    if (n_nodes > 0) {
        ensure_finite_doubles(buf, "nodes_flat must be finite");
    }
}

}  // namespace

PYBIND11_MODULE(_gnss_gpu_pf3d_bvh, m) {
  m.doc() = "GPU-accelerated BVH-accelerated 3D-aware particle filter weight computation";

  m.def("pf_weight_3d_bvh",
    [](DoubleArray px, DoubleArray py_arr,
       DoubleArray pz, DoubleArray pcb,
       DoubleArray sat_ecef,
       DoubleArray pseudoranges,
       DoubleArray weights_sat,
       DoubleArray nodes_flat,
       DoubleArray sorted_tris,
       DoubleArray log_weights,
       int n_particles, int n_sat,
       double sigma_pr_los, double sigma_pr_nlos,
       double nlos_bias,
       double blocked_nlos_prob,
       double clear_nlos_prob) {

      validate_n_particles(n_particles);
      validate_n_sat(n_sat);

      auto bpx = px.request();
      auto bpy = py_arr.request();
      auto bpz = pz.request();
      auto bpcb = pcb.request();
      validate_particle_arrays(bpx, bpy, bpz, bpcb, n_particles);

      auto bs = sat_ecef.request();
      validate_sat_ecef(bs, n_sat);
      validate_pseudoranges(pseudoranges.request(), n_sat);
      validate_weights_sat(weights_sat.request(), n_sat);
      validate_log_weights(log_weights.request(), n_particles);

      auto btri = sorted_tris.request();
      validate_triangles(btri);
      const int n_tri = static_cast<int>(btri.shape[0]);

      auto bnodes = nodes_flat.request();
      validate_nodes_flat(bnodes, n_tri);

      validate_positive_finite(sigma_pr_los, "sigma_pr_los");
      validate_positive_finite(sigma_pr_nlos, "sigma_pr_nlos");
      validate_finite(nlos_bias, "nlos_bias");
      validate_finite(blocked_nlos_prob, "blocked_nlos_prob");
      validate_finite(clear_nlos_prob, "clear_nlos_prob");

      int n_nodes = static_cast<int>(bnodes.shape[0]);

      const double* nptr = static_cast<const double*>(bnodes.ptr);
      std::vector<gnss_gpu::BVHNode> nodes(n_nodes);
      for (int i = 0; i < n_nodes; i++) {
        nodes[i].bbox.min[0] = nptr[i * 10 + 0];
        nodes[i].bbox.min[1] = nptr[i * 10 + 1];
        nodes[i].bbox.min[2] = nptr[i * 10 + 2];
        nodes[i].bbox.max[0] = nptr[i * 10 + 3];
        nodes[i].bbox.max[1] = nptr[i * 10 + 4];
        nodes[i].bbox.max[2] = nptr[i * 10 + 5];
        nodes[i].left      = static_cast<int>(nptr[i * 10 + 6]);
        nodes[i].right     = static_cast<int>(nptr[i * 10 + 7]);
        nodes[i].tri_start = static_cast<int>(nptr[i * 10 + 8]);
        nodes[i].tri_count = static_cast<int>(nptr[i * 10 + 9]);
      }

      gnss_gpu::pf_weight_3d_bvh(
          static_cast<double*>(bpx.ptr),
          static_cast<double*>(bpy.ptr),
          static_cast<double*>(bpz.ptr),
          static_cast<double*>(bpcb.ptr),
          static_cast<double*>(bs.ptr),
          static_cast<double*>(pseudoranges.request().ptr),
          static_cast<double*>(weights_sat.request().ptr),
          nodes.data(), n_nodes,
          reinterpret_cast<const gnss_gpu::Triangle*>(btri.ptr), n_tri,
          static_cast<double*>(log_weights.request().ptr),
          n_particles, n_sat,
          sigma_pr_los, sigma_pr_nlos, nlos_bias,
          blocked_nlos_prob, clear_nlos_prob);
    },
    "Compute 3D-aware pseudorange likelihood weights using BVH-accelerated ray tracing",
    py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("pcb"),
    py::arg("sat_ecef"), py::arg("pseudoranges"),
    py::arg("weights_sat"),
    py::arg("nodes_flat"), py::arg("sorted_tris"),
    py::arg("log_weights"),
    py::arg("n_particles"), py::arg("n_sat"),
    py::arg("sigma_pr_los"), py::arg("sigma_pr_nlos"),
    py::arg("nlos_bias"),
    py::arg("blocked_nlos_prob") = 1.0,
    py::arg("clear_nlos_prob") = 0.0);
}
