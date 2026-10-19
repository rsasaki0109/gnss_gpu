// Frozen RBPF release-subset move: exact DD expansion and ordered worst row.
// The GIL is held throughout. No cache, parallel reduction, FMA or FP32.
#include <Python.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
struct Owned {
    PyObject* value;
    explicit Owned(PyObject* object) : value(object) {}
    ~Owned() { Py_XDECREF(value); }
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
};
struct Buffer {
    Py_buffer view{};
    ~Buffer() { if (view.obj) PyBuffer_Release(&view); }
    bool acquire(PyObject* object) {
        return PyObject_GetBuffer(object, &view, PyBUF_FULL_RO) == 0;
    }
    bool vector(Py_ssize_t count, const char* format, Py_ssize_t size) const {
        return view.ndim == 1 && view.itemsize == size && view.shape
            && view.shape[0] == count && view.format && std::strcmp(view.format, format) == 0
            && PyBuffer_IsContiguous(&view, 'C');
    }
    template<class T> T at(Py_ssize_t i) const {
        T result;
        std::memcpy(&result, static_cast<const char*>(view.buf) + i * sizeof(T), sizeof(T));
        return result;
    }
    bool integer(Py_ssize_t count) const {
        return vector(count, "q", 8) || vector(count, "l", sizeof(long)) || vector(count, "i", 4);
    }
    int64_t index(Py_ssize_t i) const {
        return view.itemsize == 8 ? at<int64_t>(i) : at<int32_t>(i);
    }
};
bool group_key(PyObject* group) {
    return PyTuple_CheckExact(group) && PyTuple_GET_SIZE(group) == 2
        && PyLong_CheckExact(PyTuple_GET_ITEM(group, 0))
        && PyLong_CheckExact(PyTuple_GET_ITEM(group, 1));
}
bool plain(PyObject* h, PyObject* pairs) {
    if (!PyDict_CheckExact(h) || !PyList_CheckExact(pairs)) return false;
    Py_ssize_t outer = 0;
    PyObject *g, *zm;
    while (PyDict_Next(h, &outer, &g, &zm)) {
        if (!group_key(g) || !PyDict_CheckExact(zm)) return false;
        Py_ssize_t inner = 0;
        PyObject *sat, *z;
        while (PyDict_Next(zm, &inner, &sat, &z))
            if (!PyLong_CheckExact(sat) || !PyLong_CheckExact(z)) return false;
    }
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pairs); ++i) {
        PyObject* pair = PyList_GET_ITEM(pairs, i);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 3
            || !group_key(PyTuple_GET_ITEM(pair, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 1))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 2))) return false;
    }
    return true;
}
bool modest(double value) { return std::isfinite(value) && std::fabs(value) <= 1e100; }

// None means unsupported: the Python caller executes its original block.
// -1 is a successful original result: no eligible CP row. No input survives.
PyObject* worst_index(PyObject*, PyObject* args) {
    PyObject *h, *pairs, *v0, *cp, *pi, *lam, *bmeas, *sig;
    if (!PyArg_ParseTuple(args, "OOOOOOOO:worst_index", &h, &pairs, &v0, &cp, &pi, &lam, &bmeas, &sig)) return nullptr;
    if (!plain(h, pairs)) Py_RETURN_NONE;
    PyObject* objects[] = {v0, cp, pi, lam, bmeas, sig};
    for (PyObject* object : objects) if (!PyObject_CheckBuffer(object)) Py_RETURN_NONE;
    Buffer bv, bc, bi, bl, bb, bs;
    if (!bv.acquire(v0) || !bc.acquire(cp) || !bi.acquire(pi)
        || !bl.acquire(lam) || !bb.acquire(bmeas) || !bs.acquire(sig)) return nullptr;
    if (bv.view.ndim != 1 || !bv.view.shape) Py_RETURN_NONE;
    const Py_ssize_t nr = bv.view.shape[0], nb = PyList_GET_SIZE(pairs);
    if (!bv.vector(nr, "d", 8) || !bc.vector(nr, "?", 1) || !bi.integer(nr)
        || !bl.vector(nr, "d", 8) || !bb.vector(nr, "d", 8) || !bs.vector(nr, "d", 8)) Py_RETURN_NONE;
    // Safety domain, not a model limit. Outside it the original NumPy code
    // retains negative indexing, warnings, exceptions and partial state.
    for (Py_ssize_t r = 0; r < nr; ++r) {
        if (!bc.at<unsigned char>(r)) continue;
        const int64_t index = bi.index(r);
        if (index < 0 || index >= nb || !modest(bl.at<double>(r))
            || !modest(bb.at<double>(r)) || !modest(bs.at<double>(r))) Py_RETURN_NONE;
    }
    std::vector<double> dd;
    try { dd.assign(nb, std::numeric_limits<double>::quiet_NaN()); }
    catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::length_error&) { return PyErr_NoMemory(); }
    bool safe = true;
    // Expand EVERY pair before scanning rows, including PR-only/non-target
    // pairs. Subtract arbitrary-precision PyLong values before conversion.
    for (Py_ssize_t i = 0; i < nb; ++i) {
        PyObject* pair = PyList_GET_ITEM(pairs, i);
        PyObject* zm = PyDict_GetItemWithError(h, PyTuple_GET_ITEM(pair, 0));
        if (!zm) { if (PyErr_Occurred()) return nullptr; continue; }
        PyObject* zr = PyDict_GetItemWithError(zm, PyTuple_GET_ITEM(pair, 1));
        if (!zr) { if (PyErr_Occurred()) return nullptr; continue; }
        PyObject* zj = PyDict_GetItemWithError(zm, PyTuple_GET_ITEM(pair, 2));
        if (!zj) { if (PyErr_Occurred()) return nullptr; continue; }
        Owned difference(PyNumber_Subtract(zr, zj));
        if (!difference.value) return nullptr;
        const double value = PyLong_AsDouble(difference.value);
        if (PyErr_Occurred()) return nullptr;
        dd[i] = value;
        safe = safe && modest(value);
    }
    if (!safe) Py_RETURN_NONE;
    Py_ssize_t worst_pi = -1;
    double worst_r = -1.0;
    for (Py_ssize_t r = 0; r < nr; ++r) {
        if (!bc.at<unsigned char>(r)) continue;
        const Py_ssize_t index = static_cast<Py_ssize_t>(bi.index(r));
        if (!std::isfinite(dd[index])) continue;
        // Literal operation order: subtract, multiply, abs, max, divide.
        const double residual = bb.at<double>(r) - dd[index];
        const double product = bl.at<double>(r) * residual;
        const double sigma = bs.at<double>(r);
        const double denominator = sigma > 1e-6 ? sigma : 1e-6;
        const double wr = std::fabs(product) / denominator;
        if (wr > worst_r) { worst_r = wr; worst_pi = index; }
    }
    return PyLong_FromSsize_t(worst_pi);
}
PyMethodDef methods[] = {
    {"worst_index", worst_index, METH_VARARGS, "Exact ordered worst CP pair index, or None for fallback."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_shrink_probe", nullptr, -1, methods};
}
PyMODINIT_FUNC PyInit__rbpf_shrink_probe() { return PyModule_Create(&module); }
