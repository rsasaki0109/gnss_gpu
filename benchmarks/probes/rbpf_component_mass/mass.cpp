// Standalone exact DD expansion and ordered component agreement mass.
// CPython holds the GIL; no parallel/reassociated reductions or FP32.
#include <Python.h>
#include <cmath>
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
    PyObject* release() { PyObject* result = value; value = nullptr; return result; }
};

struct Buffer {
    Py_buffer view{};
    ~Buffer() { if (view.obj) PyBuffer_Release(&view); }
    bool acquire(PyObject* object) {
        return PyObject_GetBuffer(object, &view, PyBUF_FULL_RO) == 0;
    }
    bool doubles(Py_ssize_t count) const {
        return view.ndim == 1 && view.itemsize == sizeof(double) && view.shape
            && view.shape[0] == count && view.format && std::strcmp(view.format, "d") == 0
            && PyBuffer_IsContiguous(&view, 'C');
    }
    // memcpy also supports a valid contiguous buffer with an unaligned base.
    double at(Py_ssize_t index) const {
        double value;
        std::memcpy(&value, static_cast<const char*>(view.buf) + index * sizeof(double), sizeof(double));
        return value;
    }
};

bool group_key(PyObject* group) {
    return PyTuple_CheckExact(group) && PyTuple_GET_SIZE(group) == 2
        && PyLong_CheckExact(PyTuple_GET_ITEM(group, 0))
        && PyLong_CheckExact(PyTuple_GET_ITEM(group, 1));
}

bool plain_hypothesis(PyObject* h) {
    if (!PyDict_CheckExact(h)) return false;
    Py_ssize_t position = 0;
    PyObject *group, *integers;
    while (PyDict_Next(h, &position, &group, &integers)) {
        if (!group_key(group) || !PyDict_CheckExact(integers)) return false;
        Py_ssize_t inner = 0;
        PyObject *satellite, *integer;
        while (PyDict_Next(integers, &inner, &satellite, &integer)) {
            if (!PyLong_CheckExact(satellite) || !PyLong_CheckExact(integer)) return false;
        }
    }
    return true;
}

bool plain_inputs(PyObject* hypotheses, PyObject* pairs) {
    if (!PyList_CheckExact(hypotheses) || !PyList_CheckExact(pairs)) return false;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(hypotheses); ++i)
        if (!plain_hypothesis(PyList_GET_ITEM(hypotheses, i))) return false;
    for (Py_ssize_t j = 0; j < PyList_GET_SIZE(pairs); ++j) {
        PyObject* pair = PyList_GET_ITEM(pairs, j);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 3
            || !group_key(PyTuple_GET_ITEM(pair, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 1))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 2))) return false;
    }
    return true;
}

// Return a fresh builtin dict in increasing finite-target pair order.
// None is an unsupported-input sentinel. No input/buffer survives the call.
PyObject* component_mass(PyObject*, PyObject* args) {
    PyObject *hypotheses, *pairs, *target, *weights;
    if (!PyArg_ParseTuple(args, "OOOO:component_mass", &hypotheses, &pairs, &target, &weights)) return nullptr;
    if (!plain_inputs(hypotheses, pairs)) Py_RETURN_NONE;
    const Py_ssize_t n = PyList_GET_SIZE(hypotheses), nb = PyList_GET_SIZE(pairs);
    if (!PyObject_CheckBuffer(target) || !PyObject_CheckBuffer(weights)) Py_RETURN_NONE;
    Buffer reference, w;
    if (!reference.acquire(target) || !w.acquire(weights)) return nullptr;
    if (!reference.doubles(nb) || !w.doubles(n)) Py_RETURN_NONE;
    for (Py_ssize_t i = 0; i < n; ++i) {
        const double value = w.at(i);
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) Py_RETURN_NONE;
    }
    if (nb && static_cast<size_t>(n) > std::numeric_limits<size_t>::max() / sizeof(double) / static_cast<size_t>(nb))
        Py_RETURN_NONE;
    std::vector<double> dd;
    try {
        dd.assign(static_cast<size_t>(n) * static_cast<size_t>(nb), std::numeric_limits<double>::quiet_NaN());
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error&) {
        return PyErr_NoMemory();
    }
    // Exactly the original particle-major/pair-major expansion, including
    // non-target pairs: their conversion exceptions must not be skipped.
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* h = PyList_GET_ITEM(hypotheses, i);
        for (Py_ssize_t j = 0; j < nb; ++j) {
            PyObject* pair = PyList_GET_ITEM(pairs, j);
            PyObject* zmap = PyDict_GetItemWithError(h, PyTuple_GET_ITEM(pair, 0));
            if (!zmap) { if (PyErr_Occurred()) return nullptr; continue; }
            PyObject* zr = PyDict_GetItemWithError(zmap, PyTuple_GET_ITEM(pair, 1));
            if (!zr) { if (PyErr_Occurred()) return nullptr; continue; }
            PyObject* zj = PyDict_GetItemWithError(zmap, PyTuple_GET_ITEM(pair, 2));
            if (!zj) { if (PyErr_Occurred()) return nullptr; continue; }
            Owned difference(PyNumber_Subtract(zr, zj));
            if (!difference.value) return nullptr;
            const double value = PyLong_AsDouble(difference.value);
            if (PyErr_Occurred()) return nullptr;
            dd[static_cast<size_t>(i) * nb + j] = value;
        }
    }
    Owned output(PyDict_New());
    if (!output.value) return nullptr;
    for (Py_ssize_t j = 0; j < nb; ++j) {
        const double ref = reference.at(j);
        if (!std::isfinite(ref)) continue;
        // Python int(finite double) and trunc have the same integer value.
        // Every such truncated binary64 integer remains exactly representable
        // in binary64, even beyond int64. +/-0 both compare as Python int(0).
        const double wanted = std::trunc(ref);
        double mass = 0.0;
        for (Py_ssize_t i = 0; i < n; ++i) {
            const double value = dd[static_cast<size_t>(i) * nb + j];
            if (std::isfinite(value) && std::trunc(value) == wanted) mass += w.at(i);
        }
        Owned index(PyLong_FromSsize_t(j)), total(PyFloat_FromDouble(mass));
        if (!index.value || !total.value || PyDict_SetItem(output.value, index.value, total.value) < 0) return nullptr;
    }
    return output.release();
}

PyMethodDef methods[] = {
    {"component_mass", component_mass, METH_VARARGS, "Exact DD expansion and ordered agreement masses, or None."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_component_mass_probe", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__rbpf_component_mass_probe() { return PyModule_Create(&module); }
