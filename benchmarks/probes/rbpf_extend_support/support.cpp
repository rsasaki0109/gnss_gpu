// CPU-only, GIL-held support/assembly. No native references survive a call.
#include <Python.h>
#include <cmath>
#include <cstring>

namespace {
struct Owned {
    PyObject* value;
    explicit Owned(PyObject* v) : value(v) {}
    ~Owned() { Py_XDECREF(value); }
    PyObject* release() { PyObject* v = value; value = nullptr; return v; }
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
};
struct Buffer {
    Py_buffer view{};
    bool active = false;
    ~Buffer() { if (active) PyBuffer_Release(&view); }
    bool acquire(PyObject* object, bool writable) {
        if (PyObject_GetBuffer(object, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES
            | (writable ? PyBUF_WRITABLE : 0)) < 0) return false;
        active = true;
        return view.ndim == 1 && view.itemsize == sizeof(double) && view.format
            && std::strcmp(view.format, "d") == 0 && PyBuffer_IsContiguous(&view, 'C')
            && reinterpret_cast<size_t>(view.buf) % alignof(double) == 0;
    }
};

bool plain_h(PyObject* h) {
    if (!PyDict_CheckExact(h)) return false;
    Py_ssize_t p = 0;
    PyObject *g, *z;
    while (PyDict_Next(h, &p, &g, &z)) {
        if (!PyTuple_CheckExact(g) || PyTuple_GET_SIZE(g) != 2
            || !PyLong_CheckExact(PyTuple_GET_ITEM(g, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(g, 1)) || !PyDict_CheckExact(z)) return false;
        Py_ssize_t q = 0;
        PyObject *s, *v;
        while (PyDict_Next(z, &q, &s, &v))
            if (!PyLong_CheckExact(s) || !PyLong_CheckExact(v)) return false;
    }
    return true;
}

bool plain_pairs(PyObject* pairs) {
    if (!PyList_CheckExact(pairs)) return false;
    for (Py_ssize_t i = 0; i < PyList_GET_SIZE(pairs); ++i) {
        PyObject* pair = PyList_GET_ITEM(pairs, i);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 3) return false;
        PyObject* group = PyTuple_GET_ITEM(pair, 0);
        if (!PyTuple_CheckExact(group) || PyTuple_GET_SIZE(group) != 2
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 1))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 1))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 2))) return false;
    }
    return true;
}

PyObject* support(PyObject*, PyObject* args) {
    PyObject *h, *pairs, *dd;
    if (!PyArg_ParseTuple(args, "OOO", &h, &pairs, &dd)) return nullptr;
    if (!plain_h(h) || !plain_pairs(pairs)) Py_RETURN_NONE;
    Buffer out;
    if (!out.acquire(dd, true)) {
        if (PyErr_Occurred()) return nullptr;
        Py_RETURN_NONE;
    }
    const Py_ssize_t n = PyList_GET_SIZE(pairs);
    if (out.view.shape[0] != n) Py_RETURN_NONE;
    Owned held(PyList_New(0)), missing(PyList_New(0));
    if (!held.value || !missing.value) return nullptr;
    double* values = static_cast<double*>(out.view.buf);
    // Every pair is expanded in original order, including conversion errors.
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* pair = PyList_GET_ITEM(pairs, i);
        PyObject* z = PyDict_GetItemWithError(h, PyTuple_GET_ITEM(pair, 0));
        if (!z) { if (PyErr_Occurred()) return nullptr; continue; }
        PyObject* zr = PyDict_GetItemWithError(z, PyTuple_GET_ITEM(pair, 1));
        if (!zr && PyErr_Occurred()) return nullptr;
        PyObject* zj = PyDict_GetItemWithError(z, PyTuple_GET_ITEM(pair, 2));
        if (!zj && PyErr_Occurred()) return nullptr;
        if (zr && zj) {
            Owned difference(PyNumber_Subtract(zr, zj));
            if (!difference.value) return nullptr;
            const double value = PyLong_AsDouble(difference.value);
            if (PyErr_Occurred()) return nullptr;
            values[i] = value;
            Owned index(PyLong_FromSsize_t(i));
            if (!index.value || PyList_Append(held.value, index.value) < 0) return nullptr;
        } else if (zr) {
            Owned index(PyLong_FromSsize_t(i));
            if (!index.value || PyList_Append(missing.value, index.value) < 0) return nullptr;
        }
    }
    return PyTuple_Pack(2, held.value, missing.value);
}

PyObject* finish(PyObject*, PyObject* args) {
    PyObject *h, *pairs, *indices, *conditional, *rint;
    if (!PyArg_ParseTuple(args, "OOOOO", &h, &pairs, &indices, &conditional, &rint)) return nullptr;
    if (!plain_h(h) || !plain_pairs(pairs) || !PyList_CheckExact(indices)) Py_RETURN_NONE;
    Buffer input;
    if (!input.acquire(conditional, false)) {
        if (PyErr_Occurred()) return nullptr;
        Py_RETURN_NONE;
    }
    const Py_ssize_t n = PyList_GET_SIZE(indices);
    if (input.view.shape[0] != n) Py_RETURN_NONE;
    const double* values = static_cast<const double*>(input.view.buf);
    // Nonfinite values retain original scalar warning/conversion order.
    for (Py_ssize_t k = 0; k < n; ++k) {
        PyObject* index = PyList_GET_ITEM(indices, k);
        if (!PyLong_CheckExact(index)) Py_RETURN_NONE;
        const Py_ssize_t i = PyLong_AsSsize_t(index);
        if (PyErr_Occurred()) { PyErr_Clear(); Py_RETURN_NONE; }
        if (i < 0 || i >= PyList_GET_SIZE(pairs) || !std::isfinite(values[k])) Py_RETURN_NONE;
        PyObject* pair = PyList_GET_ITEM(pairs, i);
        PyObject* z = PyDict_GetItemWithError(h, PyTuple_GET_ITEM(pair, 0));
        if (!z) { if (PyErr_Occurred()) return nullptr; Py_RETURN_NONE; }
        if (!PyDict_GetItemWithError(z, PyTuple_GET_ITEM(pair, 1))) {
            if (PyErr_Occurred()) return nullptr;
            Py_RETURN_NONE;
        }
    }
    // Original NumPy ufunc, no C++ floating rounding/conversion approximation.
    Owned rounded(PyObject_CallOneArg(rint, conditional));
    if (!rounded.value) return nullptr;
    Buffer rounded_buffer;
    if (!rounded_buffer.acquire(rounded.value, false)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_RuntimeError, "Unexpected rint buffer");
        return nullptr;
    }
    if (rounded_buffer.view.shape[0] != n) {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected rint length");
        return nullptr;
    }
    Owned result(PyDict_New());
    if (!result.value) return nullptr;
    Py_ssize_t p = 0;
    PyObject *group, *z;
    while (PyDict_Next(h, &p, &group, &z)) {
        Owned copy(PyDict_Copy(z));
        if (!copy.value || PyDict_SetItem(result.value, group, copy.value) < 0) return nullptr;
    }
    const double* rounded_values = static_cast<const double*>(rounded_buffer.view.buf);
    for (Py_ssize_t k = 0; k < n; ++k) {
        const Py_ssize_t i = PyLong_AsSsize_t(PyList_GET_ITEM(indices, k));
        PyObject* pair = PyList_GET_ITEM(pairs, i);
        PyObject* map = PyDict_GetItemWithError(result.value, PyTuple_GET_ITEM(pair, 0));
        if (!map) return nullptr;
        PyObject* zr = PyDict_GetItemWithError(map, PyTuple_GET_ITEM(pair, 1));
        if (!zr) return nullptr;
        Owned integer(PyLong_FromDouble(rounded_values[k]));
        if (!integer.value) return nullptr;
        Owned difference(PyNumber_Subtract(zr, integer.value));
        if (!difference.value || PyDict_SetItem(map, PyTuple_GET_ITEM(pair, 2), difference.value) < 0) return nullptr;
    }
    return result.release();
}

PyMethodDef methods[] = {
    {"support", support, METH_VARARGS, "Expand original ordered DD and support indices."},
    {"finish", finish, METH_VARARGS, "Assemble an independent ordered integer hypothesis."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_extend_support_probe", nullptr, -1, methods};
}  // namespace
PyMODINIT_FUNC PyInit__rbpf_extend_support_probe() { return PyModule_Create(&module); }
