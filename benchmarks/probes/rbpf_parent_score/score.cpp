// Standalone CPython 3.12 experiment. The GIL stays held throughout.
#include <Python.h>

namespace {
struct Owned {
    PyObject* value;
    explicit Owned(PyObject* object) : value(object) {}
    ~Owned() { Py_XDECREF(value); }
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
};

bool plain_hypothesis(PyObject* h) {
    if (!PyDict_CheckExact(h)) return false;
    Py_ssize_t position = 0, total = 0;
    PyObject *group, *integers;
    while (PyDict_Next(h, &position, &group, &integers)) {
        if (!PyTuple_CheckExact(group) || PyTuple_GET_SIZE(group) != 2
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 1))
            || !PyDict_CheckExact(integers)) return false;
        const Py_ssize_t count = PyDict_GET_SIZE(integers);
        // Each compared DD consumes a child entry; guard the count, never the integers.
        if (count > PY_SSIZE_T_MAX - total) return false;
        total += count;
        Py_ssize_t inner = 0;
        PyObject *satellite, *integer;
        while (PyDict_Next(integers, &inner, &satellite, &integer)) {
            if (!PyLong_CheckExact(satellite) || !PyLong_CheckExact(integer)) return false;
        }
    }
    return true;
}

// None means unsupported input. A successful result is (agree, compared).
// Preflight both complete maps before arithmetic, with no retained references.
PyObject* score(PyObject*, PyObject* args) {
    PyObject *child, *parent;
    if (!PyArg_ParseTuple(args, "OO:score", &child, &parent)) return nullptr;
    if (!plain_hypothesis(child) || !plain_hypothesis(parent)) Py_RETURN_NONE;
    Py_ssize_t agree = 0, compared = 0, position = 0;
    PyObject *group, *child_map;
    while (PyDict_Next(child, &position, &group, &child_map)) {
        PyObject* parent_map = PyDict_GetItemWithError(parent, group);
        if (!parent_map) {
            if (PyErr_Occurred()) return nullptr;
            continue;
        }
        Py_ssize_t inner = 0, common = 0;
        PyObject *satellite, *integer, *reference = nullptr;
        PyObject *child_base = nullptr, *parent_base = nullptr;
        while (PyDict_Next(child_map, &inner, &satellite, &integer)) {
            PyObject* parent_integer = PyDict_GetItemWithError(parent_map, satellite);
            if (!parent_integer) {
                if (PyErr_Occurred()) return nullptr;
                continue;
            }
            ++common;
            const int less = reference ? PyObject_RichCompareBool(satellite, reference, Py_LT) : 1;
            if (less < 0) return nullptr;
            if (less) { reference = satellite; child_base = integer; parent_base = parent_integer; }
        }
        if (common < 2) continue;
        inner = 0;
        while (PyDict_Next(child_map, &inner, &satellite, &integer)) {
            // The reference is the same borrowed key selected in the preceding scan.
            if (satellite == reference) continue;
            PyObject* parent_integer = PyDict_GetItemWithError(parent_map, satellite);
            if (!parent_integer) {
                if (PyErr_Occurred()) return nullptr;
                continue;
            }
            Owned child_dd(PyNumber_Subtract(child_base, integer));
            if (!child_dd.value) return nullptr;
            Owned parent_dd(PyNumber_Subtract(parent_base, parent_integer));
            if (!parent_dd.value) return nullptr;
            const int equal = PyObject_RichCompareBool(child_dd.value, parent_dd.value, Py_EQ);
            if (equal < 0) return nullptr;
            ++compared;
            agree += equal;
        }
    }
    return Py_BuildValue("(nn)", agree, compared);
}

PyMethodDef methods[] = {
    {"score", score, METH_VARARGS, "Exact builtin-integer parent overlap, or None for unsupported inputs."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_parent_score_probe", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__rbpf_parent_score_probe() { return PyModule_Create(&module); }
