// Standalone CPython 3.12 experiment. METH_O runs with the GIL held.
#include <Python.h>

namespace {
struct Owned {
    PyObject* value;
    explicit Owned(PyObject* object) : value(object) {}
    ~Owned() { Py_XDECREF(value); }
    Owned(const Owned&) = delete;
    Owned& operator=(const Owned&) = delete;
    PyObject* release() { PyObject* result = value; value = nullptr; return result; }
};

bool plain_hypothesis(PyObject* h) {
    if (!PyDict_CheckExact(h)) return false;
    Py_ssize_t position = 0;
    PyObject *group, *integers;
    while (PyDict_Next(h, &position, &group, &integers)) {
        if (!PyTuple_CheckExact(group) || PyTuple_GET_SIZE(group) != 2
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 1))
            || !PyDict_CheckExact(integers)) return false;
        Py_ssize_t inner = 0;
        PyObject *satellite, *integer;
        while (PyDict_Next(integers, &inner, &satellite, &integer)) {
            if (!PyLong_CheckExact(satellite) || !PyLong_CheckExact(integer)) return false;
        }
    }
    return true;
}

// None is an unsupported-input sentinel. A successful key is always a tuple.
// Validate every key/value before arithmetic. No Python references or work
// buffers survive this call, and no fixed-width or floating conversion occurs.
PyObject* canonical(PyObject*, PyObject* h) {
    if (!plain_hypothesis(h)) Py_RETURN_NONE;
    if (!PyDict_GET_SIZE(h)) return PyTuple_New(0);
    Owned groups(PyDict_Keys(h));
    if (!groups.value || PyList_Sort(groups.value) < 0) return nullptr;
    Owned output(PyList_New(0));
    if (!output.value) return nullptr;
    for (Py_ssize_t index = 0; index < PyList_GET_SIZE(groups.value); ++index) {
        PyObject* group = PyList_GET_ITEM(groups.value, index);
        PyObject* integers = PyDict_GetItemWithError(h, group);
        if (!integers) return nullptr;
        const Py_ssize_t count = PyDict_GET_SIZE(integers);
        if (!count) continue;
        // Literal min(satellite keys), then z[minimum] as the gauge base.
        Py_ssize_t position = 0;
        PyObject *satellite, *integer, *minimum = nullptr, *base = nullptr;
        while (PyDict_Next(integers, &position, &satellite, &integer)) {
            const int less = minimum ? PyObject_RichCompareBool(satellite, minimum, Py_LT) : 1;
            if (less < 0) return nullptr;
            if (less) { minimum = satellite; base = integer; }
        }
        Owned entries(PyList_New(count));
        if (!entries.value) return nullptr;
        position = 0;
        Py_ssize_t entry = 0;
        while (PyDict_Next(integers, &position, &satellite, &integer)) {
            Owned difference(PyNumber_Subtract(integer, base));
            if (!difference.value) return nullptr;
            PyObject* pair = PyTuple_Pack(2, satellite, difference.value);
            if (!pair) return nullptr;
            PyList_SET_ITEM(entries.value, entry++, pair);
        }
        if (PyList_Sort(entries.value) < 0) return nullptr;
        Owned ordered(PyList_AsTuple(entries.value));
        if (!ordered.value) return nullptr;
        Owned record(PyTuple_Pack(2, group, ordered.value));
        if (!record.value || PyList_Append(output.value, record.value) < 0) return nullptr;
    }
    return PyList_AsTuple(output.value);
}

PyMethodDef methods[] = {
    {"canonical", canonical, METH_O, "Exact builtin-integer basin key, or None for unsupported input."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_canonical_probe", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__rbpf_canonical_probe() { return PyModule_Create(&module); }
