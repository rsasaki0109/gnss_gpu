// Explicit CPython 3.12 input-packing probe. Call with ctypes.PyDLL (GIL held).
#include <Python.h>

namespace {
bool group_key(PyObject* object) {
    return PyTuple_CheckExact(object) && PyTuple_GET_SIZE(object) == 2
        && PyLong_CheckExact(PyTuple_GET_ITEM(object, 0))
        && PyLong_CheckExact(PyTuple_GET_ITEM(object, 1));
}

bool hypothesis(PyObject* object) {
    if (!PyDict_CheckExact(object)) return false;
    Py_ssize_t position = 0;
    PyObject *group, *integers;
    while (PyDict_Next(object, &position, &group, &integers)) {
        if (!group_key(group)) return false;
        if (integers == Py_None) continue;
        if (!PyDict_CheckExact(integers)) return false;
        Py_ssize_t inner = 0;
        PyObject *satellite, *value;
        while (PyDict_Next(integers, &inner, &satellite, &value)) {
            if (!PyLong_CheckExact(satellite)
                || (value != Py_None && !PyLong_CheckExact(value))) return false;
        }
    }
    return true;
}
}

// 1: packed; 0: unsupported, no writes; -1: a CPython exception is pending.
// All shape/type validation precedes output writes or arithmetic. Integers are
// subtracted as arbitrary-precision Python ints, then rounded to FP64 once.
// No cached Python objects, heap buffers, CUDA state, or released-GIL interval.
extern "C" __declspec(dllexport) int rbpf_pack_dd(
    PyObject* hypotheses, PyObject* pairs, double* output,
    Py_ssize_t output_elements, double missing) {
    if (!PyGILState_Check()) return 0;
    if (!PyList_CheckExact(hypotheses) || !PyList_CheckExact(pairs)) return 0;
    const Py_ssize_t count = PyList_GET_SIZE(hypotheses);
    const Py_ssize_t columns = PyList_GET_SIZE(pairs);
    if (count > 512 || columns > 128 || output_elements < 0
        || output_elements < count * columns || (!output && count * columns)) return 0;
    for (Py_ssize_t column = 0; column < columns; ++column) {
        PyObject* pair = PyList_GET_ITEM(pairs, column);
        if (!PyTuple_CheckExact(pair) || PyTuple_GET_SIZE(pair) != 3
            || !group_key(PyTuple_GET_ITEM(pair, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 1))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(pair, 2))) return 0;
    }
    for (Py_ssize_t row = 0; row < count; ++row) {
        if (!hypothesis(PyList_GET_ITEM(hypotheses, row))) return 0;
    }
    for (Py_ssize_t row = 0; row < count; ++row) {
        PyObject* h = PyList_GET_ITEM(hypotheses, row);
        for (Py_ssize_t column = 0; column < columns; ++column) {
            double value = missing;
            PyObject* pair = PyList_GET_ITEM(pairs, column);
            PyObject* integers = PyDict_GetItemWithError(h, PyTuple_GET_ITEM(pair, 0));
            if (PyErr_Occurred()) return -1;
            if (integers && integers != Py_None) {
                PyObject* ref = PyDict_GetItemWithError(integers, PyTuple_GET_ITEM(pair, 1));
                if (PyErr_Occurred()) return -1;
                PyObject* sat = PyDict_GetItemWithError(integers, PyTuple_GET_ITEM(pair, 2));
                if (PyErr_Occurred()) return -1;
                if (ref && ref != Py_None && sat && sat != Py_None) {
                    PyObject* difference = PyNumber_Subtract(ref, sat);
                    if (!difference) return -1;
                    value = PyLong_AsDouble(difference);
                    Py_DECREF(difference);
                    if (PyErr_Occurred()) return -1;
                }
            }
            output[row * columns + column] = value;
        }
    }
    return 1;
}
