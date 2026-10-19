// Exact insertion-ordered integer fingerprints. No arithmetic or numerical code.
#include <Python.h>
#include <cstdint>

namespace {
constexpr const char* names[] = {"type", "dict", "tuple", "int", "any", "len"};
struct Keys {
    PyObject_HEAD
    PyObject *source, *code, *bindings;
    uint64_t keys, checks, native_keys, native_checks, fallback_keys, fallback_checks;
};
int traverse(Keys* self, visitproc visit, void* arg) {
    Py_VISIT(self->source); Py_VISIT(self->code); Py_VISIT(self->bindings);
    return 0;
}
int clear(Keys* self) {
    Py_CLEAR(self->source); Py_CLEAR(self->code); Py_CLEAR(self->bindings);
    return 0;
}
void dealloc(Keys* self) {
    PyObject_GC_UnTrack(self); clear(self);
    PyTypeObject* type = Py_TYPE(self);
    type->tp_free(reinterpret_cast<PyObject*>(self)); Py_DECREF(type);
}
int initialize(Keys* self, PyObject* args, PyObject*) {
    PyObject *source, *code, *bindings;
    if (!PyArg_ParseTuple(args, "OOO:OrderedKeys", &source, &code, &bindings)) return -1;
    if (!PyFunction_Check(source) || !PyCode_Check(code) || !PyTuple_CheckExact(bindings)
        || PyTuple_GET_SIZE(bindings) != 6) {
        PyErr_SetString(PyExc_TypeError, "Expected literal helper, code and six builtin bindings"); return -1;
    }
    // INCREF before clearing also supports reinitializing with existing inputs.
    Py_INCREF(source); Py_INCREF(code); Py_INCREF(bindings);
    clear(self); self->source = source; self->code = code; self->bindings = bindings;
    self->keys = self->checks = self->native_keys = self->native_checks = 0;
    self->fallback_keys = self->fallback_checks = 0;
    return 0;
}
bool ordinary(Keys* self) {
    if (!self->source || PyFunction_GET_CODE(self->source) != self->code) return false;
    PyObject* globals = PyFunction_GET_GLOBALS(self->source);
    PyObject* builtins = reinterpret_cast<PyFunctionObject*>(self->source)->func_builtins;
    for (Py_ssize_t i = 0; i < 6; ++i) {
        PyObject* value = PyDict_GetItemString(globals, names[i]);
        if (!value) value = PyDict_GetItemString(builtins, names[i]);
        if (value != PyTuple_GET_ITEM(self->bindings, i)) return false;
    }
    return true;
}
bool supported(PyObject* hypothesis) {
    if (!PyDict_CheckExact(hypothesis)) return false;
    Py_ssize_t pos = 0;
    PyObject *group, *integers;
    while (PyDict_Next(hypothesis, &pos, &group, &integers)) {
        if (!PyTuple_CheckExact(group) || PyTuple_GET_SIZE(group) != 2
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 0))
            || !PyLong_CheckExact(PyTuple_GET_ITEM(group, 1))
            || !PyDict_CheckExact(integers)) return false;
        Py_ssize_t inner = 0;
        PyObject *sat, *value;
        while (PyDict_Next(integers, &inner, &sat, &value)) {
            if (!PyLong_CheckExact(sat) || !PyLong_CheckExact(value)) return false;
        }
    }
    return true;
}
PyObject* key(Keys* self, PyObject* hypothesis) {
    ++self->keys;
    if (!self->source) { PyErr_SetString(PyExc_RuntimeError, "Uninitialized OrderedKeys"); return nullptr; }
    if (!ordinary(self)) {
        ++self->fallback_keys;
        return PyObject_CallOneArg(self->source, hypothesis);
    }
    ++self->native_keys;
    if (!supported(hypothesis)) Py_RETURN_NONE;
    PyObject* result = PyTuple_New(PyDict_GET_SIZE(hypothesis));
    if (!result) return nullptr;
    Py_ssize_t pos = 0, index = 0;
    PyObject *group, *integers;
    while (PyDict_Next(hypothesis, &pos, &group, &integers)) {
        PyObject* items = PyTuple_New(PyDict_GET_SIZE(integers));
        if (!items) { Py_DECREF(result); return nullptr; }
        Py_ssize_t inner = 0, item_index = 0;
        PyObject *sat, *value;
        while (PyDict_Next(integers, &inner, &sat, &value)) {
            PyObject* pair = PyTuple_Pack(2, sat, value);
            if (!pair) { Py_DECREF(items); Py_DECREF(result); return nullptr; }
            PyTuple_SET_ITEM(items, item_index++, pair);
        }
        PyObject* entry = PyTuple_Pack(2, group, items);
        Py_DECREF(items);
        if (!entry) { Py_DECREF(result); return nullptr; }
        PyTuple_SET_ITEM(result, index++, entry);
    }
    return result;
}
PyObject* accepts(Keys* self, PyObject* hypothesis) {
    ++self->checks;
    if (!self->source) { PyErr_SetString(PyExc_RuntimeError, "Uninitialized OrderedKeys"); return nullptr; }
    if (!ordinary(self)) {
        ++self->fallback_checks;
        PyObject* value = PyObject_CallOneArg(self->source, hypothesis);
        if (!value) return nullptr;
        bool ok = value != Py_None;
        Py_DECREF(value);
        return PyBool_FromLong(ok);
    }
    ++self->native_checks;
    return PyBool_FromLong(supported(hypothesis));
}
PyObject* summary(Keys* self, PyObject*) {
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:K,s:K}",
        "key_requests", static_cast<unsigned long long>(self->keys),
        "support_requests", static_cast<unsigned long long>(self->checks),
        "native_keys", static_cast<unsigned long long>(self->native_keys),
        "native_support", static_cast<unsigned long long>(self->native_checks),
        "fallback_keys", static_cast<unsigned long long>(self->fallback_keys),
        "fallback_support", static_cast<unsigned long long>(self->fallback_checks));
}
PyMethodDef methods[] = {
    {"key", reinterpret_cast<PyCFunction>(key), METH_O, "Original insertion order, integer objects and absolute gauges."},
    {"supported", reinterpret_cast<PyCFunction>(accepts), METH_O, "Same original eligibility without allocating a fingerprint."},
    {"summary", reinterpret_cast<PyCFunction>(summary), METH_NOARGS, "Counters only; no persistent hypothesis cache."},
    {nullptr, nullptr, 0, nullptr}
};
PyType_Slot slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(PyType_GenericNew)}, {Py_tp_init, reinterpret_cast<void*>(initialize)},
    {Py_tp_dealloc, reinterpret_cast<void*>(dealloc)}, {Py_tp_traverse, reinterpret_cast<void*>(traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(clear)}, {Py_tp_methods, methods}, {0, nullptr}
};
PyType_Spec spec = {"_rbpf_ordered_keys_probe.OrderedKeys", sizeof(Keys), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, slots};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_ordered_keys_probe", nullptr, -1, nullptr};
}
PyMODINIT_FUNC PyInit__rbpf_ordered_keys_probe() {
    PyObject* result = PyModule_Create(&module);
    if (!result) return nullptr;
    PyObject* type = PyType_FromSpec(&spec);
    if (!type || PyModule_AddObject(result, "OrderedKeys", type) < 0) {
        Py_XDECREF(type); Py_DECREF(result); return nullptr;
    }
    return result;
}
