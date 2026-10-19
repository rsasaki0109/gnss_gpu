// Identity aliases for existing immutable score keys. No numerical operations.
#include <Python.h>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {
struct State {
    PyObject* retained = nullptr;
    std::unordered_map<PyObject*, PyObject*> aliases;
    uint64_t requests = 0, hits = 0, source = 0, bypass = 0;
    bool enabled = false;
};
struct Memo { PyObject_HEAD State* state; };

void clear_aliases(State* state) {
    for (const auto& item : state->aliases) { Py_DECREF(item.first); Py_DECREF(item.second); }
    state->aliases.clear();
}
int traverse(Memo* self, visitproc visit, void* arg) {
    if (!self->state) return 0;
    Py_VISIT(self->state->retained);
    for (const auto& item : self->state->aliases) { Py_VISIT(item.first); Py_VISIT(item.second); }
    return 0;
}
int clear(Memo* self) {
    if (self->state) {
        self->state->enabled = false;
        clear_aliases(self->state);
        Py_CLEAR(self->state->retained);
    }
    return 0;
}
void dealloc(Memo* self) {
    PyObject_GC_UnTrack(self);
    clear(self);
    delete self->state;
    PyTypeObject* type = Py_TYPE(self);
    type->tp_free(reinterpret_cast<PyObject*>(self));
    Py_DECREF(type);
}
PyObject* create(PyTypeObject* type, PyObject*, PyObject*) {
    Memo* self = reinterpret_cast<Memo*>(type->tp_alloc(type, 0));
    if (!self) return nullptr;
    self->state = new(std::nothrow) State();
    if (!self->state) { Py_DECREF(self); return PyErr_NoMemory(); }
    return reinterpret_cast<PyObject*>(self);
}
int initialize(Memo* self, PyObject* args, PyObject*) {
    PyObject* retained;
    if (!PyArg_ParseTuple(args, "O:IndexMemo", &retained)) return -1;
    if (retained != Py_None && !PyDict_CheckExact(retained)) {
        PyErr_SetString(PyExc_TypeError, "retained identities must be an exact dict or None");
        return -1;
    }
    clear(self);
    self->state->requests = self->state->hits = self->state->source = self->state->bypass = 0;
    self->state->retained = Py_NewRef(retained);
    self->state->enabled = retained != Py_None;
    return 0;
}
bool plain_key(PyObject* key) {
    if (!PyTuple_CheckExact(key)) return false;
    std::vector<PyObject*> stack;
    stack.push_back(key);
    while (!stack.empty()) {
        PyObject* tuple = stack.back(); stack.pop_back();
        for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(tuple); ++i) {
            PyObject* value = PyTuple_GET_ITEM(tuple, i);
            if (PyTuple_CheckExact(value)) stack.push_back(value);
            else if (!PyLong_CheckExact(value)) return false;
        }
    }
    return true;
}
PyObject* get(Memo* self, PyObject* args) {
    PyObject *key, *hypothesis, *original;
    if (!PyArg_ParseTuple(args, "OOO:get", &key, &hypothesis, &original)) return nullptr;
    State* state = self->state;
    ++state->requests;
    if (state->enabled) {
        const auto hit = state->aliases.find(key);
        if (hit != state->aliases.end()) { ++state->hits; return Py_NewRef(hit->second); }
    }
    ++state->source;
    // Keep the original contains/get/set, first-seen hypothesis and callbacks
    // on every miss. The canonical request itself occurs before this method.
    PyObject* result = PyObject_CallFunctionObjArgs(original, key, hypothesis, nullptr);
    if (!result) { state->enabled = false; clear_aliases(state); return nullptr; }
    if (!state->enabled) { ++state->bypass; return result; }
    bool eligible = false;
    try { eligible = plain_key(key); }
    catch (const std::bad_alloc&) { eligible = false; }
    catch (const std::length_error&) { eligible = false; }
    if (!eligible || !PyLong_CheckExact(result)) {
        // A custom key could affect lookup equality later. Keep its original
        // behavior and stop all aliases for the rest of this move block.
        state->enabled = false; clear_aliases(state); ++state->bypass; return result;
    }
    bool retained = PyDict_CheckExact(hypothesis) && PyDict_GET_SIZE(hypothesis) == 0
        && PyTuple_GET_SIZE(key) == 0;
    if (!retained) {
        PyObject* identity = PyLong_FromVoidPtr(hypothesis);
        if (!identity) { PyErr_Clear(); state->enabled = false; clear_aliases(state); ++state->bypass; return result; }
        PyObject* entry = PyDict_GetItemWithError(state->retained, identity);
        Py_DECREF(identity);
        if (PyErr_Occurred()) { Py_DECREF(result); return nullptr; }
        retained = entry && PyTuple_CheckExact(entry) && PyTuple_GET_SIZE(entry) == 2
            && PyTuple_GET_ITEM(entry, 0) == hypothesis && PyTuple_GET_ITEM(entry, 1) == key;
    }
    if (!retained) { ++state->bypass; return result; }
    try {
        const auto inserted = state->aliases.emplace(key, result);
        if (inserted.second) { Py_INCREF(key); Py_INCREF(result); }
        else { state->enabled = false; clear_aliases(state); ++state->bypass; }
    } catch (const std::bad_alloc&) {
        state->enabled = false; clear_aliases(state); ++state->bypass;
    } catch (const std::length_error&) {
        state->enabled = false; clear_aliases(state); ++state->bypass;
    }
    return result;
}
PyObject* summary(Memo* self, PyObject*) {
    const State* s = self->state;
    return Py_BuildValue("{s:K,s:K,s:K,s:K,s:n,s:O,s:O}",
        "requests", static_cast<unsigned long long>(s->requests),
        "identity_hits", static_cast<unsigned long long>(s->hits),
        "source_index_calls", static_cast<unsigned long long>(s->source),
        "bypassed", static_cast<unsigned long long>(s->bypass),
        "entries", static_cast<Py_ssize_t>(s->aliases.size()),
        "enabled", s->enabled ? Py_True : Py_False, "probability_cache", Py_False);
}
PyMethodDef methods[] = {
    {"get", reinterpret_cast<PyCFunction>(get), METH_VARARGS, "Original value lookup or an exact retained-key identity alias."},
    {"summary", reinterpret_cast<PyCFunction>(summary), METH_NOARGS, "Integer counters for this move block."},
    {nullptr, nullptr, 0, nullptr}
};
PyType_Slot slots[] = {
    {Py_tp_new, reinterpret_cast<void*>(create)}, {Py_tp_init, reinterpret_cast<void*>(initialize)},
    {Py_tp_dealloc, reinterpret_cast<void*>(dealloc)}, {Py_tp_traverse, reinterpret_cast<void*>(traverse)},
    {Py_tp_clear, reinterpret_cast<void*>(clear)}, {Py_tp_methods, methods}, {0, nullptr}
};
PyType_Spec spec = {"_rbpf_score_index_probe.IndexMemo", sizeof(Memo), 0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_HAVE_GC, slots};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_score_index_probe", nullptr, -1, nullptr};
}
PyMODINIT_FUNC PyInit__rbpf_score_index_probe() {
    PyObject* result = PyModule_Create(&module);
    if (!result) return nullptr;
    PyObject* type = PyType_FromSpec(&spec);
    if (!type || PyModule_AddObject(result, "IndexMemo", type) < 0) {
        Py_XDECREF(type); Py_DECREF(result); return nullptr;
    }
    return result;
}
