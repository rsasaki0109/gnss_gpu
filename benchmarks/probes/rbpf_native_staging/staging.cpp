// Literal candidate registration using Python objects and original callbacks.
// No native floating-point arithmetic, canonical/index/probability cache or CUDA.
#include <Python.h>

namespace {
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    explicit operator bool() const { return p != nullptr; }
};
struct Registry {
    PyObject *context, *cache, *keys, *hypotheses, *lookup, *pending;
    PyObject* index(PyObject* h) {
        Ref canonical(PyObject_GetAttrString(context, "canonical"));
        if (!canonical) return nullptr;
        Ref key(PyObject_CallOneArg(canonical.p, h));
        if (!key) return nullptr;
        const int found = PyDict_Contains(lookup, key.p);
        if (found < 0) return nullptr;
        if (!found) {
            Ref i(PyLong_FromSsize_t(PyList_GET_SIZE(keys)));
            if (!i || PyDict_SetItem(lookup, key.p, i.p) < 0) return nullptr;
            if (PyList_Append(keys, key.p) < 0 || PyList_Append(hypotheses, h) < 0) return nullptr;
            // Exact dict.get(key): missing and a stored None both mean pending.
            PyObject* cached = PyDict_GetItemWithError(cache, key.p);
            if (!cached && PyErr_Occurred()) return nullptr;
            if (!cached || cached == Py_None) {
                Ref pending_index(PyLong_FromSsize_t(PyList_GET_SIZE(keys) - 1));
                if (!pending_index || PyList_Append(pending, pending_index.p) < 0) return nullptr;
            }
        }
        // Keep the original second/third key hash and any custom exceptions.
        return PyObject_GetItem(lookup, key.p);
    }
    bool add(PyObject* moves, PyObject* prior, PyObject* h, PyObject* tag) {
        Ref i(index(h));
        if (!i) return false;
        Ref record(PyTuple_Pack(4, prior, i.p, h, tag));
        return record && PyList_Append(moves, record.p) == 0;
    }
    bool add_field(PyObject* moves, PyObject* cfg, const char* field, PyObject* h, PyObject* tag) {
        Ref prior(PyObject_GetAttrString(cfg, field));
        return prior && add(moves, prior.p, h, tag);
    }
    bool shared(PyObject* moves, PyObject* cfg, const char* field, PyObject* candidates, PyObject* tag) {
        for (Py_ssize_t k = 0; k < PyList_GET_SIZE(candidates); ++k) {
            // Python list iteration keeps the current hypothesis alive while
            // original callbacks execute. Length/division still run per move.
            Ref h(Py_NewRef(PyList_GET_ITEM(candidates, k)));
            Ref prior(PyObject_GetAttrString(cfg, field));
            if (!prior) return false;
            Ref count(PyLong_FromSsize_t(PyList_GET_SIZE(candidates)));
            if (!count) return false;
            Ref divided(PyNumber_TrueDivide(prior.p, count.p));
            if (!divided || !add(moves, divided.p, h.p, tag)) return false;
        }
        return true;
    }
};
PyObject* stage(PyObject*, PyObject* args) {
    PyObject *context, *runtime, *cfg, *prob, *consensus, *candidates, *spawn,
        *cache, *keys, *hypotheses, *lookup, *pending, *staged;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOO:stage", &context, &runtime, &cfg, &prob,
        &consensus, &candidates, &spawn, &cache, &keys, &hypotheses, &lookup, &pending, &staged)) return nullptr;
    if (!PyList_CheckExact(candidates) || !PyList_CheckExact(spawn)
        || !PyList_CheckExact(keys) || !PyList_CheckExact(hypotheses)
        || !PyList_CheckExact(pending) || !PyList_CheckExact(staged)
        || !PyDict_CheckExact(cache) || !PyDict_CheckExact(lookup)) {
        PyErr_SetString(PyExc_TypeError, "Native staging requires exact lists/dicts"); return nullptr;
    }
    Ref particles(PyObject_GetAttrString(runtime, "particles"));
    if (!particles) return nullptr;
    if (!PyList_CheckExact(particles.p)) {
        PyErr_SetString(PyExc_TypeError, "Native staging requires an exact particle list"); return nullptr;
    }
    Ref survive(PyUnicode_InternFromString("survive")), extend(PyUnicode_InternFromString("extend")),
        shrink(PyUnicode_InternFromString("shrink")), release(PyUnicode_InternFromString("release")),
        stay(PyUnicode_InternFromString("stay")), adopt_cons(PyUnicode_InternFromString("adopt_cons")),
        adopt(PyUnicode_InternFromString("adopt")), spawned(PyUnicode_InternFromString("spawn"));
    if (!survive || !extend || !shrink || !release || !stay || !adopt_cons || !adopt || !spawned) return nullptr;
    Registry registry{context, cache, keys, hypotheses, lookup, pending};
    Py_ssize_t held = 0;
    for (Py_ssize_t k = 0; k < PyList_GET_SIZE(particles.p); ++k) {
        Ref particle(Py_NewRef(PyList_GET_ITEM(particles.p, k)));
        Ref h(PyObject_GetAttrString(particle.p, "H"));
        if (!h) return nullptr;
        Ref moves(PyList_New(0));
        if (!moves) return nullptr;
        int truth = PyObject_IsTrue(h.p);
        if (truth < 0) return nullptr;
        if (truth) {
            ++held;
            if (!registry.add_field(moves.p, cfg, "p_survive", h.p, survive.p)) return nullptr;
            Ref extended(PyObject_CallMethod(runtime, "_extend", "OO", h.p, prob));
            if (!extended) return nullptr;
            if (extended.p != Py_None && !registry.add_field(moves.p, cfg, "p_extend", extended.p, extend.p)) return nullptr;
            Ref shrunk(PyObject_CallMethod(runtime, "_shrink_worst", "OO", h.p, prob));
            if (!shrunk) return nullptr;
            if (shrunk.p != Py_None && !registry.add_field(moves.p, cfg, "p_shrink", shrunk.p, shrink.p)) return nullptr;
            // Original add evaluates p_release before constructing a fresh {}.
            Ref prior(PyObject_GetAttrString(cfg, "p_release"));
            if (!prior) return nullptr;
            Ref empty(PyDict_New());
            if (!empty || !registry.add(moves.p, prior.p, empty.p, release.p)) return nullptr;
        } else {
            Ref a(PyObject_GetAttrString(cfg, "p_survive"));
            if (!a) return nullptr;
            Ref b(PyObject_GetAttrString(cfg, "p_extend"));
            if (!b) return nullptr;
            Ref ab(PyNumber_Add(a.p, b.p));
            if (!ab) return nullptr;
            Ref c(PyObject_GetAttrString(cfg, "p_shrink"));
            if (!c) return nullptr;
            Ref abc(PyNumber_Add(ab.p, c.p));
            if (!abc) return nullptr;
            Ref d(PyObject_GetAttrString(cfg, "p_release"));
            if (!d) return nullptr;
            Ref prior(PyNumber_Add(abc.p, d.p));
            if (!prior || !registry.add(moves.p, prior.p, h.p, stay.p)) return nullptr;
        }
        if (consensus != Py_None && !registry.add_field(moves.p, cfg, "p_adopt_cons", consensus, adopt_cons.p)) return nullptr;
        if (!registry.shared(moves.p, cfg, "p_adopt", candidates, adopt.p)
            || !registry.shared(moves.p, cfg, "p_spawn", spawn, spawned.p)) return nullptr;
        if (PyList_Append(staged, moves.p) < 0) return nullptr;
    }
    return PyLong_FromSsize_t(held);
}
PyMethodDef methods[] = {
    {"stage", stage, METH_VARARGS, "Literal candidate/index registration. Original callbacks and Python arithmetic remain."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_native_staging_probe", nullptr, -1, methods};
}
PyMODINIT_FUNC PyInit__rbpf_native_staging_probe() { return PyModule_Create(&module); }
