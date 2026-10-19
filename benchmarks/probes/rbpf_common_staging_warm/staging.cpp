#include <Python.h>
#include <unordered_map>
#include <new>
#include <vector>
#include <cmath>
namespace d244 {
#define PyInit__rbpf_native_staging_probe PyInit__d244_control
#include "../rbpf_fused_staging/staging.cpp"
#undef PyInit__rbpf_native_staging_probe
}
namespace common {
using namespace d244;
#include "suffix.h"
PyObject* stage_impl(PyObject* args) {
    bool fused=true;
    PyObject *context, *runtime, *cfg, *prob, *consensus, *candidates, *spawn,
        *cache, *keys, *hypotheses, *lookup, *pending, *staged;
    PyObject *fn=nullptr,*code=nullptr,*memo_type=nullptr;
    if (fused) {
        if (!PyArg_ParseTuple(args,"OOOOOOOOOOOOOOOO:stage_fused",&context,&runtime,&cfg,&prob,
            &consensus,&candidates,&spawn,&cache,&keys,&hypotheses,&lookup,&pending,&staged,&fn,&code,&memo_type)) return nullptr;
    } else if (!PyArg_ParseTuple(args,"OOOOOOOOOOOOO:stage",&context,&runtime,&cfg,&prob,
        &consensus,&candidates,&spawn,&cache,&keys,&hypotheses,&lookup,&pending,&staged)) return nullptr;
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
    Fusion fusion;
    if (fused && !fusion.initialize(context,cache,lookup,fn,code,memo_type)) return nullptr;
    Registry registry{context, cache, keys, hypotheses, lookup, pending, fused?&fusion:nullptr};
    Suffix suffix;
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
        if (!suffix.replay(registry,fusion,moves.p)) {
            if (PyErr_Occurred()) return nullptr;
            Py_ssize_t begin=PyList_GET_SIZE(moves.p);
            Py_ssize_t callbacks_before=fusion.original_calls;
        if (consensus != Py_None && !registry.add_field(moves.p, cfg, "p_adopt_cons", consensus, adopt_cons.p)) return nullptr;
        if (!registry.shared(moves.p, cfg, "p_adopt", candidates, adopt.p)
            || !registry.shared(moves.p, cfg, "p_spawn", spawn, spawned.p)) return nullptr;
            if (!suffix.attempted && fusion.original_calls==callbacks_before) suffix.capture(registry,fusion,cfg,candidates,spawn,moves.p,begin);
            if (PyErr_Occurred()) return nullptr;
        }
        if (PyList_Append(staged, moves.p) < 0) return nullptr;
    }
    if (!fused) return PyLong_FromSsize_t(held);
    Ref stats(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n}","fused_hits",fusion.hits,"original_index_calls",fusion.original_calls,
        "invalidations",fusion.invalidations,"peak_entries",fusion.peak_entries,"replayed_moves",suffix.replayed_moves,"replayed_particles",suffix.replayed_particles));
    Ref result(PyLong_FromSsize_t(held));
    return stats && result ? PyTuple_Pack(2,result.p,stats.p) : nullptr;
}
PyObject* stage_common(PyObject*,PyObject* args) {
    try { return stage_impl(args); } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
}
PyMethodDef methods[] = {
    {"stage",d244::stage,METH_VARARGS,"Frozen D241 registration control."},
    {"stage_fused",d244::stage_fused,METH_VARARGS,"Frozen D244 per-hit control."},
    {"stage_common",stage_common,METH_VARARGS,"Group unchanged common suffix memo hits."},
    {nullptr,nullptr,0,nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT,"_rbpf_native_staging_probe",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__rbpf_native_staging_probe() { return PyModule_Create(&common::module); }
