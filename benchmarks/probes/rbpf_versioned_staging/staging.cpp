// Literal candidate registration using Python objects and original callbacks.
// No native floating-point arithmetic, probability cache, RNG or CUDA changes.
#include <Python.h>
#include <unordered_map>
#include <new>

namespace legacy {
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    explicit operator bool() const { return p != nullptr; }
};

// CPython3.12-only, GIL-held, call-local fusion of the existing pure memo hit
// with its already completed builtin-key lookup. Never cache a callback miss.
static uint64_t version(PyObject* d) { return reinterpret_cast<PyDictObject*>(d)->ma_version_tag; }
static bool pure_key(PyObject* key,int depth=0) {
    if (depth>8) return false;
    if (PyLong_CheckExact(key)) return true;
    if (!PyTuple_CheckExact(key)) return false;
    for (Py_ssize_t k=0;k<PyTuple_GET_SIZE(key);++k)
        if (!pure_key(PyTuple_GET_ITEM(key,k),depth+1)) return false;
    return true;
}
struct Alias {
    PyObject *h, *key, *index;
};
struct Fusion {
    PyObject *bound=nullptr,*memo=nullptr,*function=nullptr,*code=nullptr,*globals=nullptr,*builtins=nullptr;
    PyObject *class_dict=nullptr,*identities=nullptr,*counts=nullptr,*lookup=nullptr;
    uint64_t globals_v=0,builtins_v=0,class_v=0,lookup_v=0,identities_v=0;
    Ref request_name{PyUnicode_InternFromString("canonical_requests")};
    Ref identity_name{PyUnicode_InternFromString("canonical_identity_hits")};
    Ref empty_name{PyUnicode_InternFromString("canonical_empty_hits")};
    std::unordered_map<PyObject*,Alias> aliases;
    PyObject* empty_index=nullptr;
    Py_ssize_t hits=0,original_calls=0,invalidations=0,peak_entries=0;
    bool active=false;
    ~Fusion() {
        clear(); Py_XDECREF(bound);Py_XDECREF(memo);Py_XDECREF(identities);Py_XDECREF(counts);
    }
    void clear() {
        for (auto& item:aliases) {Py_DECREF(item.second.h);Py_DECREF(item.second.key);Py_DECREF(item.second.index);}
        aliases.clear();Py_CLEAR(empty_index);
    }
    bool initialize(PyObject* context,PyObject* cache,PyObject* registry,PyObject* fn,PyObject* expected_code,PyObject* memo_type) {
        function=fn;code=expected_code;lookup=registry;
        if (!request_name || !identity_name || !empty_name) return false;
        bound=PyObject_GetAttrString(context,"canonical");if (!bound) return false;
        if (!PyMethod_Check(bound) || PyMethod_GET_FUNCTION(bound)!=fn || !PyFunction_Check(fn)
            || PyFunction_GetCode(fn)!=code || !PyType_Check(memo_type)
            || Py_TYPE(PyMethod_GET_SELF(bound))!=reinterpret_cast<PyTypeObject*>(memo_type)) return true;
        memo=Py_NewRef(PyMethod_GET_SELF(bound));
        // New local registry lists/dicts are empty in the literal caller.
        // Unsupported cache keys retain every original callback/hash/exception.
        if (PyDict_Size(lookup)!=0) return true;
        Py_ssize_t pos=0;PyObject *key,*value;
        while (PyDict_Next(cache,&pos,&key,&value)) if (!pure_key(key)) return true;
        identities=PyObject_GetAttrString(memo,"identities");counts=PyObject_GetAttrString(memo,"counts");
        if (!identities || !counts) return false;
        if (!PyDict_CheckExact(identities) || !PyDict_CheckExact(counts)) return true;
        globals=PyFunction_GetGlobals(fn);builtins=reinterpret_cast<PyFunctionObject*>(fn)->func_builtins;
        class_dict=reinterpret_cast<PyTypeObject*>(memo_type)->tp_dict;
        if (!PyDict_CheckExact(globals) || !PyDict_CheckExact(builtins) || !PyDict_CheckExact(class_dict)) return true;
        globals_v=version(globals);builtins_v=version(builtins);class_v=version(class_dict);lookup_v=version(lookup);
        active=true;return true;
    }
    bool safe(PyObject* current) {
        if (!active) return false;
        if (current!=bound || PyFunction_GetCode(function)!=code || version(globals)!=globals_v
            || version(builtins)!=builtins_v || version(class_dict)!=class_v) {active=false;clear();return false;}
        if (version(lookup)!=lookup_v) {active=false;clear();++invalidations;lookup_v=version(lookup);return false;}
        // Ordinary instance attributes only. Class version is checked first.
        Ref cs(PyObject_GetAttrString(memo,"counts")),ids(PyObject_GetAttrString(memo,"identities"));
        if (!ids || !cs) return false;
        if (ids.p!=identities || cs.p!=counts) {active=false;clear();return false;}
        return true;
    }
    bool retained(PyObject* h,PyObject* key) {
        if (version(identities)!=identities_v) {
            Py_ssize_t pos=0;PyObject *stored,*value;
            while (PyDict_Next(identities,&pos,&stored,&value)) {
                if (!PyLong_CheckExact(stored)) {active=false;clear();return false;}
            }
            identities_v=version(identities);
        }
        Ref id(PyLong_FromVoidPtr(h));if (!id) return false;
        PyObject* entry=PyDict_GetItemWithError(identities,id.p);
        return entry && PyTuple_CheckExact(entry) && PyTuple_GET_SIZE(entry)==2
            && PyTuple_GET_ITEM(entry,0)==h && PyTuple_GET_ITEM(entry,1)==key;
    }
    bool increment(PyObject* name) {
        PyObject* n=PyDict_GetItemWithError(counts,name);
        if (!n || !PyLong_CheckExact(n)) return false;
        Ref one(PyLong_FromLong(1));if (!one) return false;
        Ref next(PyNumber_Add(n,one.p));return next && PyDict_SetItem(counts,name,next.p)==0;
    }
    PyObject* hit(PyObject* h,PyObject* current) {
        if (!safe(current)) return nullptr;
        PyObject *index=nullptr,*kind=nullptr;
        if (PyDict_CheckExact(h) && PyDict_GET_SIZE(h)==0) {index=empty_index;kind=empty_name.p;}
        else {
            auto it=aliases.find(h);
            if (it!=aliases.end()) {
                PyObject* saved_index=it->second.index;
                if (retained(h,it->second.key)) {index=saved_index;kind=identity_name.p;}
            }
        }
        if (!index) return nullptr;
        PyObject *requests=PyDict_GetItemWithError(counts,request_name.p),*part=PyDict_GetItemWithError(counts,kind);
        if (!requests || !part || !PyLong_CheckExact(requests) || !PyLong_CheckExact(part)) {active=false;clear();return nullptr;}
        if (!increment(request_name.p) || !increment(kind)) return nullptr;
        ++hits;return Py_NewRef(index);
    }
    bool remember(PyObject* h,PyObject* key,PyObject* index,PyObject* current) {
        if (!safe(current)) return !PyErr_Occurred();
        if (!pure_key(key) || !PyLong_CheckExact(index)) {active=false;clear();return true;}
        lookup_v=version(lookup);
        if (PyDict_CheckExact(h) && PyDict_GET_SIZE(h)==0) {
            if (!empty_index) empty_index=Py_NewRef(index);
        } else if (retained(h,key) && aliases.find(h)==aliases.end()) {
            aliases.emplace(h,Alias{h,key,index});
            Py_INCREF(h);Py_INCREF(key);Py_INCREF(index);
        }
        if (PyErr_Occurred()) return false;
        Py_ssize_t entries=static_cast<Py_ssize_t>(aliases.size())+(empty_index?1:0);
        if (entries>peak_entries) peak_entries=entries;
        return true;
    }
};

struct Registry {
    PyObject *context, *cache, *keys, *hypotheses, *lookup, *pending;
    Fusion* fusion=nullptr;
    PyObject* index(PyObject* h) {
        Ref canonical(PyObject_GetAttrString(context, "canonical"));
        if (!canonical) return nullptr;
        if (fusion) {
            PyObject* index=fusion->hit(h,canonical.p);
            if (index || PyErr_Occurred()) return index;
            ++fusion->original_calls;
        }
        Ref key(PyObject_CallOneArg(canonical.p, h));
        if (!key) return nullptr;
        if (fusion && fusion->active && version(lookup)!=fusion->lookup_v) {
            fusion->active=false;fusion->clear();++fusion->invalidations;
        }
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
        Ref result(PyObject_GetItem(lookup, key.p));
        if (!result) return nullptr;
        // Own insertion cannot invalidate prior aliases; arbitrary callback
        // writes invalidate them before any future fused hit.
        if (fusion) {
            fusion->lookup_v=version(lookup);
            if (!fusion->remember(h,key.p,result.p,canonical.p)) return nullptr;
        }
        return Py_NewRef(result.p);
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
PyObject* stage_impl(PyObject* args,bool fused) {
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
    if (!fused) return PyLong_FromSsize_t(held);
    Ref stats(Py_BuildValue("{s:n,s:n,s:n,s:n}","fused_hits",fusion.hits,"original_index_calls",fusion.original_calls,
        "invalidations",fusion.invalidations,"peak_entries",fusion.peak_entries));
    Ref result(PyLong_FromSsize_t(held));
    return stats && result ? PyTuple_Pack(2,result.p,stats.p) : nullptr;
}
PyObject* stage(PyObject*,PyObject* args) {
    try {return stage_impl(args,false);} catch (const std::bad_alloc&) {return PyErr_NoMemory();}
}
PyObject* stage_fused(PyObject*,PyObject* args) {
    try {return stage_impl(args,true);} catch (const std::bad_alloc&) {return PyErr_NoMemory();}
}
PyMethodDef methods[] = {
    {"stage", stage, METH_VARARGS, "Literal candidate/index registration. Original callbacks and Python arithmetic remain."},
    {"stage_fused", stage_fused, METH_VARARGS, "Fuse guarded pure memo hits and builtin score-index lookup in one staging call."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_native_staging_probe", nullptr, -1, methods};
}

// Literal candidate registration using Python objects and original callbacks.
// No native floating-point arithmetic, probability cache, RNG or CUDA changes.
#include <Python.h>
#include <unordered_map>
#include <new>

namespace cached {
struct Ref {
    PyObject* p;
    explicit Ref(PyObject* value) : p(value) {}
    ~Ref() { Py_XDECREF(p); }
    Ref(const Ref&) = delete;
    Ref& operator=(const Ref&) = delete;
    explicit operator bool() const { return p != nullptr; }
};

// CPython3.12-only, GIL-held, call-local fusion of the existing pure memo hit
// with its already completed builtin-key lookup. Never cache a callback miss.
static uint64_t version(PyObject* d) { return reinterpret_cast<PyDictObject*>(d)->ma_version_tag; }
static bool pure_key(PyObject* key,int depth=0) {
    if (depth>8) return false;
    if (PyLong_CheckExact(key)) return true;
    if (!PyTuple_CheckExact(key)) return false;
    for (Py_ssize_t k=0;k<PyTuple_GET_SIZE(key);++k)
        if (!pure_key(PyTuple_GET_ITEM(key,k),depth+1)) return false;
    return true;
}
struct Alias {
    PyObject *h, *key, *index;
    PyObject* identity=nullptr;
    uint64_t verified_version=0;
};
struct Fusion {
    PyObject *bound=nullptr,*memo=nullptr,*function=nullptr,*code=nullptr,*globals=nullptr,*builtins=nullptr;
    PyObject *class_dict=nullptr,*identities=nullptr,*counts=nullptr,*lookup=nullptr;
    uint64_t globals_v=0,builtins_v=0,class_v=0,lookup_v=0,identities_v=0;
    Ref request_name{PyUnicode_InternFromString("canonical_requests")};
    Ref identity_name{PyUnicode_InternFromString("canonical_identity_hits")};
    Ref empty_name{PyUnicode_InternFromString("canonical_empty_hits")};
    std::unordered_map<PyObject*,Alias> aliases;
    PyObject* empty_index=nullptr;
    Py_ssize_t hits=0,original_calls=0,invalidations=0,peak_entries=0;
    bool active=false;
    bool cached_ready=false;
    PyObject *cached_context=nullptr,*memo_dict=nullptr,*context_dict=nullptr,*memo_bases=nullptr;
    PyObject **memo_slot=nullptr,**context_slot=nullptr;
    PyTypeObject *memo_class=nullptr,*context_class=nullptr;
    uint64_t memo_dict_v=0,context_dict_v=0,counts_v=0;
    Py_ssize_t attribute_cache_reads=0,identity_cache_hits=0,identity_rechecks=0,cache_deoptimizations=0;
    bool plain_counts() {
        Py_ssize_t p=0;PyObject *k,*v;
        while (PyDict_Next(counts,&p,&k,&v))
            if (!PyUnicode_CheckExact(k) || !PyLong_CheckExact(v)) return false;
        return true;
    }
    bool prepare_cached(PyObject* context) {
        cached_context=context;memo_class=Py_TYPE(memo);context_class=Py_TYPE(context);
        // No data descriptors or custom attribute access on this fast domain.
        if (memo_class->tp_getattro!=PyObject_GenericGetAttr || memo_class->tp_base!=&PyBaseObject_Type
            || context_class->tp_getattro!=PyObject_GenericGetAttr || context_class->tp_base!=&PyBaseObject_Type
            || !(context_class->tp_flags&Py_TPFLAGS_IMMUTABLETYPE)
            || PyDict_GetItemString(context_class->tp_dict,"canonical")
            || PyDict_GetItemString(memo_class->tp_dict,"counts")
            || PyDict_GetItemString(memo_class->tp_dict,"identities")) return true;
        Ref md(PyObject_GenericGetDict(memo,nullptr)),cd(PyObject_GenericGetDict(context,nullptr));
        if (!md || !cd) return false;
        if (!PyDict_CheckExact(md.p) || !PyDict_CheckExact(cd.p)) return true;
        memo_slot=_PyObject_GetDictPtr(memo);context_slot=_PyObject_GetDictPtr(context);
        if (!memo_slot || !context_slot) return !PyErr_Occurred();
        if (*memo_slot!=md.p || *context_slot!=cd.p
            || PyDict_GetItemString(md.p,"counts")!=counts || PyDict_GetItemString(md.p,"identities")!=identities
            || PyDict_GetItemString(cd.p,"canonical")!=bound) return true;
        if (!plain_counts()) {active=false;return true;}
        memo_dict=Py_NewRef(md.p);context_dict=Py_NewRef(cd.p);memo_bases=Py_NewRef(memo_class->tp_bases);
        memo_dict_v=version(memo_dict);context_dict_v=version(context_dict);counts_v=version(counts);
        cached_ready=true;return true;
    }
    PyObject* current_canonical(PyObject* context) {
        if (cached_ready && Py_TYPE(context)==context_class && *context_slot==context_dict
            && version(context_dict)==context_dict_v) {
            ++attribute_cache_reads;return Py_NewRef(bound);
        }
        return PyObject_GetAttrString(context,"canonical");
    }
    bool retained_cached(Alias& alias) {
        const uint64_t current=version(identities);
        if (current!=identities_v) {
            Py_ssize_t p=0;PyObject *k,*v;
            while (PyDict_Next(identities,&p,&k,&v))
                if (!PyLong_CheckExact(k)) {active=false;clear();return false;}
            identities_v=current;
        }
        if (alias.verified_version==current) {++identity_cache_hits;return true;}
        ++identity_rechecks;
        PyObject* entry=PyDict_GetItemWithError(identities,alias.identity);
        if (entry && PyTuple_CheckExact(entry) && PyTuple_GET_SIZE(entry)==2
            && PyTuple_GET_ITEM(entry,0)==alias.h && PyTuple_GET_ITEM(entry,1)==alias.key) {
            alias.verified_version=current;return true;
        }
        return false;
    }
    ~Fusion() {
        clear(); Py_XDECREF(bound);Py_XDECREF(memo);Py_XDECREF(identities);Py_XDECREF(counts);
        Py_XDECREF(memo_dict);Py_XDECREF(context_dict);Py_XDECREF(memo_bases);
    }
    void clear() {
        for (auto& item:aliases) {Py_DECREF(item.second.h);Py_DECREF(item.second.key);Py_DECREF(item.second.index);Py_XDECREF(item.second.identity);}
        aliases.clear();Py_CLEAR(empty_index);
    }
    bool initialize(PyObject* context,PyObject* cache,PyObject* registry,PyObject* fn,PyObject* expected_code,PyObject* memo_type) {
        function=fn;code=expected_code;lookup=registry;
        if (!request_name || !identity_name || !empty_name) return false;
        bound=PyObject_GetAttrString(context,"canonical");if (!bound) return false;
        if (!PyMethod_Check(bound) || PyMethod_GET_FUNCTION(bound)!=fn || !PyFunction_Check(fn)
            || PyFunction_GetCode(fn)!=code || !PyType_Check(memo_type)
            || Py_TYPE(PyMethod_GET_SELF(bound))!=reinterpret_cast<PyTypeObject*>(memo_type)) return true;
        memo=Py_NewRef(PyMethod_GET_SELF(bound));
        // New local registry lists/dicts are empty in the literal caller.
        // Unsupported cache keys retain every original callback/hash/exception.
        if (PyDict_Size(lookup)!=0) return true;
        Py_ssize_t pos=0;PyObject *key,*value;
        while (PyDict_Next(cache,&pos,&key,&value)) if (!pure_key(key)) return true;
        identities=PyObject_GetAttrString(memo,"identities");counts=PyObject_GetAttrString(memo,"counts");
        if (!identities || !counts) return false;
        if (!PyDict_CheckExact(identities) || !PyDict_CheckExact(counts)) return true;
        globals=PyFunction_GetGlobals(fn);builtins=reinterpret_cast<PyFunctionObject*>(fn)->func_builtins;
        class_dict=reinterpret_cast<PyTypeObject*>(memo_type)->tp_dict;
        if (!PyDict_CheckExact(globals) || !PyDict_CheckExact(builtins) || !PyDict_CheckExact(class_dict)) return true;
        globals_v=version(globals);builtins_v=version(builtins);class_v=version(class_dict);lookup_v=version(lookup);
        active=true;return prepare_cached(context);
    }
    bool safe(PyObject* current) {
        if (!active) return false;
        if (current!=bound || PyFunction_GetCode(function)!=code || version(globals)!=globals_v
            || version(builtins)!=builtins_v || version(class_dict)!=class_v) {active=false;clear();return false;}
        if (version(lookup)!=lookup_v) {active=false;clear();++invalidations;lookup_v=version(lookup);return false;}
        // Check the same mutable state on every request without attribute resolution.
        if (cached_ready) {
            if (Py_TYPE(memo)==memo_class && memo_class->tp_getattro==PyObject_GenericGetAttr
                && memo_class->tp_bases==memo_bases && *memo_slot==memo_dict && version(memo_dict)==memo_dict_v
                && Py_TYPE(cached_context)==context_class && *context_slot==context_dict
                && version(context_dict)==context_dict_v) {
                if (version(counts)!=counts_v) {
                    if (!plain_counts()) {active=false;clear();return false;}
                    counts_v=version(counts);
                }
                return true;
            }
            cached_ready=false;++cache_deoptimizations;active=false;clear();return false;
        }
        // Exact original guard after a dictionary/layout change.
        Ref cs(PyObject_GetAttrString(memo,"counts")),ids(PyObject_GetAttrString(memo,"identities"));
        if (!ids || !cs) return false;
        if (ids.p!=identities || cs.p!=counts) {active=false;clear();return false;}
        return true;
    }
    bool retained(PyObject* h,PyObject* key) {
        if (version(identities)!=identities_v) {
            Py_ssize_t pos=0;PyObject *stored,*value;
            while (PyDict_Next(identities,&pos,&stored,&value)) {
                if (!PyLong_CheckExact(stored)) {active=false;clear();return false;}
            }
            identities_v=version(identities);
        }
        Ref id(PyLong_FromVoidPtr(h));if (!id) return false;
        PyObject* entry=PyDict_GetItemWithError(identities,id.p);
        return entry && PyTuple_CheckExact(entry) && PyTuple_GET_SIZE(entry)==2
            && PyTuple_GET_ITEM(entry,0)==h && PyTuple_GET_ITEM(entry,1)==key;
    }
    bool increment(PyObject* name) {
        PyObject* n=PyDict_GetItemWithError(counts,name);
        if (!n || !PyLong_CheckExact(n)) return false;
        Ref one(PyLong_FromLong(1));if (!one) return false;
        Ref next(PyNumber_Add(n,one.p));return next && PyDict_SetItem(counts,name,next.p)==0;
    }
    PyObject* hit(PyObject* h,PyObject* current) {
        if (!safe(current)) return nullptr;
        PyObject *index=nullptr,*kind=nullptr;
        if (PyDict_CheckExact(h) && PyDict_GET_SIZE(h)==0) {index=empty_index;kind=empty_name.p;}
        else {
            auto it=aliases.find(h);
            if (it!=aliases.end()) {
                PyObject* saved_index=it->second.index;
                if (cached_ready ? retained_cached(it->second) : retained(h,it->second.key)) {index=saved_index;kind=identity_name.p;}
            }
        }
        if (!index) return nullptr;
        PyObject *requests=PyDict_GetItemWithError(counts,request_name.p),*part=PyDict_GetItemWithError(counts,kind);
        if (!requests || !part || !PyLong_CheckExact(requests) || !PyLong_CheckExact(part)) {active=false;clear();return nullptr;}
        if (!increment(request_name.p) || !increment(kind)) return nullptr;
        if (cached_ready) counts_v=version(counts);
        ++hits;return Py_NewRef(index);
    }
    bool remember(PyObject* h,PyObject* key,PyObject* index,PyObject* current) {
        if (!safe(current)) return !PyErr_Occurred();
        if (!pure_key(key) || !PyLong_CheckExact(index)) {active=false;clear();return true;}
        lookup_v=version(lookup);
        if (PyDict_CheckExact(h) && PyDict_GET_SIZE(h)==0) {
            if (!empty_index) empty_index=Py_NewRef(index);
        } else if (retained(h,key) && aliases.find(h)==aliases.end()) {
            Ref identity(cached_ready?PyLong_FromVoidPtr(h):nullptr);
            if (cached_ready && !identity) return false;
            aliases.emplace(h,Alias{h,key,index,identity.p,version(identities)});
            Py_INCREF(h);Py_INCREF(key);Py_INCREF(index);Py_XINCREF(identity.p);
        }
        if (PyErr_Occurred()) return false;
        Py_ssize_t entries=static_cast<Py_ssize_t>(aliases.size())+(empty_index?1:0);
        if (entries>peak_entries) peak_entries=entries;
        return true;
    }
};

struct Registry {
    PyObject *context, *cache, *keys, *hypotheses, *lookup, *pending;
    Fusion* fusion=nullptr;
    PyObject* index(PyObject* h) {
        Ref canonical(fusion?fusion->current_canonical(context):PyObject_GetAttrString(context,"canonical"));
        if (!canonical) return nullptr;
        if (fusion) {
            PyObject* index=fusion->hit(h,canonical.p);
            if (index || PyErr_Occurred()) return index;
            ++fusion->original_calls;
        }
        Ref key(PyObject_CallOneArg(canonical.p, h));
        if (!key) return nullptr;
        if (fusion && fusion->active && version(lookup)!=fusion->lookup_v) {
            fusion->active=false;fusion->clear();++fusion->invalidations;
        }
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
        Ref result(PyObject_GetItem(lookup, key.p));
        if (!result) return nullptr;
        // Own insertion cannot invalidate prior aliases; arbitrary callback
        // writes invalidate them before any future fused hit.
        if (fusion) {
            fusion->lookup_v=version(lookup);
            if (!fusion->remember(h,key.p,result.p,canonical.p)) return nullptr;
        }
        return Py_NewRef(result.p);
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
PyObject* stage_impl(PyObject* args,bool fused) {
    PyObject *context, *runtime, *cfg, *prob, *consensus, *candidates, *spawn,
        *cache, *keys, *hypotheses, *lookup, *pending, *staged;
    PyObject *fn=nullptr,*code=nullptr,*memo_type=nullptr;
    if (fused) {
        if (!PyArg_ParseTuple(args,"OOOOOOOOOOOOOOOO:stage_versioned",&context,&runtime,&cfg,&prob,
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
    if (!fused) return PyLong_FromSsize_t(held);
    Ref stats(Py_BuildValue("{s:n,s:n,s:n,s:n,s:n,s:n,s:n,s:n}","fused_hits",fusion.hits,"original_index_calls",fusion.original_calls,
        "invalidations",fusion.invalidations,"peak_entries",fusion.peak_entries,
        "attribute_cache_reads",fusion.attribute_cache_reads,"identity_cache_hits",fusion.identity_cache_hits,
        "identity_rechecks",fusion.identity_rechecks,"cache_deoptimizations",fusion.cache_deoptimizations));
    Ref result(PyLong_FromSsize_t(held));
    return stats && result ? PyTuple_Pack(2,result.p,stats.p) : nullptr;
}
PyObject* stage(PyObject*,PyObject* args) {
    try {return stage_impl(args,false);} catch (const std::bad_alloc&) {return PyErr_NoMemory();}
}
PyObject* stage_fused(PyObject*,PyObject* args) {
    try {return stage_impl(args,true);} catch (const std::bad_alloc&) {return PyErr_NoMemory();}
}
PyMethodDef methods[] = {
    {"stage", stage, METH_VARARGS, "Literal candidate/index registration. Original callbacks and Python arithmetic remain."},
    {"stage_fused", stage_fused, METH_VARARGS, "Fuse guarded pure memo hits and builtin score-index lookup in one staging call."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_native_staging_probe", nullptr, -1, methods};
}

namespace {
PyMethodDef methods[] = {
    {"stage",legacy::stage,METH_VARARGS,"Original D241 loop, unmodified control."},
    {"stage_fused",legacy::stage_fused,METH_VARARGS,"Original D244 fusion, unmodified control."},
    {"stage_versioned",cached::stage_fused,METH_VARARGS,"Per-request guarded dictionary/identity access reuse."},
    {nullptr,nullptr,0,nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT,"_rbpf_native_staging_probe",nullptr,-1,methods};
}
PyMODINIT_FUNC PyInit__rbpf_native_staging_probe() { return PyModule_Create(&module); }
