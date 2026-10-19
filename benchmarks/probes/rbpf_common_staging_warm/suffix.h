// Call-local replay of pure shared memo hits; no probability/RNG changes.
struct Suffix {
    struct Item { PyObject *h, *key, *index; bool empty; };
    struct Stamp { PyObject* object; uint64_t value; };
    Ref records{nullptr};
    Ref context_dict{nullptr}, cfg_dict{nullptr};
    Ref context_class_dict{nullptr}, cfg_class_dict{nullptr};
    PyObject* cfg_source=nullptr;
    std::vector<Item> items;
    std::vector<Stamp> stamps;
    PyObject *candidates=nullptr, *spawn=nullptr;
    std::vector<PyObject*> candidate_items, spawn_items;
    uint64_t context_v=0, cfg_v=0;
    PyTypeObject *context_type=nullptr, *cfg_type=nullptr;
    uint64_t context_class_v=0, cfg_class_v=0;
    bool attempted=false, ready=false;
    Py_ssize_t replayed_moves=0, replayed_particles=0;

    ~Suffix() {
        for (auto& item: items) { Py_DECREF(item.h); Py_DECREF(item.key); Py_DECREF(item.index); }
        for (auto& stamp: stamps) Py_DECREF(stamp.object);
        for (auto* h: candidate_items) Py_DECREF(h);
        for (auto* h: spawn_items) Py_DECREF(h);
    }
    void watch(PyObject* object) {
        stamps.push_back({object,version(object)}); Py_INCREF(object);
    }
    bool watch_hypothesis(PyObject* h) {
        if (!PyDict_CheckExact(h)) return false;
        Py_ssize_t pos=0; PyObject *group,*integers;
        while (PyDict_Next(h,&pos,&group,&integers)) {
            if (!PyTuple_CheckExact(group) || PyTuple_GET_SIZE(group)!=2
                || !PyLong_CheckExact(PyTuple_GET_ITEM(group,0))
                || !PyLong_CheckExact(PyTuple_GET_ITEM(group,1)) || !PyDict_CheckExact(integers)) return false;
            Py_ssize_t inner=0; PyObject *sat,*integer;
            while (PyDict_Next(integers,&inner,&sat,&integer))
                if (!PyLong_CheckExact(sat) || !PyLong_CheckExact(integer)) return false;
            watch(integers);
        }
        watch(h); return true;
    }
    bool ordinary_attributes(PyObject* object, const char* const* names, Py_ssize_t count, Ref& dict) {
        auto* type=Py_TYPE(object);
        if (type->tp_getattro!=PyObject_GenericGetAttr || type->tp_base!=&PyBaseObject_Type) return false;
        Ref class_dict(PyType_GetDict(type));
        if (!class_dict || !PyDict_CheckExact(class_dict.p)) return false;
        for (Py_ssize_t i=0;i<count;++i)
            if (PyDict_GetItemString(class_dict.p,names[i])) return false;
        dict.p=PyObject_GenericGetDict(object,nullptr);
        if (!dict) { PyErr_Clear(); return false; }
        return PyDict_CheckExact(dict.p);
    }
    void capture(Registry& registry, Fusion& fusion, PyObject* cfg, PyObject* cand, PyObject* spawned,
                 PyObject* moves, Py_ssize_t begin) {
        attempted=true;
        if (!fusion.active || PyList_GET_SIZE(moves)-begin<2) return;
        const char* cn[] = {"canonical"};
        const char* pn[] = {"p_adopt_cons","p_adopt","p_spawn"};
        if (!ordinary_attributes(registry.context,cn,1,context_dict)
            || !ordinary_attributes(cfg,pn,3,cfg_dict)) return;
        for (const char* name: pn) {
            PyObject* prior=PyDict_GetItemString(cfg_dict.p,name);
            if (!prior || !PyFloat_CheckExact(prior) || !std::isfinite(PyFloat_AS_DOUBLE(prior))
                || !(PyFloat_AS_DOUBLE(prior)>0)) return;
        }
        Ref current(PyObject_GetAttrString(registry.context,"canonical"));
        if (!current || !fusion.safe(current.p)) return;
        records.p=PyList_GetSlice(moves,begin,PyList_GET_SIZE(moves));
        if (!records) return;
        for (Py_ssize_t i=0;i<PyList_GET_SIZE(records.p);++i) {
            PyObject* record=PyList_GET_ITEM(records.p,i);
            PyObject *index=PyTuple_GET_ITEM(record,1), *h=PyTuple_GET_ITEM(record,2);
            if (!PyLong_CheckExact(index) || !watch_hypothesis(h)) return;
            Py_ssize_t n=PyLong_AsSsize_t(index);
            if (PyErr_Occurred()) { PyErr_Clear(); return; }
            if (n<0 || n>=PyList_GET_SIZE(registry.keys)) return;
            bool empty=PyDict_GET_SIZE(h)==0;
            PyObject* key=PyList_GET_ITEM(registry.keys,n);
            if (!empty) {
                Ref identity(PyLong_FromVoidPtr(h));
                if (!identity) return;
                PyObject* entry=PyDict_GetItemWithError(fusion.identities,identity.p);
                if (!entry || !PyTuple_CheckExact(entry) || PyTuple_GET_SIZE(entry)!=2
                    || PyTuple_GET_ITEM(entry,0)!=h) return;
                // Equal canonical keys can have different identities. Retain
                // this H's own memo key; the registry still owns first-seen H.
                key=PyTuple_GET_ITEM(entry,1);
            }
            if (!pure_key(key) || (!empty && !fusion.retained(h,key))
                || PyDict_GetItemWithError(registry.lookup,key)!=index) return;
            items.push_back({h,key,index,empty}); Py_INCREF(h); Py_INCREF(key); Py_INCREF(index);
        }
        candidates=cand; spawn=spawned;
        for (Py_ssize_t i=0;i<PyList_GET_SIZE(cand);++i) {
            auto* h=PyList_GET_ITEM(cand,i); candidate_items.push_back(h); Py_INCREF(h);
        }
        for (Py_ssize_t i=0;i<PyList_GET_SIZE(spawned);++i) {
            auto* h=PyList_GET_ITEM(spawned,i); spawn_items.push_back(h); Py_INCREF(h);
        }
        context_v=version(context_dict.p); cfg_v=version(cfg_dict.p);
        context_type=Py_TYPE(registry.context); cfg_type=Py_TYPE(cfg);
        context_class_dict.p=PyType_GetDict(context_type); cfg_class_dict.p=PyType_GetDict(cfg_type);
        if (!context_class_dict || !cfg_class_dict) return;
        cfg_source=cfg;
        context_class_v=version(context_class_dict.p); cfg_class_v=version(cfg_class_dict.p);
        ready=true;
    }
    bool same_list(PyObject* values,const std::vector<PyObject*>& expected) {
        if (PyList_GET_SIZE(values)!=static_cast<Py_ssize_t>(expected.size())) return false;
        for (Py_ssize_t i=0;i<PyList_GET_SIZE(values);++i)
            if (PyList_GET_ITEM(values,i)!=expected[i]) return false;
        return true;
    }
    bool safe(Registry& registry,Fusion& fusion) {
        if (!ready || !fusion.active || Py_TYPE(registry.context)!=context_type || Py_TYPE(cfg_source)!=cfg_type) return false;
        Ref ctype(PyType_GetDict(context_type)), ftype(PyType_GetDict(cfg_type));
        Ref cdict(PyObject_GenericGetDict(registry.context,nullptr)), fdict(PyObject_GenericGetDict(cfg_source,nullptr));
        if (!ctype || !ftype || !cdict || !fdict) return false;
        if (ctype.p!=context_class_dict.p || ftype.p!=cfg_class_dict.p
            || cdict.p!=context_dict.p || fdict.p!=cfg_dict.p
            || version(ctype.p)!=context_class_v || version(ftype.p)!=cfg_class_v
            || version(context_dict.p)!=context_v || version(cfg_dict.p)!=cfg_v
            || !same_list(candidates,candidate_items) || !same_list(spawn,spawn_items)) return false;
        for (auto& stamp: stamps) if (version(stamp.object)!=stamp.value) return false;
        Ref current(PyObject_GetAttrString(registry.context,"canonical"));
        if (!current || !fusion.safe(current.p)) return false;
        // Reject foreign equality/hash hooks before grouped dictionary reads.
        Py_ssize_t pos=0; PyObject *key,*value;
        while (PyDict_Next(registry.lookup,&pos,&key,&value)) if (!pure_key(key)) return false;
        pos=0;
        while (PyDict_Next(fusion.counts,&pos,&key,&value)) if (!PyUnicode_CheckExact(key)) return false;
        PyObject* names[]={fusion.request_name.p,fusion.identity_name.p,fusion.empty_name.p};
        for (auto* name:names) {
            auto* count=PyDict_GetItemWithError(fusion.counts,name);
            if (!count || !PyLong_CheckExact(count)) return false;
        }
        for (auto& item:items) {
            if ((!item.empty && !fusion.retained(item.h,item.key))
                || PyDict_GetItemWithError(registry.lookup,item.key)!=item.index) return false;
        }
        return true;
    }
    bool increment(Fusion& fusion,PyObject* name,Py_ssize_t n) {
        if (!n) return true;
        auto* old=PyDict_GetItemWithError(fusion.counts,name);
        Ref delta(PyLong_FromSsize_t(n));
        Ref next(delta ? PyNumber_Add(old,delta.p):nullptr);
        return next && PyDict_SetItem(fusion.counts,name,next.p)==0;
    }
    bool replay(Registry& registry,Fusion& fusion,PyObject* moves) {
        if (!safe(registry,fusion)) { ready=false; return false; }
        Py_ssize_t empty=0;
        for (auto& item:items) empty+=item.empty;
        auto size=static_cast<Py_ssize_t>(items.size());
        if (!increment(fusion,fusion.request_name.p,size)
            || !increment(fusion,fusion.identity_name.p,size-empty)
            || !increment(fusion,fusion.empty_name.p,empty)) return false;
        for (Py_ssize_t i=0;i<PyList_GET_SIZE(records.p);++i) {
            auto* old=PyList_GET_ITEM(records.p,i);
            Ref record(PyTuple_Pack(4,PyTuple_GET_ITEM(old,0),PyTuple_GET_ITEM(old,1),
                                   PyTuple_GET_ITEM(old,2),PyTuple_GET_ITEM(old,3)));
            if (!record || PyList_Append(moves,record.p)<0) return false;
        }
        fusion.hits+=size; replayed_moves+=size; ++replayed_particles; return true;
    }
};
