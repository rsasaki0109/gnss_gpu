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
PyObject* canonical_preflighted(PyObject* h) {
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

bool overlap_preflighted(PyObject* child, PyObject* parent, Py_ssize_t& agree, Py_ssize_t& compared) {
    agree = compared = 0;
    Py_ssize_t position = 0;
    PyObject *group, *child_map;
    while (PyDict_Next(child, &position, &group, &child_map)) {
        PyObject* parent_map = PyDict_GetItemWithError(parent, group);
        if (!parent_map) {
            if (PyErr_Occurred()) return false;
            continue;
        }
        Py_ssize_t inner = 0, common = 0;
        PyObject *satellite, *integer, *reference = nullptr;
        PyObject *child_base = nullptr, *parent_base = nullptr;
        while (PyDict_Next(child_map, &inner, &satellite, &integer)) {
            PyObject* parent_integer = PyDict_GetItemWithError(parent_map, satellite);
            if (!parent_integer) {
                if (PyErr_Occurred()) return false;
                continue;
            }
            ++common;
            const int less = reference ? PyObject_RichCompareBool(satellite, reference, Py_LT) : 1;
            if (less < 0) return false;
            if (less) { reference = satellite; child_base = integer; parent_base = parent_integer; }
        }
        if (common < 2) continue;
        inner = 0;
        while (PyDict_Next(child_map, &inner, &satellite, &integer)) {
            // The reference is the same borrowed key selected in the preceding scan.
            if (satellite == reference) continue;
            PyObject* parent_integer = PyDict_GetItemWithError(parent_map, satellite);
            if (!parent_integer) {
                if (PyErr_Occurred()) return false;
                continue;
            }
            Owned child_dd(PyNumber_Subtract(child_base, integer));
            if (!child_dd.value) return false;
            Owned parent_dd(PyNumber_Subtract(parent_base, parent_integer));
            if (!parent_dd.value) return false;
            const int equal = PyObject_RichCompareBool(child_dd.value, parent_dd.value, Py_EQ);
            if (equal < 0) return false;
            ++compared;
            agree += equal;
        }
    }
    return true;
}

// No custom equality/hash is invoked by key lookup on the supported domain.
bool plain_key(PyObject* key, int depth = 0) {
    if (PyLong_CheckExact(key)) return true;
    if (!PyTuple_CheckExact(key) || depth >= 8) return false;
    for (Py_ssize_t i = 0; i < PyTuple_GET_SIZE(key); ++i)
        if (!plain_key(PyTuple_GET_ITEM(key, i), depth + 1)) return false;
    return true;
}

bool bounded_support(PyObject* h, Py_ssize_t& support) {
    support = 0;
    Py_ssize_t position = 0;
    PyObject *group, *integers;
    while (PyDict_Next(h, &position, &group, &integers)) {
        const Py_ssize_t n = PyDict_GET_SIZE(integers);
        const Py_ssize_t add = n > 0 ? n - 1 : 0;
        if (add > PY_SSIZE_T_MAX - support) return false;
        support += add;
    }
    return true;
}

// Unsupported returns None before arithmetic, callbacks, counters or writes.
// Successful output owns state references only until its caller finishes packing.
PyObject* select_states(PyObject*, PyObject* args) {
    PyObject *hypotheses, *states, *minimum, *fraction;
    if (!PyArg_ParseTuple(args, "OOOO:select_states", &hypotheses, &states, &minimum, &fraction)) return nullptr;
    if (!PyList_CheckExact(hypotheses) || !PyDict_CheckExact(states)
        || !PyLong_CheckExact(minimum) || !PyFloat_CheckExact(fraction)) Py_RETURN_NONE;
    Owned zero(PyLong_FromLong(0));
    if (!zero.value) return nullptr;
    const int positive = PyObject_RichCompareBool(minimum, zero.value, Py_GT);
    if (positive < 0) return nullptr;
    if (!positive) Py_RETURN_NONE;
    const Py_ssize_t count = PyList_GET_SIZE(hypotheses), parents = PyDict_GET_SIZE(states);
    if (count && parents > PY_SSIZE_T_MAX / count) Py_RETURN_NONE;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* h = PyList_GET_ITEM(hypotheses, i);
        Py_ssize_t support;
        if (!plain_hypothesis(h) || !bounded_support(h, support)) Py_RETURN_NONE;
    }
    Py_ssize_t position = 0;
    PyObject *key, *state;
    while (PyDict_Next(states, &position, &key, &state)) {
        if (!plain_key(key) || !PyDict_CheckExact(state)) Py_RETURN_NONE;
        Py_ssize_t item = 0;
        PyObject *name, *value;
        while (PyDict_Next(state, &item, &name, &value))
            if (!PyUnicode_CheckExact(name)) Py_RETURN_NONE;
        PyObject* h = PyDict_GetItemString(state, "H");
        if (!h || !plain_hypothesis(h)) Py_RETURN_NONE;
    }
    Owned output(PyList_New(count));
    if (!output.value) return nullptr;
    Py_ssize_t canonical_requests = 0, parent_requests = 0, exact_hits = 0, selected_parents = 0, ties = 0;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject* h = PyList_GET_ITEM(hypotheses, i);
        PyObject* chosen = Py_None;
        Py_ssize_t support;
        if (!bounded_support(h, support)) return nullptr; // already preflighted
        Owned support_value(PyLong_FromSsize_t(support));
        if (!support_value.value) return nullptr;
        const int enough = PyObject_RichCompareBool(support_value.value, minimum, Py_GE);
        if (enough < 0) return nullptr;
        if (PyDict_GET_SIZE(h) && enough) {
            ++canonical_requests;
            Owned bid(canonical_preflighted(h));
            if (!bid.value) return nullptr;
            PyObject* exact = PyDict_GetItemWithError(states, bid.value);
            if (!exact && PyErr_Occurred()) return nullptr;
            if (exact) { chosen = exact; ++exact_hits; }
            else {
                Py_ssize_t scan = 0, best_agree = -1, best_compared = -1;
                PyObject *parent_key, *parent_state, *best = nullptr;
                bool tied = false;
                while (PyDict_Next(states, &scan, &parent_key, &parent_state)) {
                    ++parent_requests;
                    Py_ssize_t agree, compared;
                    if (!overlap_preflighted(h, PyDict_GetItemString(parent_state, "H"), agree, compared)) return nullptr;
                    Owned a(PyLong_FromSsize_t(agree)), c(PyLong_FromSsize_t(compared));
                    if (!a.value || !c.value) return nullptr;
                    const int adequate = PyObject_RichCompareBool(c.value, minimum, Py_GE);
                    if (adequate < 0) return nullptr;
                    if (!adequate) continue;
                    // Use Python int true division, exactly like agree / compared.
                    Owned ratio(PyNumber_TrueDivide(a.value, c.value));
                    if (!ratio.value) return nullptr;
                    const int accepted = PyObject_RichCompareBool(ratio.value, fraction, Py_GE);
                    if (accepted < 0) return nullptr;
                    if (!accepted) continue;
                    if (agree > best_agree || (agree == best_agree && compared > best_compared)) {
                        best = parent_state; best_agree = agree; best_compared = compared; tied = false;
                    } else if (agree == best_agree && compared == best_compared) tied = true;
                }
                if (best && !tied) { chosen = best; ++selected_parents; }
                if (best && tied) ++ties;
            }
        }
        Py_INCREF(chosen);
        PyList_SET_ITEM(output.value, i, chosen);
    }
    return Py_BuildValue("(Onnnnn)", output.value, canonical_requests, parent_requests, exact_hits, selected_parents, ties);
}

PyMethodDef methods[] = {
    {"select_states", select_states, METH_VARARGS, "Batch exact builtin-integer cluster selection; None on unsupported inputs."},
    {nullptr, nullptr, 0, nullptr}
};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_rbpf_cluster_batch_probe", nullptr, -1, methods};
}

PyMODINIT_FUNC PyInit__rbpf_cluster_batch_probe() { return PyModule_Create(&module); }
