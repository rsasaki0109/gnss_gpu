"""Pinned transport and bounded CUDA Graph;inherit frozen score math/packing."""

import ctypes

from rbpf_likelihood_batch import BatchScorer


class GraphBatchScorer(BatchScorer):
    def __init__(self, dll, capacity=512, *, transport="graph"):
        if transport not in ("stream", "graph"):
            raise ValueError("Select an explicit stream/graph transport")
        super().__init__(dll, capacity)
        try:
            self.dll.rbpf_graph_select.argtypes = [ctypes.c_void_p, ctypes.c_int]
            self.dll.rbpf_graph_select.restype = ctypes.c_int
            self.dll.rbpf_graph_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
            self.dll.rbpf_graph_stats.restype = ctypes.c_int
            code = self.dll.rbpf_graph_select(self.handle, int(transport == "graph"))
            if code:
                raise RuntimeError(f"Graph transport setup failed: {code}")
        except BaseException:
            self.close()
            raise
        self.transport = transport

    def stats(self):
        """Read submission counts outside score timing;no CUDA events or work."""
        if not self.handle:
            raise RuntimeError("Closed scorer")
        counts = (ctypes.c_ulonglong * 4)()
        if self.dll.rbpf_graph_stats(self.handle, counts):
            raise RuntimeError("Graph transport count query failed")
        return dict(zip(("instantiations", "shape_updates", "graph_launches", "stream_launches"), counts, strict=True))
