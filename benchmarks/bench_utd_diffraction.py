"""Benchmark CPU vs GPU Kouyoumjian-Pathak (UTD) wedge diffraction.

Not part of the pytest suite (no assertions, just timing) -- run directly:

    python benchmarks/bench_utd_diffraction.py

Requires the compiled ``_gnss_gpu_diffraction`` extension (see
BUILD_NOTES.md) for the GPU half of the comparison; the CPU-only run still
prints a comparable baseline if the extension is unavailable.
"""

import time
from types import SimpleNamespace

import numpy as np

from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths


def _make_boxes(n_boxes, seed):
    """Box "buildings" as wedge edges (4 vertical corners each), scaled up to
    reach roughly n_boxes * 4 edges -- used to approximate a ~10k-triangle
    scene's worth of candidate wedge edges (a box mesh of B buildings has
    12*B triangles and 4*B vertical wedge edges).
    """
    rng = np.random.default_rng(seed)
    starts, ends, face_a, face_b, wedge_n = [], [], [], [], []

    for _ in range(n_boxes):
        cx, cy = rng.uniform(-300.0, 300.0, 2)
        hw = rng.uniform(5.0, 20.0)
        hd = rng.uniform(5.0, 20.0)
        h = rng.uniform(10.0, 60.0)

        corners = [
            (cx - hw, cy - hd), (cx + hw, cy - hd),
            (cx + hw, cy + hd), (cx - hw, cy + hd),
        ]
        normals = [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)]
        for i in range(4):
            x, y = corners[i]
            starts.append([x, y, -5.0])
            ends.append([x, y, -5.0 + h])
            fa = normals[i]
            fb = normals[(i - 1) % 4]
            face_a.append([fa[0], fa[1], 0.0])
            face_b.append([fb[0], fb[1], 0.0])
            wedge_n.append(1.5)

    start = np.asarray(starts, dtype=float)
    end = np.asarray(ends, dtype=float)
    n = len(starts)
    return SimpleNamespace(
        start=start, end=end, midpoint=0.5 * (start + end), size=n,
        face_dir_a=np.asarray(face_a, dtype=float),
        face_dir_b=np.asarray(face_b, dtype=float),
        wedge_n=np.asarray(wedge_n, dtype=float),
    )


def _make_sats(rx, n, seed):
    rng = np.random.default_rng(seed)
    d = rng.uniform(-1.0, 1.0, (n, 3))
    d[:, 2] = np.abs(d[:, 2]) + 0.2
    return rx + d / np.linalg.norm(d, axis=1, keepdims=True) * 2.2e7


def _make_receivers(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-250.0, 250.0, (n, 2))
    z = np.full(n, 1.5)
    return np.column_stack([xy, z])


def benchmark_utd(n_edge_target=2500, n_rx=125, n_sat=8, n_iter=1):
    """~n_rx * n_sat ~= 1000 receiver-satellite pairs against a scene with
    roughly n_edge_target candidate wedge edges (a box mesh with
    n_edge_target/4 buildings has ~3 * n_edge_target triangles, i.e. this
    targets the ~10k-triangle scale called out in the task).
    """
    n_boxes = max(1, n_edge_target // 4)
    edges = _make_boxes(n_boxes, seed=42)
    n_edge = edges.size
    n_tri_approx = 3 * n_edge  # each box: 4 wedge edges, 12 triangles

    receivers = _make_receivers(n_rx, seed=1)
    sats_per_rx = [_make_sats(rx, n_sat, seed=i) for i, rx in enumerate(receivers)]

    kw = dict(max_paths=4, max_ray_edge_distance_m=40.0,
              max_excess_path_m=150.0, max_edge_range_m=400.0)

    print("=" * 78)
    print("UTD (Kouyoumjian-Pathak) Diffraction Benchmark")
    print(f"scene: {n_boxes} boxes, {n_edge} wedge edges (~{n_tri_approx} triangles)")
    print(f"workload: {n_rx} receivers x {n_sat} satellites "
          f"= {n_rx * n_sat} receiver-satellite pairs")
    print("=" * 78)

    # --- CPU ---
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for rx, sats in zip(receivers, sats_per_rx):
            compute_utd_diffraction_paths(rx, sats, edges, mode="absorbing", **kw)
    cpu_time = (time.perf_counter() - t0) / n_iter
    print(f"CPU:  {cpu_time * 1e3:10.2f} ms/iter over all "
          f"{n_rx * n_sat} pairs "
          f"({cpu_time / (n_rx * n_sat) * 1e6:8.2f} us/pair)")

    # --- GPU ---
    try:
        from gnss_gpu.utd_diffraction import compute_utd_diffraction_paths_gpu
        from gnss_gpu._gnss_gpu_diffraction import utd_diffraction_candidates  # noqa: F401
    except Exception as e:
        print(f"[SKIP] GPU UTD extension not available: {e}")
        return {"cpu_s": cpu_time}

    # Warm-up (first call pays CUDA context init cost).
    compute_utd_diffraction_paths_gpu(receivers[0], sats_per_rx[0], edges,
                                       mode="absorbing", **kw)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        for rx, sats in zip(receivers, sats_per_rx):
            compute_utd_diffraction_paths_gpu(rx, sats, edges, mode="absorbing", **kw)
    gpu_time = (time.perf_counter() - t0) / n_iter
    print(f"GPU:  {gpu_time * 1e3:10.2f} ms/iter over all "
          f"{n_rx * n_sat} pairs "
          f"({gpu_time / (n_rx * n_sat) * 1e6:8.2f} us/pair)")

    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    print("-" * 78)
    print(f"Speedup (CPU/GPU, per-call launch overhead included): {speedup:6.1f}x")
    print("=" * 78)

    return {"cpu_s": cpu_time, "gpu_s": gpu_time, "speedup": speedup}


if __name__ == "__main__":
    benchmark_utd()
