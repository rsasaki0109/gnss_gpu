"""Post-smoothing guards and stop-segment corrections."""

from __future__ import annotations

import numpy as np

from gnss_gpu.pf_smoother_common import finite_float as _finite_float


_STOP_SEGMENT_BASE_SOURCES = {"smoothed", "forward", "combined"}
_STOP_SEGMENT_DENSITY_SUFFIX = "_density"
_STOP_SEGMENT_AUTO_TAIL_SUFFIX = "_auto_tail"
_STOP_SEGMENT_AUTO_SUFFIX = "_auto"


def _diagnostic_bool(row: dict[str, object], key: str) -> bool:
    value = row.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _diagnostic_int_at_most(row: dict[str, object], key: str, max_value: int) -> bool:
    value = row.get(key)
    if value is None:
        return False
    try:
        return int(value) <= int(max_value)
    except (TypeError, ValueError):
        return False


def _apply_smoother_tail_guard(
    smoothed_aligned: np.ndarray,
    forward_aligned: np.ndarray,
    epoch_diagnostics: list[dict[str, object]] | None,
    *,
    ess_max_ratio: float | None = None,
    dd_carrier_max_pairs: int | None = None,
    dd_pseudorange_max_pairs: int | None = None,
    min_shift_m: float | None = None,
    expand_epochs: int | None = None,
    expand_min_shift_m: float | None = None,
    expand_dd_pseudorange_max_pairs: int | None = None,
) -> tuple[np.ndarray, int]:
    if epoch_diagnostics is None:
        return smoothed_aligned, 0

    guarded = np.asarray(smoothed_aligned, dtype=np.float64).copy()
    forward = np.asarray(forward_aligned, dtype=np.float64)
    applied = 0
    shift_m = np.linalg.norm(guarded - forward, axis=1)
    guard_enabled = not (
        ess_max_ratio is None
        and dd_carrier_max_pairs is None
        and dd_pseudorange_max_pairs is None
        and min_shift_m is None
    )
    initially_applied: list[bool] = []
    for i, row in enumerate(epoch_diagnostics):
        row["smoothed_shift_3d_m"] = float(shift_m[i])
        row["tail_guard_applied"] = False
        row["tail_guard_expanded"] = False
        if not guard_enabled:
            initially_applied.append(False)
            continue
        conds: list[bool] = []
        if ess_max_ratio is not None:
            ess_ratio = _finite_float(row.get("gate_ess_ratio"))
            conds.append(
                ess_ratio is not None and float(ess_ratio) <= float(ess_max_ratio)
            )
        if dd_carrier_max_pairs is not None:
            dd_cp_kept = row.get("dd_cp_kept_pairs")
            conds.append(
                dd_cp_kept is not None
                and int(dd_cp_kept) <= int(dd_carrier_max_pairs)
            )
        if dd_pseudorange_max_pairs is not None:
            dd_pr_kept = row.get("dd_pr_kept_pairs")
            conds.append(
                dd_pr_kept is not None
                and int(dd_pr_kept) <= int(dd_pseudorange_max_pairs)
            )
        if min_shift_m is not None:
            conds.append(float(shift_m[i]) >= float(min_shift_m))
        use_forward = bool(conds) and all(conds)
        row["tail_guard_applied"] = use_forward
        initially_applied.append(use_forward)
        if use_forward:
            guarded[i] = forward[i]
            applied += 1
    if expand_epochs is not None and int(expand_epochs) > 0:
        expand_radius = int(expand_epochs)
        anchors = [
            i
            for i, is_applied in enumerate(initially_applied)
            if is_applied and not _diagnostic_bool(epoch_diagnostics[i], "imu_stop_detected")
        ]
        if anchors:
            near_guard = np.zeros(len(epoch_diagnostics), dtype=bool)
            for anchor in anchors:
                lo = max(0, anchor - expand_radius)
                hi = min(len(epoch_diagnostics), anchor + expand_radius + 1)
                near_guard[lo:hi] = True
            for i, row in enumerate(epoch_diagnostics):
                if initially_applied[i] or not bool(near_guard[i]):
                    continue
                if _diagnostic_bool(row, "imu_stop_detected"):
                    continue
                if expand_min_shift_m is not None and float(shift_m[i]) < float(expand_min_shift_m):
                    continue
                expand_dd_pr_ok = (
                    expand_dd_pseudorange_max_pairs is None
                    or _diagnostic_int_at_most(
                        row,
                        "dd_pr_kept_pairs",
                        int(expand_dd_pseudorange_max_pairs),
                    )
                )
                if not expand_dd_pr_ok:
                    continue
                row["tail_guard_applied"] = True
                row["tail_guard_expanded"] = True
                guarded[i] = forward[i]
                applied += 1
    return guarded, applied


def _apply_smoother_widelane_forward_guard(
    smoothed_aligned: np.ndarray,
    forward_aligned: np.ndarray,
    epoch_diagnostics: list[dict[str, object]] | None,
    *,
    min_shift_m: float | None = None,
) -> tuple[np.ndarray, int]:
    if epoch_diagnostics is None:
        return smoothed_aligned, 0

    guarded = np.asarray(smoothed_aligned, dtype=np.float64).copy()
    forward = np.asarray(forward_aligned, dtype=np.float64)
    shift_m = np.linalg.norm(guarded - forward, axis=1)
    applied = 0
    for i, row in enumerate(epoch_diagnostics):
        row["widelane_forward_guard_applied"] = False
        if not bool(row.get("used_widelane")):
            continue
        if min_shift_m is not None and float(shift_m[i]) < float(min_shift_m):
            continue
        guarded[i] = forward[i]
        row["widelane_forward_guard_applied"] = True
        applied += 1
    return guarded, applied


def _stop_segment_ranges(stop_flags: list[bool] | np.ndarray, *, min_epochs: int) -> list[tuple[int, int]]:
    flags = np.asarray(stop_flags, dtype=bool).ravel()
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for i, is_stop in enumerate(flags):
        if is_stop:
            if start is None:
                start = i
            continue
        if start is not None and i - start >= int(min_epochs):
            ranges.append((start, i))
        start = None
    if start is not None and len(flags) - start >= int(min_epochs):
        ranges.append((start, len(flags)))
    return ranges


def _normalize_stop_segment_source(source: str) -> tuple[str, str]:
    src_name = str(source).strip().lower().replace("-", "_")
    if src_name.endswith(_STOP_SEGMENT_DENSITY_SUFFIX):
        base_source = src_name[: -len(_STOP_SEGMENT_DENSITY_SUFFIX)]
        mode = "density"
    elif src_name.endswith(_STOP_SEGMENT_AUTO_TAIL_SUFFIX):
        base_source = src_name[: -len(_STOP_SEGMENT_AUTO_TAIL_SUFFIX)]
        mode = "auto_tail"
    elif src_name.endswith(_STOP_SEGMENT_AUTO_SUFFIX):
        base_source = src_name[: -len(_STOP_SEGMENT_AUTO_SUFFIX)]
        mode = "auto"
    else:
        base_source = src_name
        mode = "median"
    if base_source not in _STOP_SEGMENT_BASE_SOURCES:
        raise ValueError(
            "stop segment source must be one of: smoothed, forward, combined, "
            "smoothed_density, forward_density, combined_density, "
            "smoothed_auto, forward_auto, combined_auto, "
            "smoothed_auto_tail, forward_auto_tail, combined_auto_tail"
        )
    return base_source, mode


def _stop_segment_source_samples(
    smoothed: np.ndarray,
    forward: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    if source == "forward":
        return forward
    if source == "combined":
        return np.vstack([smoothed, forward])
    return smoothed


def _density_stop_segment_center(
    samples: np.ndarray,
    *,
    neighbors: int,
) -> tuple[np.ndarray | None, float | None, int]:
    finite = np.isfinite(samples).all(axis=1)
    finite_samples = np.asarray(samples[finite], dtype=np.float64).reshape(-1, 3)
    if len(finite_samples) == 0:
        return None, None, 0
    k = int(np.clip(int(neighbors), 1, len(finite_samples)))
    diff = finite_samples[:, None, :] - finite_samples[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    kth_distances = np.partition(distances, k - 1, axis=1)[:, k - 1]
    densest_index = int(np.argmin(kth_distances))
    neighbor_indices = np.argsort(distances[densest_index])[:k]
    center = np.median(finite_samples[neighbor_indices], axis=0)
    if not np.isfinite(center).all():
        return None, None, k
    return center, float(kth_distances[densest_index]), k


def _auto_stop_segment_use_density(
    finite_samples: np.ndarray,
    median_center: np.ndarray,
    density_center: np.ndarray,
    density_radius: float | None,
) -> tuple[bool, float, float]:
    median_radius = float(
        np.percentile(np.linalg.norm(finite_samples - median_center, axis=1), 90)
    )
    center_shift = float(np.linalg.norm(density_center - median_center))
    use_density = (
        2.0 <= median_radius <= 15.0
        and density_radius is not None
        and float(density_radius) <= 3.0
        and center_shift >= 0.05
    )
    return use_density, median_radius, center_shift


def _auto_stop_segment_use_principal_tail(
    *,
    median_radius: float,
    density_radius: float | None,
    center_shift: float,
) -> bool:
    return (
        median_radius > 15.0
        and density_radius is not None
        and float(density_radius) <= 3.0
        and center_shift >= 1.0
    )


def _principal_percentile_stop_segment_center(
    finite_samples: np.ndarray,
    *,
    percentile: float = 20.0,
    reference_point: np.ndarray | None = None,
) -> np.ndarray | None:
    if len(finite_samples) == 0:
        return None
    center = np.median(finite_samples, axis=0)
    demeaned = finite_samples - center
    try:
        _u, _s, vh = np.linalg.svd(demeaned, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if len(vh) == 0:
        return center
    axis = vh[0]
    if reference_point is not None and np.isfinite(reference_point).all():
        reference_projection = float((np.asarray(reference_point) - center) @ axis)
        if reference_projection < 0.0:
            axis = -axis
    projection = demeaned @ axis
    target = float(np.percentile(projection, float(percentile)))
    k = max(3, min(25, max(1, len(finite_samples) // 10)))
    k = min(k, len(finite_samples))
    selected = np.argsort(np.abs(projection - target))[:k]
    principal_center = np.median(finite_samples[selected], axis=0)
    if not np.isfinite(principal_center).all():
        return None
    return principal_center


def _apply_stop_segment_constant_position(
    smoothed_aligned: np.ndarray,
    forward_aligned: np.ndarray,
    stop_flags: list[bool] | np.ndarray,
    epoch_diagnostics: list[dict[str, object]] | None,
    *,
    min_epochs: int = 5,
    source: str = "smoothed",
    max_radius_m: float | None = None,
    blend: float = 1.0,
    density_neighbors: int = 200,
) -> tuple[np.ndarray, dict[str, object]]:
    corrected = np.asarray(smoothed_aligned, dtype=np.float64).copy()
    forward = np.asarray(forward_aligned, dtype=np.float64)
    src_name = str(source).strip().lower().replace("-", "_")
    base_source, center_mode = _normalize_stop_segment_source(src_name)
    if len(corrected) == 0:
        return corrected, {
            "segments": 0,
            "segments_applied": 0,
            "epochs_applied": 0,
            "source": src_name,
            "density_neighbors": int(density_neighbors),
            "density_segments_selected": 0,
            "principal_segments_selected": 0,
        }

    ranges = _stop_segment_ranges(stop_flags, min_epochs=min_epochs)
    alpha = float(np.clip(float(blend), 0.0, 1.0))
    info = {
        "segments": int(len(ranges)),
        "segments_applied": 0,
        "epochs_applied": 0,
        "source": src_name,
        "density_neighbors": int(density_neighbors),
        "density_segments_selected": 0,
        "principal_segments_selected": 0,
    }
    if epoch_diagnostics is not None:
        for row in epoch_diagnostics:
            row["stop_segment_constant_applied"] = False
            row["stop_segment_id"] = None
            row["stop_segment_radius_m"] = None
            row["stop_segment_density_radius_m"] = None
            row["stop_segment_density_neighbors"] = None
            row["stop_segment_center_mode"] = None
            row["stop_segment_auto_median_radius_m"] = None
            row["stop_segment_auto_center_shift_m"] = None
            row["stop_segment_principal_percentile"] = None

    for seg_id, (start, end) in enumerate(ranges):
        if start < 0 or end > len(corrected) or end <= start:
            continue
        samples = _stop_segment_source_samples(
            corrected[start:end],
            forward[start:end],
            source=base_source,
        )
        samples = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
        finite = np.isfinite(samples).all(axis=1)
        if np.count_nonzero(finite) < max(1, int(min_epochs)):
            continue
        finite_samples = np.asarray(samples[finite], dtype=np.float64).reshape(-1, 3)
        center_source = "median"
        auto_median_radius = None
        auto_center_shift = None
        if center_mode in {"density", "auto", "auto_tail"}:
            center, density_radius, density_k = _density_stop_segment_center(
                samples,
                neighbors=density_neighbors,
            )
            if center is None:
                continue
            center_source = "density"
            if center_mode in {"auto", "auto_tail"}:
                median_center = np.median(finite_samples, axis=0)
                use_density, auto_median_radius, auto_center_shift = (
                    _auto_stop_segment_use_density(
                        finite_samples,
                        median_center,
                        center,
                        density_radius,
                    )
                )
                if not use_density:
                    use_principal = (
                        center_mode == "auto_tail"
                        and _auto_stop_segment_use_principal_tail(
                            median_radius=auto_median_radius,
                            density_radius=density_radius,
                            center_shift=auto_center_shift,
                        )
                    )
                    if use_principal:
                        principal_center = _principal_percentile_stop_segment_center(
                            finite_samples,
                            percentile=20.0,
                            reference_point=center,
                        )
                        if principal_center is not None:
                            center = principal_center
                            center_source = "principal"
                        else:
                            center = median_center
                            center_source = "median"
                    else:
                        center = median_center
                        center_source = "median"
        else:
            center = np.median(finite_samples, axis=0)
            density_radius = None
            density_k = None
        if not np.isfinite(center).all():
            continue
        segment_radius = float(
            np.percentile(np.linalg.norm(samples[finite] - center, axis=1), 90)
        )
        if max_radius_m is not None and segment_radius > float(max_radius_m):
            continue
        corrected[start:end] = (1.0 - alpha) * corrected[start:end] + alpha * center
        info["segments_applied"] = int(info["segments_applied"]) + 1
        info["epochs_applied"] = int(info["epochs_applied"]) + int(end - start)
        if center_source == "density":
            info["density_segments_selected"] = (
                int(info["density_segments_selected"]) + 1
            )
        if center_source == "principal":
            info["principal_segments_selected"] = (
                int(info["principal_segments_selected"]) + 1
            )
        if epoch_diagnostics is not None:
            for i in range(start, end):
                epoch_diagnostics[i]["stop_segment_constant_applied"] = True
                epoch_diagnostics[i]["stop_segment_id"] = int(seg_id)
                epoch_diagnostics[i]["stop_segment_radius_m"] = float(segment_radius)
                epoch_diagnostics[i]["stop_segment_density_radius_m"] = (
                    None if density_radius is None else float(density_radius)
                )
                epoch_diagnostics[i]["stop_segment_density_neighbors"] = (
                    None if density_k is None else int(density_k)
                )
                epoch_diagnostics[i]["stop_segment_center_mode"] = center_source
                epoch_diagnostics[i]["stop_segment_auto_median_radius_m"] = (
                    None if auto_median_radius is None else float(auto_median_radius)
                )
                epoch_diagnostics[i]["stop_segment_auto_center_shift_m"] = (
                    None if auto_center_shift is None else float(auto_center_shift)
                )
                epoch_diagnostics[i]["stop_segment_principal_percentile"] = (
                    20.0 if center_source == "principal" else None
                )
    return corrected, info
