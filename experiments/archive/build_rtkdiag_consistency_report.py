#!/usr/bin/env python3
"""Build an HTML report for RTKDiag candidate consistency diagnostics."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import plotly.graph_objects as go
from plotly.io import to_html


DEFAULT_LABELS = (
    "fgo_v14_snr38",
    "full_ratio15_lock3_trustedseed_rtkout3oGem3",
    "dev_demo5_trusted_o3",
    "n2_nobds",
    "fgo_v1",
    "full_ratio15_lock3_trustedseed_rtkout3mlc1",
    "full_ratio15_lock3_trustedseed_rtkout5",
    "libgnss_ext_subset",
)

CONSISTENCY_FIELDS = (
    "rtkdiag_selected_diag_status",
    "rtkdiag_selected_diag_sats",
    "rtkdiag_selected_diag_ratio",
    "rtkdiag_selected_diag_rms",
    "rtkdiag_selected_diag_abs_max",
    "rtkdiag_candidate_agreement_count_1m",
    "rtkdiag_candidate_agreement_count_3m",
    "rtkdiag_candidate_family_disagreement_m",
    "rtkdiag_candidate_family_span_m",
    "rtkdiag_selected_to_nearest_fixed_m",
    "rtkdiag_selected_to_fgo_v14_m",
    "rtkdiag_selected_to_fgo_v1_m",
    "rtkdiag_selected_to_prev_selected_m",
    "rtkdiag_selected_velocity_mps",
    "rtkdiag_selected_to_tdcp_velocity_mps",
)


def _float(value: object, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _int(value: object, default: int = 0) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def _tow_key(value: object) -> float:
    return round(_float(value), 1)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _select_method(rows: list[dict[str, str]], method_contains: str) -> list[dict[str, str]]:
    selected = [row for row in rows if method_contains in str(row.get("method", ""))]
    return selected if selected else rows


def _finite(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def _median(values: list[float]) -> float:
    vals = _finite(values)
    return float(median(vals)) if vals else float("nan")


def _p95(values: list[float]) -> float:
    vals = sorted(_finite(values))
    if not vals:
        return float("nan")
    idx = int(math.ceil(0.95 * len(vals))) - 1
    return vals[max(0, min(idx, len(vals) - 1))]


def _fmt(value: object, digits: int = 3) -> str:
    number = _float(value)
    if math.isfinite(number):
        return f"{number:.{digits}f}"
    return html.escape(str(value)) if value not in (None, "") else ""


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _dist(a: tuple[float, float, float] | None, b: tuple[float, float, float] | None) -> float:
    if a is None or b is None:
        return float("nan")
    return math.sqrt(sum((aa - bb) ** 2 for aa, bb in zip(a, b)))


def _xyz_from_row(row: dict[str, str], prefix: str) -> tuple[float, float, float] | None:
    xyz = (
        _float(row.get(f"{prefix}_x")),
        _float(row.get(f"{prefix}_y")),
        _float(row.get(f"{prefix}_z")),
    )
    return xyz if all(math.isfinite(value) for value in xyz) else None


def _ref_from_row(row: dict[str, str]) -> tuple[float, float, float] | None:
    xyz = (_float(row.get("ref_x")), _float(row.get("ref_y")), _float(row.get("ref_z")))
    return xyz if all(math.isfinite(value) for value in xyz) else None


def _load_pos(path: Path) -> dict[float, tuple[float, float, float]]:
    out: dict[float, tuple[float, float, float]] = {}
    if not path.exists():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            xyz = (_float(parts[2]), _float(parts[3]), _float(parts[4]))
            if all(math.isfinite(value) for value in xyz):
                out[_tow_key(parts[1])] = xyz
    return out


def _load_diag(path: Path) -> dict[float, dict[str, str]]:
    if not path.exists():
        return {}
    return {_tow_key(row.get("tow")): row for row in _read_csv(path)}


def _load_candidates(
    base_dir: Path,
    labels: tuple[str, ...],
    city: str,
    run: str,
) -> dict[str, dict[str, Any]]:
    stem = f"{city}_{run}_full"
    out: dict[str, dict[str, Any]] = {}
    for label in labels:
        label_dir = base_dir / label
        out[label] = {
            "pos": _load_pos(label_dir / f"{stem}.pos"),
            "diag": _load_diag(label_dir / f"{stem}.csv"),
        }
    return out


def _candidate_snapshot(
    row: dict[str, str],
    candidates: dict[str, dict[str, Any]],
) -> list[dict[str, object]]:
    tow = _tow_key(row.get("tow"))
    ref = _ref_from_row(row)
    selected = str(row.get("rtkdiag_selected_base_label") or row.get("rtkdiag_selected_label") or "")
    rows: list[dict[str, object]] = []
    for label, loaded in candidates.items():
        pos = loaded["pos"].get(tow)
        diag = loaded["diag"].get(tow)
        rows.append(
            {
                "label": label,
                "selected": label == selected,
                "to_ref_m": _dist(pos, ref),
                "status": _int(diag.get("final_status")) if diag else "",
                "sats": _int(diag.get("final_sats")) if diag else "",
                "ratio": _float(diag.get("final_ratio")) if diag else float("nan"),
                "rms": _float(diag.get("final_residual_rms")) if diag else float("nan"),
                "abs_max": _float(diag.get("final_residual_abs_max")) if diag else float("nan"),
                "update_rows": _float(diag.get("final_update_rows")) if diag else float("nan"),
                "output_added": _int(diag.get("output_added")) if diag else "",
            }
        )
    rows.sort(
        key=lambda item: (
            0 if item["selected"] else 1,
            _float(item["to_ref_m"]),
            _float(item["rms"]),
        )
    )
    return rows


def _summary(rows: list[dict[str, str]]) -> dict[str, object]:
    errors = [_float(row.get("emit_to_ref_m")) for row in rows]
    labels = Counter(str(row.get("rtkdiag_selected_base_label") or row.get("rtkdiag_selected_label") or "") for row in rows)
    sources = Counter(str(row.get("emitted_source", "")) for row in rows)
    fail3 = sum(1 for value in errors if math.isfinite(value) and value > 3.0)
    tail20 = sum(1 for value in errors if math.isfinite(value) and value >= 20.0)
    return {
        "epochs": len(rows),
        "pass_0p5": sum(1 for value in errors if math.isfinite(value) and value <= 0.5),
        "fail_3m": fail3,
        "tail_20m": tail20,
        "median": _median(errors),
        "p95": _p95(errors),
        "max": max(_finite(errors)) if _finite(errors) else float("nan"),
        "labels": labels,
        "sources": sources,
    }


def _failure_spans(rows: list[dict[str, str]], threshold_m: float) -> list[dict[str, object]]:
    failed = [
        (idx, row)
        for idx, row in enumerate(rows)
        if _float(row.get("emit_to_ref_m")) > float(threshold_m)
    ]
    if not failed:
        return []
    spans: list[dict[str, object]] = []
    start = 0
    span_id = 1
    for pos in range(1, len(failed) + 1):
        contiguous = pos < len(failed) and _int(failed[pos][1].get("epoch")) == _int(failed[pos - 1][1].get("epoch")) + 1
        label = str(failed[pos][1].get("rtkdiag_selected_base_label", "")) if pos < len(failed) else ""
        prev_label = str(failed[pos - 1][1].get("rtkdiag_selected_base_label", "")) if pos < len(failed) else ""
        if contiguous and label == prev_label:
            continue
        chunk = [row for _idx, row in failed[start:pos]]
        errors = [_float(row.get("emit_to_ref_m")) for row in chunk]
        labels = Counter(str(row.get("rtkdiag_selected_base_label") or row.get("rtkdiag_selected_label") or "") for row in chunk)
        spans.append(
            {
                "span_id": span_id,
                "start_epoch": _int(chunk[0].get("epoch")),
                "end_epoch": _int(chunk[-1].get("epoch")),
                "start_tow": _float(chunk[0].get("tow")),
                "end_tow": _float(chunk[-1].get("tow")),
                "n_epochs": len(chunk),
                "top_label": labels.most_common(1)[0][0] if labels else "",
                "median_error_m": _median(errors),
                "max_error_m": max(_finite(errors)) if _finite(errors) else float("nan"),
                "median_family_disagreement_m": _median(
                    [_float(row.get("rtkdiag_candidate_family_disagreement_m")) for row in chunk]
                ),
                "median_velocity_mps": _median(
                    [_float(row.get("rtkdiag_selected_velocity_mps")) for row in chunk]
                ),
                "median_tdcp_velocity_disagreement_mps": _median(
                    [_float(row.get("rtkdiag_selected_to_tdcp_velocity_mps")) for row in chunk]
                ),
            }
        )
        span_id += 1
        start = pos
    spans.sort(key=lambda row: (float(row["max_error_m"]), int(row["n_epochs"])), reverse=True)
    return spans


def _plot_time_series(rows: list[dict[str, str]], title: str) -> str:
    x = [_float(row.get("tow")) for row in rows]
    err = [_float(row.get("emit_to_ref_m")) for row in rows]
    family = [_float(row.get("rtkdiag_candidate_family_disagreement_m")) for row in rows]
    vel = [_float(row.get("rtkdiag_selected_velocity_mps")) for row in rows]
    tdcp_vel = [_float(row.get("rtkdiag_selected_to_tdcp_velocity_mps")) for row in rows]
    labels = [str(row.get("rtkdiag_selected_base_label") or row.get("rtkdiag_selected_label") or "") for row in rows]
    hover = [
        f"epoch={_escape(row.get('epoch'))}<br>tow={_fmt(row.get('tow'), 1)}"
        f"<br>label={_escape(labels[idx])}<br>err={_fmt(row.get('emit_to_ref_m'))}m"
        for idx, row in enumerate(rows)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=err, name="3D error", mode="lines", text=hover, hoverinfo="text+y"))
    fig.add_trace(go.Scatter(x=x, y=family, name="family disagreement", mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=vel, name="selected velocity", mode="lines"))
    fig.add_trace(go.Scatter(x=x, y=tdcp_vel, name="velocity disagreement", mode="lines"))
    fig.add_hline(y=3.0, line_dash="dot", line_color="#c2410c")
    fig.add_hline(y=20.0, line_dash="dot", line_color="#991b1b")
    fig.update_layout(
        title=title,
        xaxis_title="TOW",
        yaxis_title="meters / mps",
        height=430,
        margin=dict(l=48, r=24, t=56, b=44),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return to_html(fig, include_plotlyjs="cdn", full_html=False)


def _plot_enu(rows: list[dict[str, str]], title: str) -> str:
    east = [_float(row.get("emit_err_e_m")) for row in rows]
    north = [_float(row.get("emit_err_n_m")) for row in rows]
    err = [_float(row.get("emit_to_ref_m")) for row in rows]
    labels = [str(row.get("rtkdiag_selected_base_label") or row.get("rtkdiag_selected_label") or "") for row in rows]
    hover = [
        f"epoch={_escape(row.get('epoch'))}<br>tow={_fmt(row.get('tow'), 1)}"
        f"<br>label={_escape(labels[idx])}<br>err={_fmt(row.get('emit_to_ref_m'))}m"
        for idx, row in enumerate(rows)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=east,
            y=north,
            mode="markers",
            marker=dict(size=6, color=err, colorscale="Viridis", showscale=True, colorbar=dict(title="3D m")),
            text=hover,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="East error m",
        yaxis_title="North error m",
        height=430,
        margin=dict(l=48, r=24, t=56, b=44),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return to_html(fig, include_plotlyjs=False, full_html=False)


def _table(headers: list[str], rows: list[list[object]], *, class_name: str = "") -> str:
    head = "".join(f"<th>{_escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in row) + "</tr>")
    return f"<table class=\"{class_name}\"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _summary_cards(label: str, summary: dict[str, object]) -> str:
    labels = summary["labels"]
    assert isinstance(labels, Counter)
    top_labels = ", ".join(f"{key}:{value}" for key, value in labels.most_common(5) if key)
    return (
        f"<section class=\"band\"><h2>{_escape(label)}</h2><div class=\"cards\">"
        f"<div><b>{summary['epochs']}</b><span>epochs</span></div>"
        f"<div><b>{summary['pass_0p5']}</b><span>pass <=0.5m</span></div>"
        f"<div><b>{summary['fail_3m']}</b><span>fail >3m</span></div>"
        f"<div><b>{summary['tail_20m']}</b><span>tail >=20m</span></div>"
        f"<div><b>{_fmt(summary['median'])}</b><span>median 3D m</span></div>"
        f"<div><b>{_fmt(summary['p95'])}</b><span>p95 3D m</span></div>"
        f"</div><p class=\"muted\">Top selected labels: {_escape(top_labels)}</p></section>"
    )


def _comparison_rows(
    base_rows: list[dict[str, str]],
    compare_rows: list[dict[str, str]],
    *,
    tail_threshold_m: float,
) -> list[list[object]]:
    compare_by_epoch = {_int(row.get("epoch")): row for row in compare_rows}
    interesting = [
        row
        for row in base_rows
        if _float(row.get("emit_to_ref_m")) >= tail_threshold_m
        or _float(row.get("rtkdiag_candidate_family_disagreement_m")) >= 10.0
        or _float(row.get("rtkdiag_selected_velocity_mps")) >= 30.0
        or _float(row.get("rtkdiag_selected_to_tdcp_velocity_mps")) >= 30.0
    ]
    out: list[list[object]] = []
    for row in interesting[:80]:
        cmp_row = compare_by_epoch.get(_int(row.get("epoch")), {})
        out.append(
            [
                _int(row.get("epoch")),
                _fmt(row.get("tow"), 1),
                row.get("rtkdiag_selected_base_label") or row.get("rtkdiag_selected_label"),
                _fmt(row.get("emit_to_ref_m")),
                cmp_row.get("rtkdiag_selected_base_label") or cmp_row.get("rtkdiag_selected_label", ""),
                _fmt(cmp_row.get("emit_to_ref_m")),
                _fmt(row.get("rtkdiag_candidate_family_disagreement_m")),
                _fmt(row.get("rtkdiag_selected_velocity_mps")),
                _fmt(row.get("rtkdiag_selected_to_tdcp_velocity_mps")),
            ]
        )
    return out


def _candidate_tables(
    rows: list[dict[str, str]],
    candidates: dict[str, dict[str, Any]],
    *,
    tail_threshold_m: float,
    max_epochs: int,
) -> str:
    interesting = [
        row
        for row in rows
        if _float(row.get("emit_to_ref_m")) >= tail_threshold_m
        or (
            str(row.get("rtkdiag_selected_base_label")) in {"libgnss_ext_subset", "dev_demo5_trusted_o3"}
            and _float(row.get("emit_to_ref_m")) > 3.0
        )
    ][:max_epochs]
    parts: list[str] = []
    for row in interesting:
        snapshot = _candidate_snapshot(row, candidates)
        table_rows = [
            [
                "*" if item["selected"] else "",
                item["label"],
                _fmt(item["to_ref_m"]),
                item["status"],
                item["sats"],
                _fmt(item["ratio"], 2),
                _fmt(item["rms"]),
                _fmt(item["abs_max"]),
                _fmt(item["update_rows"], 1),
                item["output_added"],
            ]
            for item in snapshot
        ]
        parts.append(
            f"<details open><summary>epoch {_int(row.get('epoch'))}, tow {_fmt(row.get('tow'), 1)}, "
            f"selected {_escape(row.get('rtkdiag_selected_base_label'))}, "
            f"err {_fmt(row.get('emit_to_ref_m'))}m</summary>"
            + _table(
                ["sel", "label", "to_ref_m", "status", "sats", "ratio", "rms", "abs_max", "rows", "out"],
                table_rows,
                class_name="dense",
            )
            + "</details>"
        )
    return "".join(parts) if parts else "<p class=\"muted\">No tail candidate epochs selected.</p>"


def _write_html(
    path: Path,
    *,
    title: str,
    base_label: str,
    base_rows: list[dict[str, str]],
    compare_label: str,
    compare_rows: list[dict[str, str]],
    candidates: dict[str, dict[str, Any]],
    tail_threshold_m: float,
) -> None:
    base_summary = _summary(base_rows)
    compare_summary = _summary(compare_rows) if compare_rows else None
    spans = _failure_spans(base_rows, threshold_m=3.0)
    span_rows = [
        [
            row["span_id"],
            row["start_epoch"],
            row["end_epoch"],
            row["n_epochs"],
            row["top_label"],
            _fmt(row["median_error_m"]),
            _fmt(row["max_error_m"]),
            _fmt(row["median_family_disagreement_m"]),
            _fmt(row["median_velocity_mps"]),
            _fmt(row["median_tdcp_velocity_disagreement_mps"]),
        ]
        for row in spans[:30]
    ]
    comparison = ""
    if compare_rows:
        comparison = (
            "<section class=\"band\"><h2>Before / After</h2>"
            + _table(
                [
                    "epoch",
                    "tow",
                    "before_label",
                    "before_err",
                    "after_label",
                    "after_err",
                    "family_disagree",
                    "velocity",
                    "tdcp_vel_diff",
                ],
                _comparison_rows(base_rows, compare_rows, tail_threshold_m=tail_threshold_m),
                class_name="dense",
            )
            + "</section>"
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>
body {{ margin:0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:#172033; background:#f6f7f9; }}
header {{ padding:28px 32px 18px; background:#ffffff; border-bottom:1px solid #d9dee7; }}
h1 {{ margin:0; font-size:24px; font-weight:700; letter-spacing:0; }}
h2 {{ margin:0 0 14px; font-size:18px; }}
.muted {{ color:#5e6a7d; }}
.band {{ padding:22px 32px; border-bottom:1px solid #d9dee7; background:#ffffff; }}
.plot {{ padding:18px 24px; background:#ffffff; border-bottom:1px solid #d9dee7; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:10px; }}
.cards div {{ border:1px solid #d9dee7; border-radius:6px; padding:12px; background:#fafbfc; }}
.cards b {{ display:block; font-size:20px; }}
.cards span {{ color:#5e6a7d; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; background:#ffffff; }}
th, td {{ padding:8px 10px; border-bottom:1px solid #e3e7ee; text-align:left; white-space:nowrap; }}
th {{ position:sticky; top:0; background:#eef2f6; z-index:1; }}
.dense {{ font-size:12px; }}
.tablewrap {{ max-height:460px; overflow:auto; border:1px solid #d9dee7; border-radius:6px; }}
details {{ margin:0 0 12px; border:1px solid #d9dee7; border-radius:6px; background:#ffffff; }}
summary {{ cursor:pointer; padding:10px 12px; font-weight:600; }}
</style>
</head>
<body>
<header>
<h1>{_escape(title)}</h1>
<p class="muted">RTKDiag candidate consistency report. Error-to-reference fields are post-hoc diagnostics; guard fields are runtime/non-GT.</p>
</header>
{_summary_cards(base_label, base_summary)}
{_summary_cards(compare_label, compare_summary) if compare_summary is not None else ""}
<section class="plot">{_plot_time_series(base_rows, base_label + " time series")}</section>
<section class="plot">{_plot_enu(base_rows, base_label + " ENU error scatter")}</section>
{comparison}
<section class="band"><h2>Failure Spans</h2><div class="tablewrap">
{_table(["span", "start", "end", "n", "label", "median_err", "max_err", "family_disagree", "velocity", "tdcp_vel_diff"], span_rows, class_name="dense")}
</div></section>
<section class="band"><h2>Candidate Snapshots</h2>
{_candidate_tables(base_rows, candidates, tail_threshold_m=tail_threshold_m, max_epochs=20)}
</section>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RTKDiag consistency HTML report")
    parser.add_argument("internal_epochs_csv", type=Path)
    parser.add_argument("--compare-internal-epochs-csv", type=Path, default=None)
    parser.add_argument("--candidate-base-dir", type=Path, default=Path("experiments/results/libgnss_diag_phase10"))
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--city", default="nagoya")
    parser.add_argument("--run", default="run2")
    parser.add_argument("--method-contains", default="rtkdiag_pf")
    parser.add_argument("--title", default="RTKDiag Candidate Consistency")
    parser.add_argument("--base-label", default="baseline")
    parser.add_argument("--compare-label", default="guard")
    parser.add_argument("--tail-threshold-m", type=float, default=20.0)
    parser.add_argument("--out-html", type=Path, default=Path("experiments/results/rtkdiag_consistency_report.html"))
    args = parser.parse_args()

    base_rows = _select_method(_read_csv(args.internal_epochs_csv), args.method_contains)
    compare_rows: list[dict[str, str]] = []
    if args.compare_internal_epochs_csv is not None:
        compare_rows = _select_method(_read_csv(args.compare_internal_epochs_csv), args.method_contains)
    labels = tuple(label.strip() for label in str(args.labels).split(",") if label.strip())
    candidates = _load_candidates(args.candidate_base_dir, labels, args.city, args.run)
    _write_html(
        args.out_html,
        title=args.title,
        base_label=args.base_label,
        base_rows=base_rows,
        compare_label=args.compare_label,
        compare_rows=compare_rows,
        candidates=candidates,
        tail_threshold_m=float(args.tail_threshold_m),
    )
    print(f"wrote {args.out_html}")


if __name__ == "__main__":
    main()
