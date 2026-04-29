#!/usr/bin/env python3
"""Render a self-contained HTML dashboard from the product deliverable CSVs.

Output: `internal_docs/product_deliverable/dashboard.html`

No JavaScript dependencies: inline SVG, CSS grid, and a static HTML
table.  Opens correctly from the file system in any modern browser.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DELIVERABLE_DIR = REPO_ROOT / "internal_docs" / "product_deliverable"
DEFAULT_ROUTE_CSV = DELIVERABLE_DIR / "route_level_fix_rate_prediction.csv"
DEFAULT_WINDOW_CSV = DELIVERABLE_DIR / "window_level_details.csv"
DEFAULT_OUTPUT = DELIVERABLE_DIR / "dashboard.html"


TIER_COLOR = {
    "high": "#2ecc71",
    "medium": "#f39c12",
    "low": "#e74c3c",
}

FOCUS_COLOR = {
    "": "#95a5a6",
    "false_high": "#c0392b",
    "hidden_high": "#8e44ad",
    "false_lift": "#d35400",
    "false_lift_mild": "#e67e22",
    "false_lift_resolved": "#27ae60",
}


def _render_per_run_bar(route_df: pd.DataFrame) -> str:
    # One bar group per run: actual vs predicted side by side
    width = 720
    height = 280
    pad_left = 90
    pad_right = 20
    pad_top = 30
    pad_bottom = 60
    chart_w = width - pad_left - pad_right
    chart_h = height - pad_top - pad_bottom
    n = len(route_df)
    group_w = chart_w / max(n, 1)
    bar_w = (group_w - 10) / 2
    max_val = max(route_df[["actual_fix_rate_pct", "adopted_pred_fix_rate_pct"]].max().max(), 35.0)

    svg = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    # axes
    svg.append(f'<line x1="{pad_left}" y1="{pad_top}" x2="{pad_left}" y2="{pad_top+chart_h}" stroke="#333" />')
    svg.append(f'<line x1="{pad_left}" y1="{pad_top+chart_h}" x2="{pad_left+chart_w}" y2="{pad_top+chart_h}" stroke="#333" />')
    # gridlines and y labels
    for tick in range(0, int(max_val) + 1, 10):
        y = pad_top + chart_h - (tick / max_val) * chart_h
        svg.append(f'<line x1="{pad_left}" y1="{y:.1f}" x2="{pad_left+chart_w}" y2="{y:.1f}" stroke="#eee" />')
        svg.append(f'<text x="{pad_left-8}" y="{y+4:.1f}" font-size="11" text-anchor="end" fill="#555">{tick}</text>')
    svg.append(f'<text x="20" y="{pad_top+chart_h/2}" font-size="12" fill="#555" transform="rotate(-90 20 {pad_top+chart_h/2})" text-anchor="middle">FIX rate (%)</text>')

    for i, row in enumerate(route_df.itertuples(index=False)):
        gx = pad_left + i * group_w + 5
        # actual
        a_h = (row.actual_fix_rate_pct / max_val) * chart_h
        svg.append(f'<rect x="{gx:.1f}" y="{pad_top+chart_h-a_h:.1f}" width="{bar_w:.1f}" height="{a_h:.1f}" fill="#34495e" />')
        # predicted
        p_h = (row.adopted_pred_fix_rate_pct / max_val) * chart_h
        tier_fill = TIER_COLOR.get(row.confidence_tier, "#aaa")
        svg.append(f'<rect x="{gx+bar_w+2:.1f}" y="{pad_top+chart_h-p_h:.1f}" width="{bar_w:.1f}" height="{p_h:.1f}" fill="{tier_fill}" />')
        # label
        svg.append(f'<text x="{gx+bar_w:.1f}" y="{pad_top+chart_h+15}" font-size="11" text-anchor="middle" fill="#333">{html.escape(str(row.city))} {html.escape(str(row.run))}</text>')
        svg.append(f'<text x="{gx+bar_w:.1f}" y="{pad_top+chart_h+32}" font-size="10" text-anchor="middle" fill="#777">|err|={row.adopted_abs_error_pp:.1f} pp</text>')

    # legend
    svg.append(f'<rect x="{pad_left}" y="8" width="10" height="10" fill="#34495e" />')
    svg.append(f'<text x="{pad_left+14}" y="17" font-size="11" fill="#333">actual</text>')
    svg.append(f'<rect x="{pad_left+70}" y="8" width="10" height="10" fill="#2ecc71" />')
    svg.append(f'<text x="{pad_left+84}" y="17" font-size="11" fill="#333">predicted (high)</text>')
    svg.append(f'<rect x="{pad_left+180}" y="8" width="10" height="10" fill="#f39c12" />')
    svg.append(f'<text x="{pad_left+194}" y="17" font-size="11" fill="#333">predicted (medium)</text>')
    svg.append(f'<rect x="{pad_left+310}" y="8" width="10" height="10" fill="#e74c3c" />')
    svg.append(f'<text x="{pad_left+324}" y="17" font-size="11" fill="#333">predicted (low)</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def _render_window_scatter(window_df: pd.DataFrame) -> str:
    width = 720
    height = 520
    pad = 60
    chart_w = width - 2 * pad
    chart_h = height - 2 * pad
    svg = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']

    # axes
    svg.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+chart_h}" stroke="#333" />')
    svg.append(f'<line x1="{pad}" y1="{pad+chart_h}" x2="{pad+chart_w}" y2="{pad+chart_h}" stroke="#333" />')

    # Reference y=x
    svg.append(f'<line x1="{pad}" y1="{pad+chart_h}" x2="{pad+chart_w}" y2="{pad}" stroke="#aaa" stroke-dasharray="4,4" />')

    # gridlines
    for tick in range(0, 101, 20):
        x = pad + (tick / 100.0) * chart_w
        y = pad + chart_h - (tick / 100.0) * chart_h
        svg.append(f'<line x1="{x:.1f}" y1="{pad}" x2="{x:.1f}" y2="{pad+chart_h}" stroke="#f0f0f0" />')
        svg.append(f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+chart_w}" y2="{y:.1f}" stroke="#f0f0f0" />')
        svg.append(f'<text x="{x:.1f}" y="{pad+chart_h+14}" font-size="10" text-anchor="middle" fill="#555">{tick}</text>')
        svg.append(f'<text x="{pad-5}" y="{y+4:.1f}" font-size="10" text-anchor="end" fill="#555">{tick}</text>')

    # Axis labels
    svg.append(f'<text x="{pad+chart_w/2}" y="{height-15}" font-size="12" text-anchor="middle" fill="#333">actual FIX rate (%)</text>')
    svg.append(f'<text x="20" y="{pad+chart_h/2}" font-size="12" fill="#333" transform="rotate(-90 20 {pad+chart_h/2})" text-anchor="middle">adopted prediction (%)</text>')

    for _, row in window_df.iterrows():
        cx = pad + (row.actual_fix_rate_pct / 100.0) * chart_w
        cy = pad + chart_h - (row.adopted_pred_fix_rate_pct / 100.0) * chart_h
        tag = str(row.focus_case_tag) if row.focus_case_tag and row.focus_case_tag != "nan" else ""
        fill = FOCUS_COLOR.get(tag, FOCUS_COLOR[""])
        radius = 5 if tag else 2.5
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{fill}" opacity="0.75" />')

    # Legend
    legend_y = 18
    legend_entries = [
        ("", "baseline window"),
        ("false_high", "false high (actual 0%)"),
        ("hidden_high", "hidden high (actual 100%)"),
        ("false_lift", "false lift"),
        ("false_lift_mild", "mild false lift"),
        ("false_lift_resolved", "resolved false lift"),
    ]
    for i, (tag, label) in enumerate(legend_entries):
        lx = pad + i * 115
        color = FOCUS_COLOR.get(tag, FOCUS_COLOR[""])
        svg.append(f'<circle cx="{lx}" cy="{legend_y}" r="4" fill="{color}" />')
        svg.append(f'<text x="{lx+8}" y="{legend_y+4}" font-size="10" fill="#333">{html.escape(label)}</text>')

    svg.append('</svg>')
    return "\n".join(svg)


def _render_table(route_df: pd.DataFrame) -> str:
    rows = []
    rows.append("<table><thead><tr>")
    for col in ["city", "run", "actual_fix_rate_pct", "adopted_pred_fix_rate_pct", "adopted_abs_error_pp", "confidence_tier", "focus_case_window_count"]:
        rows.append(f'<th>{html.escape(col)}</th>')
    rows.append("<th>note</th></tr></thead><tbody>")
    for _, r in route_df.iterrows():
        tier = r["confidence_tier"]
        color = TIER_COLOR.get(tier, "#aaa")
        rows.append("<tr>")
        rows.append(f'<td>{html.escape(str(r["city"]))}</td>')
        rows.append(f'<td>{html.escape(str(r["run"]))}</td>')
        rows.append(f'<td>{r["actual_fix_rate_pct"]:.2f} %</td>')
        rows.append(f'<td>{r["adopted_pred_fix_rate_pct"]:.2f} %</td>')
        rows.append(f'<td>{r["adopted_abs_error_pp"]:.2f} pp</td>')
        rows.append(f'<td><span class="tier" style="background:{color}">{html.escape(str(tier))}</span></td>')
        rows.append(f'<td>{int(r["focus_case_window_count"])}</td>')
        rows.append(f'<td>{html.escape(str(r["confidence_note"]))}</td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _render_focus_cases(window_df: pd.DataFrame) -> str:
    focus = window_df[window_df["focus_case_tag"].astype(str) != ""].sort_values(["city", "run", "window_index"])
    if focus.empty:
        return "<p>No focus-case windows detected.</p>"
    rows = ["<table><thead><tr>"]
    for col in ["city", "run", "window_index", "actual_fix_rate_pct", "adopted_pred_fix_rate_pct", "abs_error_pp", "focus_case_tag"]:
        rows.append(f'<th>{html.escape(col)}</th>')
    rows.append("<th>note</th></tr></thead><tbody>")
    for _, r in focus.iterrows():
        tag = str(r["focus_case_tag"])
        color = FOCUS_COLOR.get(tag, "#aaa")
        rows.append("<tr>")
        rows.append(f'<td>{html.escape(str(r["city"]))}</td>')
        rows.append(f'<td>{html.escape(str(r["run"]))}</td>')
        rows.append(f'<td>{int(r["window_index"])}</td>')
        rows.append(f'<td>{r["actual_fix_rate_pct"]:.2f} %</td>')
        rows.append(f'<td>{r["adopted_pred_fix_rate_pct"]:.2f} %</td>')
        rows.append(f'<td>{r["abs_error_pp"]:.2f} pp</td>')
        rows.append(f'<td><span class="tier" style="background:{color}">{html.escape(tag)}</span></td>')
        rows.append(f'<td>{html.escape(str(r["focus_case_note"]))}</td>')
        rows.append("</tr>")
    rows.append("</tbody></table>")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render product dashboard HTML")
    parser.add_argument("--route-csv", type=Path, default=DEFAULT_ROUTE_CSV)
    parser.add_argument("--window-csv", type=Path, default=DEFAULT_WINDOW_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_df = pd.read_csv(args.route_csv)
    window_df = pd.read_csv(args.window_csv).fillna({"focus_case_tag": "", "focus_case_note": ""})

    # aggregate stats
    total_windows = len(window_df)
    focus_windows = int((window_df["focus_case_tag"] != "").sum())
    agg_actual = float((window_df["actual_fix_rate_pct"] * 1.0).mean())
    agg_pred = float((window_df["adopted_pred_fix_rate_pct"] * 1.0).mean())
    wmae = float((window_df["adopted_pred_fix_rate_pct"] - window_df["actual_fix_rate_pct"]).abs().mean())
    run_mae = float(route_df["adopted_abs_error_pp"].mean())
    tier_counts = route_df["confidence_tier"].value_counts().to_dict()

    # Assemble HTML
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>PPC demo5 FIX-Rate Predictor — Dashboard</title>",
        "<style>",
        "body { font-family: system-ui, -apple-system, sans-serif; max-width: 1000px; margin: 30px auto; padding: 0 20px; color: #222; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 6px; margin-top: 36px; }",
        ".summary { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 20px 0; }",
        ".summary .cell { background: #ecf0f1; padding: 14px; border-radius: 6px; text-align: center; }",
        ".summary .cell .label { color: #7f8c8d; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }",
        ".summary .cell .value { font-size: 24px; font-weight: 600; color: #2c3e50; margin-top: 4px; }",
        "table { border-collapse: collapse; width: 100%; margin: 12px 0 20px; }",
        "th { background: #34495e; color: white; text-align: left; padding: 8px 10px; font-weight: 500; font-size: 13px; }",
        "td { border-bottom: 1px solid #ecf0f1; padding: 8px 10px; font-size: 13px; }",
        "tr:hover td { background: #f8f9fa; }",
        ".tier { display: inline-block; padding: 2px 10px; border-radius: 10px; color: white; font-size: 11px; font-weight: 600; }",
        "p { color: #555; line-height: 1.5; }",
        ".metadata { color: #7f8c8d; font-size: 12px; margin-bottom: 20px; }",
        "</style></head><body>",
        "<h1>PPC demo5 FIX-Rate Predictor — Dashboard</h1>",
        f"<p class='metadata'>Adopted model: <code>current_tight_hold + carry + α=0.75 + isotonic</code> · strict nested LORO on 6 runs / {total_windows} windows</p>",
        "<div class='summary'>",
        f"<div class='cell'><div class='label'>run MAE</div><div class='value'>{run_mae:.2f} pp</div></div>",
        f"<div class='cell'><div class='label'>window MAE</div><div class='value'>{wmae:.2f} pp</div></div>",
        f"<div class='cell'><div class='label'>aggregate actual</div><div class='value'>{agg_actual:.2f} %</div></div>",
        f"<div class='cell'><div class='label'>aggregate predicted</div><div class='value'>{agg_pred:.2f} %</div></div>",
        "</div>",
        "<p>",
        f"Confidence tiers: high={tier_counts.get('high', 0)} · medium={tier_counts.get('medium', 0)} · low={tier_counts.get('low', 0)} &nbsp;|&nbsp;",
        f"Focus-case windows: {focus_windows} / {total_windows}",
        "</p>",
        "<h2>Route-level predictions</h2>",
        _render_table(route_df),
        "<h2>Actual vs predicted (per run)</h2>",
        _render_per_run_bar(route_df),
        "<h2>Window-level scatter (actual vs predicted)</h2>",
        "<p>The dashed diagonal is the ideal y=x line.  Colored markers are focus-case windows documented in README §5.  Baseline windows are small gray dots.</p>",
        _render_window_scatter(window_df),
        "<h2>Focus-case windows detail</h2>",
        _render_focus_cases(window_df),
        "<h2>Interpretation cheatsheet</h2>",
        "<ul>",
        "<li><b>high</b> tier (green): trust the route-level prediction directly; error expected &le; 3 pp.</li>",
        "<li><b>medium</b> tier (orange): use with caution near decision boundaries; error 3-8 pp expected.</li>",
        "<li><b>low</b> tier (red): contains focus-case windows.  Drill into the scatter / focus table before acting.</li>",
        "<li>False-high markers (red dots on bottom-left): adopted model partially suppresses but residual inflation remains (Tokyo run2 w7/w9).</li>",
        "<li>Hidden-high markers (purple on top-left): adopted model lifts but still undershoots (Tokyo run2 w23-w27).</li>",
        "<li>False-lift markers (orange, upper-left): adopted model partially rejects (Nagoya run2) or still over-predicts (Tokyo run3 w17).</li>",
        "</ul>",
        "<p class='metadata'>See <a href='README.md'>README.md</a> for model scope and <a href='RUNBOOK.md'>RUNBOOK.md</a> for operational procedure.</p>",
        "</body></html>",
    ]
    html_doc = "\n".join(parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"saved: {args.output}")
    print(f"  {total_windows} windows, {focus_windows} focus cases, tiers={tier_counts}")


if __name__ == "__main__":
    main()
