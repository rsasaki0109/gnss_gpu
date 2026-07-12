"""WP12c telemetry signature analysis (step 2)."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_telemetry(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def segment_rms(rows: list[dict], ep_lo: int, ep_hi: int) -> tuple[int, float]:
    subset = [r for r in rows if ep_lo <= int(r["epoch"]) <= ep_hi]
    if not subset:
        return 0, float("nan")
    errs = [float(r["pos_err_m"]) for r in subset]
    import math

    return len(subset), math.sqrt(sum(e * e for e in errs) / len(errs))


def recovery_post_fire(rows: list[dict], post: int = 10) -> dict:
    """Mean pos_err at recovery fire and +post epochs after each fire."""
    fire_eps = [int(r["epoch"]) for r in rows if r.get("recovery_fired", "").lower() == "true"]
    at_fire: list[float] = []
    post_fire: list[float] = []
    by_ep = {int(r["epoch"]): float(r["pos_err_m"]) for r in rows}
    for ep in fire_eps:
        if ep in by_ep:
            at_fire.append(by_ep[ep])
        post_vals = [by_ep[ep + k] for k in range(1, post + 1) if ep + k in by_ep]
        if post_vals:
            post_fire.append(sum(post_vals) / len(post_vals))
    import statistics as stats

    return {
        "n_fires": len(fire_eps),
        "mean_at_fire_m": stats.mean(at_fire) if at_fire else float("nan"),
        "mean_post_m": stats.mean(post_fire) if post_fire else float("nan"),
        "n_post_samples": len(post_fire),
    }


def analyze(label: str, path: Path) -> dict:
    rows = load_telemetry(path)
    rec = recovery_post_fire(rows, post=10)
    n_drift, rms_drift = segment_rms(rows, 1000, 1499)
    n_early, rms_early = segment_rms(rows, 0, 999)
    n_open, rms_open = segment_rms(rows, 500, 799)
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "n_epochs": len(rows),
        "recovery": rec,
        "ep0_999_rms_m": rms_early,
        "ep500_799_rms_m": rms_open,
        "ep1000_1499_rms_m": rms_drift,
        "ep1000_1499_n": n_drift,
    }


def main() -> None:
    pairs = [
        ("WP12b persist (no schur)", ROOT / "results/wp12b/probe_persist_4000_telemetry.csv"),
        ("WP12c schur fixed win5", ROOT / "results/wp12c/probe_schur_fixed_4000_telemetry.csv"),
        ("WP12c schur fixed win10", ROOT / "results/wp12c/probe_schur_fixed_win10_4000_telemetry.csv"),
    ]
    for label, path in pairs:
        if not path.exists():
            print(f"SKIP {label}: missing {path}")
            continue
        r = analyze(label, path)
        rec = r["recovery"]
        print(f"\n=== {label} ===")
        print(f"  epochs: {r['n_epochs']}")
        print(f"  ep 0-999 RMS: {r['ep0_999_rms_m']:.2f} m")
        print(f"  ep 500-799 RMS: {r['ep500_799_rms_m']:.2f} m")
        print(f"  ep 1000-1499 RMS: {r['ep1000_1499_rms_m']:.2f} m (n={r['ep1000_1499_n']})")
        print(
            f"  recovery fires: {rec['n_fires']}; "
            f"mean @fire {rec['mean_at_fire_m']:.1f} m; "
            f"+10ep mean {rec['mean_post_m']:.1f} m (n={rec['n_post_samples']})"
        )


if __name__ == "__main__":
    main()
