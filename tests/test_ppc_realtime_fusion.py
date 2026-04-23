# ruff: noqa: E402
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import exp_ppc_realtime_fusion as fusion


class _FakeWidelaneComputer:
    def compute_dd(self, *_args, **_kwargs):
        return object(), SimpleNamespace(reason="ok", n_candidate_pairs=6, n_fixed_pairs=6, fix_rate=1.0, n_dd=6)


def test_try_widelane_anchor_vetoes_mid_residual_band(monkeypatch):
    monkeypatch.setattr(fusion, "_dd_measurements", lambda *_args, **_kwargs: [])
    stats = fusion._DDAnchorStats(
        accepted=True,
        n_dd=6,
        kept_pairs=5,
        shift_m=0.7,
        robust_rms_m=0.3,
    )
    monkeypatch.setattr(
        fusion,
        "_robust_dd_pr_anchor",
        lambda *_args, **_kwargs: (np.ones(3), stats),
    )

    anchor, anchor_stats, wl_stats = fusion._try_widelane_anchor(
        _FakeWidelaneComputer(),
        {"times": [10.0]},
        0,
        np.zeros(3),
        huber_k_m=1.0,
        trim_m=1.5,
        min_kept_pairs=3,
        max_shift_m=5.0,
        max_robust_rms_m=0.8,
        veto_rms_band_min_m=0.15,
        veto_rms_band_max_m=0.35,
        veto_min_kept_pairs=4,
    )

    assert anchor is None
    assert anchor_stats is stats
    assert wl_stats.reason == "ok"


def test_try_widelane_anchor_keeps_low_residual_fix(monkeypatch):
    monkeypatch.setattr(fusion, "_dd_measurements", lambda *_args, **_kwargs: [])
    stats = fusion._DDAnchorStats(
        accepted=True,
        n_dd=6,
        kept_pairs=5,
        shift_m=0.7,
        robust_rms_m=0.03,
    )
    monkeypatch.setattr(
        fusion,
        "_robust_dd_pr_anchor",
        lambda *_args, **_kwargs: (np.ones(3), stats),
    )

    anchor, anchor_stats, _wl_stats = fusion._try_widelane_anchor(
        _FakeWidelaneComputer(),
        {"times": [10.0]},
        0,
        np.zeros(3),
        huber_k_m=1.0,
        trim_m=1.5,
        min_kept_pairs=3,
        max_shift_m=5.0,
        max_robust_rms_m=0.8,
        veto_rms_band_min_m=0.15,
        veto_rms_band_max_m=0.35,
        veto_min_kept_pairs=4,
    )

    np.testing.assert_allclose(anchor, np.ones(3))
    assert anchor_stats is stats
