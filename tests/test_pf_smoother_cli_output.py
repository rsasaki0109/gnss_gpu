from pathlib import Path

from gnss_gpu.pf_smoother_cli_output import (
    build_pf_smoother_variant_metric_lines,
    pf_smoother_variant_metrics,
    print_pf_smoother_run_header,
    print_pf_smoother_variant_start,
    select_pf_smoother_variants,
    write_pf_smoother_result_csv,
)


def test_select_pf_smoother_variants_matches_cli_modes():
    assert select_pf_smoother_variants(compare_both=True, use_smoother=False) == [
        ("forward_only", False),
        ("with_smoother", True),
    ]
    assert select_pf_smoother_variants(compare_both=False, use_smoother=False) == [
        ("forward_only", False)
    ]
    assert select_pf_smoother_variants(compare_both=False, use_smoother=True) == [
        ("with_smoother", True)
    ]


def test_print_pf_smoother_run_header_and_variant_start_use_injected_printer():
    calls = []

    print_pf_smoother_run_header("Odaiba", print_func=lambda *args, **kwargs: calls.append((args, kwargs)))
    print_pf_smoother_variant_start(
        label="forward_only",
        predict_guide="imu",
        position_update_sigma=1.9,
        sigma_pos_tdcp=None,
        use_smoother=False,
        print_func=lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    assert calls[0] == (("\n============================================================\n  Odaiba\n============================================================",), {})
    assert calls[1] == (
        ("  [forward_only] guide=imu PU=1.9 sp_tdcp=None smooth=False...",),
        {"end": " ", "flush": True},
    )


def test_variant_metrics_extracts_metrics_and_formats_status_lines():
    out = {
        "elapsed_ms": 30.0,
        "forward_metrics": {"n_epochs": 10, "p50": 1.234, "rms_2d": 2.345},
        "smoothed_metrics": {"p50": 0.987, "rms_2d": 1.876},
        "n_tail_guard_applied": 2,
        "n_widelane_forward_guard_applied": 1,
        "n_stop_segment_epochs_applied": 3,
        "stop_segment_info": {"segments_applied": 1},
        "fgo_local_applied": True,
        "fgo_local_info": {
            "window": (1, 9),
            "solve_window": (0, 10),
            "lambda": {
                "n_fixed": 4,
                "n_fixed_observations": 6,
                "fixed_by_system": {"G": 4},
            },
        },
    }

    forward, smoothed, n_epochs, ms_per_epoch = pf_smoother_variant_metrics(out)
    lines = build_pf_smoother_variant_metric_lines(
        out,
        forward,
        smoothed,
        n_epochs=n_epochs,
        ms_per_epoch=ms_per_epoch,
    )

    assert n_epochs == 10
    assert ms_per_epoch == 3.0
    assert lines == [
        "FWD P50=1.23m RMS=2.35m (10 ep, 3.00ms/ep)",
        "       SMTH P50=0.99m RMS=1.88m",
        "       tail guard applied: 2 epochs",
        "       wide-lane forward guard applied: 1 epochs",
        "       stop segment constant applied: 3 epochs segments=1",
        "       local FGO window: (1, 9) solve=(0, 10)",
        "       local FGO lambda: fixed=4 obs=6 by_system={'G': 4}",
    ]


def test_write_pf_smoother_result_csv_writes_header_rows_and_saved_message(tmp_path):
    out_csv = tmp_path / "results.csv"
    printed = []

    write_pf_smoother_result_csv(
        [{"run": "Odaiba", "forward_p50": 1.2}],
        out_csv,
        print_func=printed.append,
    )

    assert out_csv.read_text(encoding="utf-8").splitlines() == [
        "run,forward_p50",
        "Odaiba,1.2",
    ]
    assert printed == [f"\nSaved: {Path(out_csv)}"]


def test_write_pf_smoother_result_csv_ignores_empty_rows(tmp_path):
    out_csv = tmp_path / "results.csv"
    printed = []

    write_pf_smoother_result_csv([], out_csv, print_func=printed.append)

    assert not out_csv.exists()
    assert printed == []
