import argparse
import importlib.util
import sys
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "exp_gpu_scenario_sweeper.py"
_SPEC = importlib.util.spec_from_file_location("exp_gpu_scenario_sweeper", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SWEEP = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _SWEEP
_SPEC.loader.exec_module(_SWEEP)


def test_parse_sweep_lists():
    assert _SWEEP._parse_float_list("0, 1.5,2") == (0.0, 1.5, 2.0)
    assert _SWEEP._parse_int_list("0, 48,112") == (0, 48, 112)


def test_cpu_only_sweeper_writes_outputs(tmp_path):
    args = argparse.Namespace(
        out_dir=tmp_path,
        cpu_only=True,
        seed=123,
        jammer_jnr_db=(0.0, 16.0),
        spoof_delay_samples=(0, 64),
        building_height_scale=(0.8, 1.2),
        particles_per_epoch=(3,),
        prn=7,
        prn_search=(7, 8),
        sampling_freq=1.023e6,
        duration_ms=1.0,
        code_phase_samples=37,
        doppler_hz=1000.0,
        snr_db=22.0,
        spoof_doppler_offset_hz=0.0,
        spoof_jsr_db=6.0,
        doppler_range_hz=2000.0,
        doppler_step_hz=500.0,
        acquisition_threshold=2.0,
        fft_size=256,
        hop_size=64,
        interference_threshold_db=12.0,
        false_lock_code_error_samples=16.0,
        false_lock_doppler_error_hz=750.0,
        n_epochs=6,
        length_m=120.0,
        block_depth_m=32.0,
        road_half_width_m=10.0,
        building_width_m=22.0,
        base_height_m=26.0,
        height_wave_m=16.0,
        n_blocks_per_side=3,
        rx_height_m=1.6,
        sat_range_m=10000.0,
    )

    rows = _SWEEP.run(args)
    summary = _SWEEP._summarize(rows, elapsed_s=0.0)
    _SWEEP._write_csv(tmp_path / "gpu_scenario_sweep_summary.csv", rows)
    _SWEEP._write_json(tmp_path / "gpu_scenario_sweep_summary.json", rows, summary)
    _SWEEP._write_html(tmp_path / "gpu_scenario_sweep_report.html", rows, summary)

    assert len(rows) == 8
    assert summary["n_rows"] == 8
    assert 0.0 <= summary["mean_risk"] <= 1.0
    assert 0.0 <= summary["max_risk"] <= 1.0
    assert any(row.false_lock for row in rows)
    assert (tmp_path / "gpu_scenario_sweep_summary.csv").exists()
    assert (tmp_path / "gpu_scenario_sweep_summary.json").exists()
    assert (tmp_path / "gpu_scenario_sweep_report.html").exists()
