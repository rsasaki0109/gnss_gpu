from experiments.run_cpu_batch_receiver import run_experiment, scene


def test_contiguous_batch_and_individual_receiver_pipeline_match():
    result = run_experiment(blocks=24, repeats=1, calibration_trials=99)
    summary = result["summary"]
    assert summary["iq_bitwise_equal"] and summary["receiver_results_equal"]
    assert summary["input_blocks"] == 24
    assert summary["signal_present_blocks"]+summary["signal_absent_blocks"] == 24
    assert sum(summary["states"].values()) == 24
    assert summary["lost_observed_during_outage"]
    assert summary["initial_acquisition_decision_ms"] == 2
    assert summary["reacquisition_after_outage_ms"] == 2
    assert {r["mode"] for r in result["timings"]} == {"individual", "batch"}
    assert all(r["total_s"] >= r["generation_s"] > 0 for r in result["timings"])


def test_scene_offsets_continue_code_carrier_and_navigation_time():
    rows, starts, seeds, outage = scene(128)
    assert starts == [2046*i for i in range(128)]
    assert len(set(seeds)) == len(seeds)
    # These parameters stay referenced to absolute sample zero, including the
    # navigation sequence; block-specific power changes represent the outage.
    for key in ("code_phase", "carrier_phase", "doppler_hz", "nav_bits"):
        assert rows[0][0][key] == rows[-1][0][key]
    assert outage == (48, 64)
    assert all(ch["amplitude"] == 0 for row in rows[48:64] for ch in row)
    assert rows[64][0]["amplitude"] == 1.
