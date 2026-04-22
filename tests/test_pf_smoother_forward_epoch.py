from types import SimpleNamespace

import numpy as np

import gnss_gpu.pf_smoother_forward_epoch as epoch_mod
from gnss_gpu.epoch_gate_state import EpochGateState
from gnss_gpu.epoch_observation_inputs import EpochObservationInputs
from gnss_gpu.pf_smoother_epoch_measurements import ForwardEpochMeasurementInputs
from gnss_gpu.pf_smoother_epoch_predict import ForwardEpochPredictResult
from gnss_gpu.pf_smoother_epoch_updates import ForwardEpochUpdatesResult
from gnss_gpu.pf_smoother_forward_epoch import process_pf_smoother_forward_epoch
from pf_smoother_forward_helpers import make_forward_context


def test_process_pf_smoother_forward_epoch_wires_update_sequence(monkeypatch):
    calls = []
    context = make_forward_context(calls)
    epoch_state = SimpleNamespace(
        imu_stop_detected=False,
        used_tdcp=False,
        tdcp_rms=None,
        velocity=np.array([0.1, 0.2, 0.3], dtype=np.float64),
        tdcp_motion_velocity=None,
    )

    def fake_predict(context_arg, sol_epoch_arg, measurements_arg):
        calls.append(("predict", sol_epoch_arg.time.tow, len(measurements_arg)))
        context_arg.pf.predict(
            velocity=epoch_state.velocity,
            dt=0.1,
            sigma_pos=0.25,
            rbpf_velocity_kf=False,
            velocity_process_noise=1.0,
        )
        return ForwardEpochPredictResult(
            epoch_state=epoch_state,
            tow=sol_epoch_arg.time.tow,
            tow_key=round(sol_epoch_arg.time.tow, 1),
            dt=0.1,
            sigma_predict=0.25,
        )

    monkeypatch.setattr(epoch_mod, "apply_forward_epoch_prediction", fake_predict)

    def fake_measurements(context_arg, epoch_state_arg, sol_epoch_arg, measurements_arg):
        epoch_state_arg.carrier_anchor_rows = ["anchor"]
        return ForwardEpochMeasurementInputs(
            observation_inputs=EpochObservationInputs(
                sat_ecef=np.zeros((4, 3), dtype=np.float64),
                pseudoranges=np.ones(4, dtype=np.float64),
                weights=np.ones(4, dtype=np.float64),
                carrier_anchor_rows=["anchor"],
            ),
            gate_state=EpochGateState(
                pf_estimate=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                ess_ratio=0.9,
                spread_m=1.5,
                dd_pr_gate_scale=1.0,
                dd_cp_gate_scale=1.0,
            ),
            sat_ecef=np.zeros((4, 3), dtype=np.float64),
            pseudoranges=np.ones(4, dtype=np.float64),
            weights=np.ones(4, dtype=np.float64),
            gate_pf_estimate=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            gate_ess_ratio=0.9,
            gate_spread_m=1.5,
        )

    monkeypatch.setattr(epoch_mod, "prepare_forward_epoch_measurements", fake_measurements)

    def fake_updates(
        context_arg,
        epoch_state_arg,
        measurement_inputs_arg,
        sol_epoch_arg,
        measurements_arg,
        *,
        tow,
        dt,
    ):
        calls.append(("updates", tow, dt))
        return ForwardEpochUpdatesResult(
            spp_position_ecef=np.array(sol_epoch_arg.position_ecef_m[:3], dtype=np.float64)
        )

    monkeypatch.setattr(epoch_mod, "apply_forward_epoch_updates", fake_updates)

    def fake_record(
        context_arg,
        epoch_state_arg,
        measurement_inputs_arg,
        updates_result_arg,
        *,
        measurements,
        tow,
        dt,
    ):
        calls.append(("record", tow, dt))

    monkeypatch.setattr(epoch_mod, "record_forward_epoch", fake_record)

    sol_epoch = SimpleNamespace(
        time=SimpleNamespace(tow=123.4),
        position_ecef_m=np.array([10.0, 20.0, 30.0], dtype=np.float64),
    )

    process_pf_smoother_forward_epoch(
        context,
        sol_epoch,
        [object(), object(), object(), object()],
    )

    assert [call[0] for call in calls] == [
        "predict",
        "pf_predict",
        "updates",
        "record",
    ]
    assert calls[1][1]["sigma_pos"] == 0.25
    assert calls[2] == ("updates", 123.4, 0.1)
    assert calls[-1] == ("record", 123.4, 0.1)
    assert epoch_state.carrier_anchor_rows == ["anchor"]
