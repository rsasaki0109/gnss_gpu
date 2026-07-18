from types import SimpleNamespace

import numpy as np

from gnss_gpu.dd_carrier import DDCarrierComputer
from gnss_gpu.dd_pseudorange import DDPseudorangeComputer
from gnss_gpu.io.rinex_cache import RinexObservationCache


def test_dd_computers_share_parsed_base_and_rover_observations(tmp_path, monkeypatch):
    base = tmp_path / "base.obs"
    rover = tmp_path / "rover.obs"
    base.touch()
    rover.touch()
    calls = []

    def fake_read(path):
        calls.append(path)
        return SimpleNamespace(
            header=SimpleNamespace(approx_position=np.array([4.0e6, 3.0e6, 3.5e6])),
            epochs=[],
        )

    monkeypatch.setattr("gnss_gpu.io.rinex_cache.read_rinex_obs", fake_read)
    cache = RinexObservationCache()
    kwargs = dict(
        rover_obs_path=rover,
        base_position=np.array([4.0e6, 3.0e6, 3.5e6]),
        observation_cache=cache,
    )

    DDCarrierComputer(base, **kwargs)
    DDPseudorangeComputer(base, **kwargs)

    assert len(calls) == 2
    assert len(cache) == 2
