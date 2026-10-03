"""DD updates refuse libgnsspp rows that cannot be paired per satellite."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gnss_gpu.pf_smoother_runtime import (
    ObservationComputers,
    require_measurement_satellite_ids,
)

WITH_DD = ObservationComputers(base_obs_path=None, dd_computer=object())
WITHOUT_DD = ObservationComputers(base_obs_path=None)


def _epochs(*rows):
    return [(None, []), (None, list(rows))]


def test_rows_without_prn_fail_when_dd_is_enabled():
    with pytest.raises(RuntimeError, match="CorrectedMeasurement.prn"):
        require_measurement_satellite_ids(
            _epochs(SimpleNamespace(system_id=0), SimpleNamespace(system_id=0)),
            WITH_DD,
        )


def test_rows_with_prn_pass():
    require_measurement_satellite_ids(
        _epochs(SimpleNamespace(system_id=0, prn=5), SimpleNamespace(system_id=0, prn=7)),
        WITH_DD,
    )


def test_rows_without_prn_are_fine_without_dd():
    require_measurement_satellite_ids(_epochs(SimpleNamespace(system_id=0)), WITHOUT_DD)
