"""CPU-side validation tests for SVGD wrapper."""

import numpy as np
import pytest

from gnss_fixtures import generate_satellites
from gnss_gpu.svgd import SVGDParticleFilter


def test_svgd_init_rejects_invalid_config():
    with pytest.raises(ValueError, match="n_particles must be positive"):
        SVGDParticleFilter(n_particles=0)
    with pytest.raises(ValueError, match="sigma_pr must be positive"):
        SVGDParticleFilter(sigma_pr=0.0)
    with pytest.raises(ValueError, match="step_size must be positive"):
        SVGDParticleFilter(step_size=-1.0)
    with pytest.raises(ValueError, match="n_neighbors must be positive"):
        SVGDParticleFilter(n_neighbors=0)
    with pytest.raises(ValueError, match="n_bandwidth_subsample must be positive"):
        SVGDParticleFilter(n_bandwidth_subsample=0)
    with pytest.raises(ValueError, match="sigma_pos must be finite"):
        SVGDParticleFilter(sigma_pos=np.inf)


def test_svgd_initialize_rejects_invalid_position_before_native_call():
    pf = SVGDParticleFilter(n_particles=100)

    with pytest.raises(ValueError, match="position_ecef must have shape"):
        pf.initialize([0.0, 0.0])
    with pytest.raises(ValueError, match="position_ecef must be finite"):
        pf.initialize([0.0, np.nan, 0.0])
    with pytest.raises(ValueError, match="spread_pos must be positive"):
        pf.initialize([0.0, 0.0, 0.0], spread_pos=0.0)
    with pytest.raises(ValueError, match="spread_cb must be finite"):
        pf.initialize([0.0, 0.0, 0.0], spread_cb=np.inf)


def test_svgd_predict_rejects_invalid_inputs_before_native_call():
    pf = SVGDParticleFilter(n_particles=100)
    pf.initialize(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="dt must be non-negative"):
        pf.predict(dt=-1.0)
    with pytest.raises(ValueError, match="dt must be finite"):
        pf.predict(dt=np.nan)
    with pytest.raises(ValueError, match="velocity must have shape"):
        pf.predict(velocity=[1.0, 2.0])
    with pytest.raises(ValueError, match="velocity must be finite"):
        pf.predict(velocity=[1.0, np.inf, 3.0])


def test_svgd_update_rejects_invalid_inputs_before_native_call():
    pf = SVGDParticleFilter(n_particles=100)
    pf.initialize(np.array([1.0, 2.0, 3.0]))
    sat = generate_satellites(4)
    pr = np.ones(4)

    with pytest.raises(ValueError, match="pseudoranges must contain at least"):
        pf.update(sat, [])
    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        pf.update(sat[:2], pr)
    with pytest.raises(ValueError, match="sat_ecef must be finite"):
        bad_sat = sat.copy()
        bad_sat[0, 0] = np.inf
        pf.update(bad_sat, pr)
    with pytest.raises(ValueError, match="pseudoranges must be finite"):
        pf.update(sat, [1.0, np.nan, 1.0, 1.0])
    with pytest.raises(ValueError, match="weights length must match"):
        pf.update(sat, pr, weights=np.ones(3))
    with pytest.raises(ValueError, match="weights must be non-negative"):
        pf.update(sat, pr, weights=[1.0, -1.0, 1.0, 1.0])
