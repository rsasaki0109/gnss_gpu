"""Historical oracle hashes stay distinct from the live estimator."""

import hashlib
from pathlib import Path

import rbpf_campaign_source_policy as policy
import rbpf_likelihood_cases as cases


def test_historical_sources_and_routed_fixture():
    for path, digest in policy.PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert cases.SCALAR_SOURCE == policy.SCALAR
    assert Path(cases.__file__).resolve() == policy.ROOT / "benchmarks/rbpf_likelihood_cases.py"
    assert policy.SCALAR != policy.ROOT / "experiments/rbpf_fgo/rbpf_fgo.py"
    assert policy.JOINT != policy.ROOT / "experiments/rbpf_fgo/gtsam_rtk_standalone.py"
