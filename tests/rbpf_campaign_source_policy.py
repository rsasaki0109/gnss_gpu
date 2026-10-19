"""Route frozen campaign oracles without replacing the live estimator."""

import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "benchmarks/probes/rbpf_scalar_20261019"
SCALAR = SNAPSHOT / "experiments/rbpf_fgo/rbpf_fgo.py"
JOINT = SNAPSHOT / "experiments/rbpf_fgo/gtsam_rtk_standalone.py"
PINS = {
    SCALAR: "c977508d6a20413f7669efb39dfb1d87d74b254ce89cf1f57ae8681d7d6b8bb6",
    JOINT: "1cdf4a7931bce541a3ad0c41a7d7e7e5aa404ca11e47bacdbbd832b17e5f8f38",
}


def pytest_configure(config):
    for path, digest in PINS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    path = ROOT / "benchmarks/rbpf_likelihood_cases.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "05f91d2540a047347fa6f30b0380cb146daff816ec3ea10d6217b29d9ebceade"
    module = sys.modules.get("rbpf_likelihood_cases")
    if module is None:
        spec = importlib.util.spec_from_file_location("rbpf_likelihood_cases", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    assert module.SCALAR_SHA256 == PINS[SCALAR]
    module.SCALAR_SOURCE = SCALAR


def pytest_collection_modifyitems(items):
    module = sys.modules.get("rbpf_joint_covariance")
    if module is not None:
        assert Path(module.__file__).resolve() == ROOT / "benchmarks/rbpf_joint_covariance.py"
        assert module.SOURCE_SHA256 == PINS[JOINT]
        module.SOURCE = JOINT
