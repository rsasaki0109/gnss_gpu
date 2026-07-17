import numpy as np
import pytest

from gnss_gpu.annealed_smc import annealed_smc_update, ess_ratio_from_log_weights


class _FakePF:
    """Small deterministic PF implementing the annealing protocol."""

    def __init__(self, log_weights, labels=None, ess_threshold=0.5):
        self.log_weights = np.asarray(log_weights, dtype=np.float64).copy()
        self.labels = (
            np.arange(self.log_weights.size, dtype=np.int64)
            if labels is None
            else np.asarray(labels, dtype=np.int64).copy()
        )
        self.ess_threshold = float(ess_threshold)
        self.resamples = 0

    def get_log_weights(self):
        return self.log_weights.copy()

    def set_log_weights(self, values):
        self.log_weights = np.asarray(values, dtype=np.float64).copy()

    def add_likelihood(self, values_by_label):
        values = np.asarray(values_by_label, dtype=np.float64)
        self.log_weights += values[self.labels]

    def resample(self):
        shifted = self.log_weights - np.max(self.log_weights)
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        cdf = np.cumsum(weights)
        points = (np.arange(weights.size, dtype=np.float64) + 0.5) / weights.size
        ancestors = np.searchsorted(cdf, points, side="left")
        self.labels = self.labels[ancestors]
        self.log_weights = np.zeros_like(self.log_weights)
        self.resamples += 1

    def resample_if_needed(self):
        if ess_ratio_from_log_weights(self.log_weights) < self.ess_threshold:
            self.resample()
            return True
        return False


def _logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    vmax = np.max(values)
    return float(vmax + np.log(np.sum(np.exp(values - vmax))))


def test_unsharp_likelihood_is_consumed_exactly_without_resampling():
    pre = np.array([0.0, -0.1, -0.2, -0.3])
    log_likelihood = np.array([-0.2, 0.1, -0.1, 0.0])
    pf = _FakePF(pre, ess_threshold=0.1)

    result = annealed_smc_update(
        pf,
        lambda: pf.add_likelihood(log_likelihood),
        target_ess_ratio=0.5,
        resample_at_end=False,
    )

    assert result.beta_increments == (1.0,)
    assert result.beta_consumed == pytest.approx(1.0, abs=1e-12)
    assert result.likelihood_evaluations == 1
    assert result.resample_count == 0
    np.testing.assert_allclose(pf.get_log_weights(), pre + log_likelihood)
    expected_log_z = _logsumexp(pre + log_likelihood) - _logsumexp(pre)
    assert result.log_evidence == pytest.approx(expected_log_z, abs=1e-12)


def test_sharp_likelihood_uses_multiple_increments_and_full_beta():
    n = 256
    log_likelihood = np.linspace(-40.0, 0.0, n)
    pf = _FakePF(np.zeros(n), ess_threshold=0.5)

    result = annealed_smc_update(
        pf,
        lambda: pf.add_likelihood(log_likelihood),
        target_ess_ratio=0.8,
        max_bisection_iters=24,
        max_tempering_steps=64,
        resample_at_end=False,
    )

    assert len(result.beta_increments) > 1
    assert result.likelihood_evaluations == len(result.beta_increments)
    assert result.resample_count == len(result.beta_increments) - 1
    assert result.beta_consumed == pytest.approx(1.0, abs=1e-10)
    assert sum(result.beta_increments) == pytest.approx(1.0, abs=1e-10)
    assert np.isfinite(result.log_evidence)
    assert all(delta > 0.0 for delta in result.beta_increments)


def test_low_entering_ess_is_resampled_instead_of_reverting_stage():
    n = 64
    entering = np.full(n, -30.0)
    entering[0] = 0.0
    log_likelihood = np.linspace(-1.0, 0.0, n)
    pf = _FakePF(entering, ess_threshold=0.5)

    result = annealed_smc_update(
        pf,
        lambda: pf.add_likelihood(log_likelihood),
        target_ess_ratio=0.8,
        resample_at_end=False,
    )

    assert result.initial_ess_ratio < 0.8
    assert result.resample_count >= 1
    assert result.beta_consumed == pytest.approx(1.0, abs=1e-12)


def test_step_limit_fails_loudly_instead_of_discarding_remainder():
    n = 128
    log_likelihood = np.linspace(-50.0, 0.0, n)
    pf = _FakePF(np.zeros(n), ess_threshold=0.5)

    with pytest.raises(RuntimeError, match="exhausted"):
        annealed_smc_update(
            pf,
            lambda: pf.add_likelihood(log_likelihood),
            target_ess_ratio=0.9,
            max_tempering_steps=1,
            resample_at_end=False,
        )
