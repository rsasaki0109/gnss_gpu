import numpy as np
import pytest

from gnss_gpu.validation.calibration import (
    coordinate_descent,
    evaluate,
    grid_search,
    score,
)


def make_gen(n=4000):
    def gen(params):
        rng = np.random.default_rng(12345)
        mu, sigma = params["mu_nlos"], params["sigma_los"]
        frac = 0.3
        is_nlos = rng.random(n) < frac
        x = rng.normal(0.0, sigma, n)
        x[is_nlos] = np.abs(rng.normal(mu, 25.0, int(is_nlos.sum())))
        return x

    return gen


def test_score_identity_shift_and_empty():
    x = np.array([0.0, 1.0, 2.0, 3.0])

    assert score(x, x) == 0.0
    assert score(x + 10.0, x) > score(x, x)
    assert np.isinf(score([], x))
    assert np.isinf(score(x, []))


def test_evaluate_keys_and_score_consistency():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = evaluate(x, y, ks_weight=2.5)

    assert set(metrics) == {"wasserstein", "ks", "score"}
    assert metrics["score"] == pytest.approx(
        metrics["wasserstein"] + 2.5 * metrics["ks"]
    )


def test_grid_search_recovers_true_params():
    gen = make_gen()
    true = {"mu_nlos": 20.0, "sigma_los": 4.0}
    target = gen(true)

    result = grid_search(
        gen,
        target,
        {
            "mu_nlos": [10.0, 20.0, 30.0],
            "sigma_los": [2.0, 4.0, 6.0],
        },
    )

    assert result["best_params"] == true
    assert result["best_score"] == pytest.approx(0.0)
    assert len(result["results"]) == 9


def test_grid_search_empty_grid_raises():
    gen = make_gen()
    target = gen({"mu_nlos": 20.0, "sigma_los": 4.0})

    with pytest.raises(ValueError):
        grid_search(gen, target, {})

    with pytest.raises(ValueError):
        grid_search(
            gen,
            target,
            {
                "mu_nlos": [10.0, 20.0],
                "sigma_los": [],
            },
        )


def test_coordinate_descent_improves_and_moves_toward_true():
    gen = make_gen()
    true = {"mu_nlos": 20.0, "sigma_los": 4.0}
    init = {"mu_nlos": 28.0, "sigma_los": 8.0}
    target = gen(true)

    init_score = score(gen(init), target)

    result = coordinate_descent(
        gen,
        target,
        init,
        bounds={
            "mu_nlos": (0.0, 40.0),
            "sigma_los": (1.0, 12.0),
        },
        step={
            "mu_nlos": 4.0,
            "sigma_los": 2.0,
        },
        n_iter=16,
        shrink=0.5,
    )

    assert result["best_score"] < init_score * 0.5
    assert abs(result["best_params"]["mu_nlos"] - true["mu_nlos"]) <= 4.0
    assert abs(result["best_params"]["sigma_los"] - true["sigma_los"]) <= 2.0
    assert len(result["history"]) == 17
    assert result["history"][0] == pytest.approx(init_score)
    assert result["history"][-1] == pytest.approx(result["best_score"])
