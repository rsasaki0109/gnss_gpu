"""Paper-locked FFRT threshold evaluation for WP174 shadow analysis."""

from __future__ import annotations

import math

# CoefficientMu.csv, Pf_tol=0.001, supplementary material for:
# Hou, Verhagen, Wu, Sensors 2016, 16, 945.
# Index zero corresponds to one ambiguity.
COEFFICIENTS_PF_TOL_001 = (
    (0.0549, -0.4626, -0.1968),
    (0.0507, -0.4739, -0.1450),
    (0.0838, -0.3960, -0.1556),
    (0.1343, -0.3225, -0.1755),
    (0.1946, -0.2672, -0.1980),
    (0.1876, -0.2651, -0.1429),
    (0.1645, -0.2750, -0.0755),
    (0.1751, -0.2605, -0.0404),
    (0.1229, -0.3011, 0.0634),
    (0.1133, -0.3065, 0.1151),
    (0.0938, -0.3238, 0.1795),
    (0.0636, -0.3737, 0.2505),
    (0.0630, -0.3670, 0.2833),
    (0.0522, -0.3879, 0.3263),
    (0.0512, -0.3843, 0.3543),
    (0.0498, -0.3824, 0.3789),
    (0.0483, -0.3801, 0.4054),
    (0.0489, -0.3726, 0.4257),
    (0.0492, -0.3659, 0.4450),
    (0.0454, -0.3699, 0.4690),
    (0.0443, -0.3689, 0.4880),
    (0.0419, -0.3721, 0.5072),
    (0.0347, -0.3933, 0.5322),
    (0.0321, -0.3999, 0.5500),
    (0.0318, -0.3958, 0.5613),
    (0.0273, -0.4144, 0.5805),
    (0.0261, -0.4147, 0.5928),
    (0.0242, -0.4219, 0.6072),
    (0.0226, -0.4288, 0.6193),
    (0.0208, -0.4348, 0.6309),
    (0.0172, -0.4602, 0.6431),
    (0.0189, -0.4421, 0.6524),
    (0.0212, -0.4206, 0.6574),
    (0.0197, -0.4278, 0.6673),
    (0.0206, -0.4178, 0.6716),
    (0.0174, -0.4399, 0.6852),
    (0.0182, -0.4294, 0.6901),
    (0.0161, -0.4431, 0.7004),
    (0.0132, -0.4681, 0.7071),
    (0.0137, -0.4613, 0.7155),
    (0.0117, -0.4808, 0.7232),
    (0.0118, -0.4736, 0.7286),
    (0.0103, -0.4912, 0.7351),
    (0.0111, -0.4773, 0.7402),
    (0.0095, -0.4982, 0.7474),
    (0.0095, -0.4969, 0.7525),
    (0.0085, -0.5058, 0.7578),
    (0.0098, -0.4837, 0.7602),
    (0.0105, -0.4706, 0.7633),
    (0.0108, -0.4651, 0.7673),
    (0.0072, -0.5210, 0.7757),
    (0.0079, -0.5051, 0.7767),
    (0.0082, -0.4956, 0.7819),
    (0.0094, -0.4744, 0.7840),
    (0.0077, -0.5017, 0.7885),
    (0.0056, -0.5433, 0.7956),
    (0.0057, -0.5400, 0.7998),
    (0.0086, -0.4742, 0.7975),
    (0.0070, -0.4977, 0.7998),
    (0.0085, -0.4741, 0.8039),
    (0.0107, -0.4327, 0.8016),
    (0.0058, -0.5173, 0.8121),
    (0.0050, -0.5369, 0.8181),
    (0.0081, -0.4521, 0.8137),
    (0.0015, -0.7293, 0.8205),
    (0.0016, -0.7571, 0.8317),
)


def minimum_second_to_best_ratio(
    ambiguity_count: int,
    bootstrapped_success_rate: float,
    *,
    tolerable_failure_rate: float = 0.001,
) -> float | None:
    """Convert the paper's best/second mu to gnssplusplus second/best."""
    if (
        ambiguity_count < 1
        or ambiguity_count > len(COEFFICIENTS_PF_TOL_001)
        or not math.isfinite(bootstrapped_success_rate)
        or not 0.0 <= bootstrapped_success_rate <= 1.0
        or not math.isclose(
            tolerable_failure_rate, 0.001, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        return None
    failure_rate_proxy = 1.0 - bootstrapped_success_rate
    if failure_rate_proxy >= 0.2 - 1e-12:
        return math.inf
    if failure_rate_proxy <= tolerable_failure_rate:
        return 1.0
    p1, p2, p3 = COEFFICIENTS_PF_TOL_001[ambiguity_count - 1]
    mu = min(1.0, max(0.0, p1 * failure_rate_proxy**p2 + p3))
    return 1.0 / mu if mu > 0.0 and math.isfinite(mu) else math.inf


def passes_ffrt(
    ambiguity_count: int,
    bootstrapped_success_rate: float,
    second_to_best_ratio: float,
) -> bool:
    threshold = minimum_second_to_best_ratio(
        ambiguity_count, bootstrapped_success_rate
    )
    return (
        threshold is not None
        and math.isfinite(second_to_best_ratio)
        and second_to_best_ratio >= threshold
    )
