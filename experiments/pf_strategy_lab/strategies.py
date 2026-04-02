from __future__ import annotations

from dataclasses import dataclass, field

from .interfaces import StrategyContext, StrategyDecision


def _feature(context: StrategyContext, key: str) -> float:
    return float(context.features[key])


def _quality_veto_components(
    *,
    blocked: float,
    positive: float,
    disagreement: float,
    cb_disagreement: float,
    residual: float,
    p95_abs_residual: float,
    satellites: float,
    close_blocked_low: float,
    close_blocked_high: float,
    close_disagreement_max_m: float,
    close_cb_max_m: float,
    close_residual_max_m: float,
    close_satellite_max: float,
    close_p95_abs_residual_max_m: float,
    far_blocked_max: float,
    far_positive_min: float,
    far_disagreement_min_m: float,
    far_cb_min_m: float,
) -> tuple[bool, bool, float]:
    close_mode = (
        close_blocked_low <= blocked <= close_blocked_high
        and disagreement <= close_disagreement_max_m
        and cb_disagreement <= close_cb_max_m
        and residual <= close_residual_max_m
        and satellites <= close_satellite_max
        and p95_abs_residual <= close_p95_abs_residual_max_m
    )
    far_mode = (
        blocked <= far_blocked_max
        and positive >= far_positive_min
        and disagreement >= far_disagreement_min_m
        and cb_disagreement >= far_cb_min_m
    )
    score = max(
        min(
            blocked / max(close_blocked_low, 1e-9),
            close_blocked_high / max(blocked, 1e-9),
            close_disagreement_max_m / max(disagreement, 1e-9),
            close_satellite_max / max(satellites, 1e-9),
            close_p95_abs_residual_max_m / max(p95_abs_residual, 1e-9),
        ) if close_mode else 0.0,
        min(
            far_blocked_max / max(blocked, 1e-9),
            positive / max(far_positive_min, 1e-9),
            disagreement / max(far_disagreement_min_m, 1e-9),
        ) if far_mode else 0.0,
    )
    return close_mode, far_mode, score


@dataclass(frozen=True)
class AlwaysRobustStrategy:
    name: str = "always_robust"
    style: str = "constant"

    def required_features(self) -> tuple[str, ...]:
        return ()

    def parameters(self) -> dict[str, float]:
        return {}

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(use_blocked=False, score=0.0, rationale="always robust")


@dataclass(frozen=True)
class AlwaysBlockedStrategy:
    name: str = "always_blocked"
    style: str = "constant"

    def required_features(self) -> tuple[str, ...]:
        return ()

    def parameters(self) -> dict[str, float]:
        return {}

    def decide(self, context: StrategyContext) -> StrategyDecision:
        return StrategyDecision(use_blocked=True, score=1.0, rationale="always blocked")


@dataclass(frozen=True)
class DisagreementGateStrategy:
    disagreement_threshold_m: float = 80.0
    name: str = "disagreement_gate"
    style: str = "oop-threshold"

    def required_features(self) -> tuple[str, ...]:
        return ("disagreement_m",)

    def parameters(self) -> dict[str, float]:
        return {"disagreement_threshold_m": float(self.disagreement_threshold_m)}

    def decide(self, context: StrategyContext) -> StrategyDecision:
        disagreement = _feature(context, "disagreement_m")
        use_blocked = disagreement >= self.disagreement_threshold_m
        return StrategyDecision(
            use_blocked=use_blocked,
            score=disagreement,
            rationale=f"disagreement {disagreement:.2f} >= {self.disagreement_threshold_m:.2f}",
        )


@dataclass(frozen=True)
class RuleChainGateStrategy:
    blocked_threshold: float = 0.001
    positive_threshold: float = 0.25
    disagreement_threshold_m: float = 80.0
    name: str = "rule_chain_gate"
    style: str = "pipeline-rule-chain"

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "blocked_threshold": float(self.blocked_threshold),
            "positive_threshold": float(self.positive_threshold),
            "disagreement_threshold_m": float(self.disagreement_threshold_m),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        use_blocked = (
            blocked >= self.blocked_threshold
            and positive >= self.positive_threshold
            and disagreement >= self.disagreement_threshold_m
        )
        return StrategyDecision(
            use_blocked=use_blocked,
            score=min(
                blocked / max(self.blocked_threshold, 1e-9),
                positive / max(self.positive_threshold, 1e-9) if self.positive_threshold > 0 else 1.0,
                disagreement / max(self.disagreement_threshold_m, 1e-9),
            ),
            rationale=(
                f"blocked={blocked:.4f}, positive={positive:.4f}, disagreement={disagreement:.2f}"
            ),
        )


@dataclass(frozen=True)
class WeightedScoreGateStrategy:
    blocked_scale: float = 0.01
    positive_scale: float = 0.50
    disagreement_scale_m: float = 80.0
    threshold: float = 1.60
    name: str = "weighted_score_gate"
    style: str = "functional-score"

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "blocked_scale": float(self.blocked_scale),
            "positive_scale": float(self.positive_scale),
            "disagreement_scale_m": float(self.disagreement_scale_m),
            "threshold": float(self.threshold),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = min(_feature(context, "mean_weighted_blocked_frac") / self.blocked_scale, 1.0)
        positive = min(_feature(context, "blocked_positive_frac_gt5") / self.positive_scale, 1.0)
        disagreement = min(_feature(context, "disagreement_m") / self.disagreement_scale_m, 1.0)
        score = blocked + positive + disagreement
        use_blocked = score >= self.threshold
        return StrategyDecision(
            use_blocked=use_blocked,
            score=score,
            rationale=(
                f"weighted_score={score:.3f} from blocked={blocked:.3f}, "
                f"positive={positive:.3f}, disagreement={disagreement:.3f}"
            ),
        )


@dataclass(frozen=True)
class ClockVetoGateStrategy:
    disagreement_threshold_m: float = 85.0
    cb_disagreement_threshold_m: float = 30.0
    blocked_ceiling: float = 0.03
    name: str = "clock_veto_gate"
    style: str = "pipeline-veto"

    def required_features(self) -> tuple[str, ...]:
        return (
            "disagreement_m",
            "cb_disagreement_m",
            "mean_weighted_blocked_frac",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "disagreement_threshold_m": float(self.disagreement_threshold_m),
            "cb_disagreement_threshold_m": float(self.cb_disagreement_threshold_m),
            "blocked_ceiling": float(self.blocked_ceiling),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        blocked = _feature(context, "mean_weighted_blocked_frac")
        signal = (
            disagreement >= self.disagreement_threshold_m
            and cb_disagreement >= self.cb_disagreement_threshold_m
        )
        vetoed = blocked > self.blocked_ceiling
        use_blocked = signal and not vetoed
        score = min(
            disagreement / max(self.disagreement_threshold_m, 1e-9),
            cb_disagreement / max(self.cb_disagreement_threshold_m, 1e-9),
        )
        return StrategyDecision(
            use_blocked=use_blocked,
            score=score,
            rationale=(
                f"signal(d={disagreement:.2f}, cb={cb_disagreement:.2f}), "
                f"blocked={blocked:.4f}, veto={vetoed}"
            ),
        )


@dataclass(frozen=True)
class DualModeRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    name: str = "dual_mode_regime_gate"
    style: str = "regime-branch"

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")

        close_mode = (
            self.close_blocked_low <= blocked <= self.close_blocked_high
            and disagreement <= self.close_disagreement_max_m
            and cb_disagreement <= self.close_cb_max_m
            and residual <= self.close_residual_max_m
        )
        far_mode = (
            blocked <= self.far_blocked_max
            and positive >= self.far_positive_min
            and disagreement >= self.far_disagreement_min_m
            and cb_disagreement >= self.far_cb_min_m
        )
        use_blocked = close_mode or far_mode
        score = max(
            min(
                blocked / max(self.close_blocked_low, 1e-9),
                self.close_blocked_high / max(blocked, 1e-9),
                self.close_disagreement_max_m / max(disagreement, 1e-9),
            ) if close_mode else 0.0,
            min(
                self.far_blocked_max / max(blocked, 1e-9),
                positive / max(self.far_positive_min, 1e-9),
                disagreement / max(self.far_disagreement_min_m, 1e-9),
            ) if far_mode else 0.0,
        )
        return StrategyDecision(
            use_blocked=use_blocked,
            score=score,
            rationale=(
                f"close={close_mode}, far={far_mode}, blocked={blocked:.4f}, "
                f"positive={positive:.3f}, disagreement={disagreement:.2f}, "
                f"cb={cb_disagreement:.2f}, residual={residual:.2f}"
            ),
        )


@dataclass(frozen=True)
class QualityVetoRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    close_satellite_max: float = 8.0
    close_p95_abs_residual_max_m: float = 50.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    name: str = "quality_veto_regime_gate"
    style: str = "regime-quality-veto"

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
            "robust_p95_abs_residual",
            "satellite_count",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "close_satellite_max": float(self.close_satellite_max),
            "close_p95_abs_residual_max_m": float(self.close_p95_abs_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")
        p95_abs_residual = _feature(context, "robust_p95_abs_residual")
        satellites = _feature(context, "satellite_count")

        close_mode, far_mode, score = _quality_veto_components(
            blocked=blocked,
            positive=positive,
            disagreement=disagreement,
            cb_disagreement=cb_disagreement,
            residual=residual,
            p95_abs_residual=p95_abs_residual,
            satellites=satellites,
            close_blocked_low=self.close_blocked_low,
            close_blocked_high=self.close_blocked_high,
            close_disagreement_max_m=self.close_disagreement_max_m,
            close_cb_max_m=self.close_cb_max_m,
            close_residual_max_m=self.close_residual_max_m,
            close_satellite_max=self.close_satellite_max,
            close_p95_abs_residual_max_m=self.close_p95_abs_residual_max_m,
            far_blocked_max=self.far_blocked_max,
            far_positive_min=self.far_positive_min,
            far_disagreement_min_m=self.far_disagreement_min_m,
            far_cb_min_m=self.far_cb_min_m,
        )
        use_blocked = close_mode or far_mode
        return StrategyDecision(
            use_blocked=use_blocked,
            score=score,
            rationale=(
                f"close={close_mode}, far={far_mode}, blocked={blocked:.4f}, "
                f"positive={positive:.3f}, disagreement={disagreement:.2f}, "
                f"cb={cb_disagreement:.2f}, residual={residual:.2f}, "
                f"p95_abs={p95_abs_residual:.2f}, sats={satellites:.0f}"
            ),
        )


@dataclass
class HysteresisQualityVetoRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    close_satellite_max: float = 8.0
    close_p95_abs_residual_max_m: float = 50.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    enter_confirm_epochs: int = 1
    exit_confirm_epochs: int = 2
    name: str = "hysteresis_quality_veto_regime_gate"
    style: str = "stateful-hysteresis"
    _blocked_active: bool = field(default=False, init=False, repr=False, compare=False)
    _candidate_streak: int = field(default=0, init=False, repr=False, compare=False)
    _clear_streak: int = field(default=0, init=False, repr=False, compare=False)

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
            "robust_p95_abs_residual",
            "satellite_count",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "close_satellite_max": float(self.close_satellite_max),
            "close_p95_abs_residual_max_m": float(self.close_p95_abs_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
            "enter_confirm_epochs": float(self.enter_confirm_epochs),
            "exit_confirm_epochs": float(self.exit_confirm_epochs),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")
        p95_abs_residual = _feature(context, "robust_p95_abs_residual")
        satellites = _feature(context, "satellite_count")

        close_mode, far_mode, score = _quality_veto_components(
            blocked=blocked,
            positive=positive,
            disagreement=disagreement,
            cb_disagreement=cb_disagreement,
            residual=residual,
            p95_abs_residual=p95_abs_residual,
            satellites=satellites,
            close_blocked_low=self.close_blocked_low,
            close_blocked_high=self.close_blocked_high,
            close_disagreement_max_m=self.close_disagreement_max_m,
            close_cb_max_m=self.close_cb_max_m,
            close_residual_max_m=self.close_residual_max_m,
            close_satellite_max=self.close_satellite_max,
            close_p95_abs_residual_max_m=self.close_p95_abs_residual_max_m,
            far_blocked_max=self.far_blocked_max,
            far_positive_min=self.far_positive_min,
            far_disagreement_min_m=self.far_disagreement_min_m,
            far_cb_min_m=self.far_cb_min_m,
        )
        candidate = close_mode or far_mode
        if candidate:
            self._candidate_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._candidate_streak = 0

        if not self._blocked_active and self._candidate_streak >= self.enter_confirm_epochs:
            self._blocked_active = True
        if self._blocked_active and self._clear_streak >= self.exit_confirm_epochs:
            self._blocked_active = False

        return StrategyDecision(
            use_blocked=self._blocked_active,
            score=score,
            rationale=(
                f"candidate={candidate}, active={self._blocked_active}, close={close_mode}, far={far_mode}, "
                f"enter={self._candidate_streak}, clear={self._clear_streak}, "
                f"blocked={blocked:.4f}, p95_abs={p95_abs_residual:.2f}, sats={satellites:.0f}"
            ),
        )


@dataclass
class ModeAwareHysteresisQualityVetoRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    close_satellite_max: float = 9.0
    close_p95_abs_residual_max_m: float = 55.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    enter_confirm_close_epochs: int = 2
    enter_confirm_far_epochs: int = 1
    exit_confirm_epochs: int = 4
    name: str = "mode_aware_hysteresis_quality_veto_regime_gate"
    style: str = "stateful-branch-hysteresis"
    _blocked_active: bool = field(default=False, init=False, repr=False, compare=False)
    _close_streak: int = field(default=0, init=False, repr=False, compare=False)
    _far_streak: int = field(default=0, init=False, repr=False, compare=False)
    _clear_streak: int = field(default=0, init=False, repr=False, compare=False)

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
            "robust_p95_abs_residual",
            "satellite_count",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "close_satellite_max": float(self.close_satellite_max),
            "close_p95_abs_residual_max_m": float(self.close_p95_abs_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
            "enter_confirm_close_epochs": float(self.enter_confirm_close_epochs),
            "enter_confirm_far_epochs": float(self.enter_confirm_far_epochs),
            "exit_confirm_epochs": float(self.exit_confirm_epochs),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")
        p95_abs_residual = _feature(context, "robust_p95_abs_residual")
        satellites = _feature(context, "satellite_count")

        close_mode, far_mode, score = _quality_veto_components(
            blocked=blocked,
            positive=positive,
            disagreement=disagreement,
            cb_disagreement=cb_disagreement,
            residual=residual,
            p95_abs_residual=p95_abs_residual,
            satellites=satellites,
            close_blocked_low=self.close_blocked_low,
            close_blocked_high=self.close_blocked_high,
            close_disagreement_max_m=self.close_disagreement_max_m,
            close_cb_max_m=self.close_cb_max_m,
            close_residual_max_m=self.close_residual_max_m,
            close_satellite_max=self.close_satellite_max,
            close_p95_abs_residual_max_m=self.close_p95_abs_residual_max_m,
            far_blocked_max=self.far_blocked_max,
            far_positive_min=self.far_positive_min,
            far_disagreement_min_m=self.far_disagreement_min_m,
            far_cb_min_m=self.far_cb_min_m,
        )
        candidate = close_mode or far_mode
        self._close_streak = self._close_streak + 1 if close_mode else 0
        self._far_streak = self._far_streak + 1 if far_mode else 0
        self._clear_streak = 0 if candidate else self._clear_streak + 1

        if not self._blocked_active and (
            self._close_streak >= self.enter_confirm_close_epochs
            or self._far_streak >= self.enter_confirm_far_epochs
        ):
            self._blocked_active = True
        if self._blocked_active and self._clear_streak >= self.exit_confirm_epochs:
            self._blocked_active = False

        return StrategyDecision(
            use_blocked=self._blocked_active,
            score=score,
            rationale=(
                f"active={self._blocked_active}, close={close_mode}, far={far_mode}, "
                f"close_streak={self._close_streak}, far_streak={self._far_streak}, clear={self._clear_streak}, "
                f"blocked={blocked:.4f}, p95_abs={p95_abs_residual:.2f}, sats={satellites:.0f}"
            ),
        )


@dataclass
class BranchAwareHysteresisQualityVetoRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    close_satellite_max: float = 9.0
    close_p95_abs_residual_max_m: float = 55.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    enter_confirm_close_epochs: int = 2
    enter_confirm_far_epochs: int = 1
    exit_confirm_close_epochs: int = 3
    exit_confirm_far_epochs: int = 5
    name: str = "branch_aware_hysteresis_quality_veto_regime_gate"
    style: str = "stateful-branch-exit-hysteresis"
    _blocked_active: bool = field(default=False, init=False, repr=False, compare=False)
    _active_mode: str = field(default="", init=False, repr=False, compare=False)
    _close_streak: int = field(default=0, init=False, repr=False, compare=False)
    _far_streak: int = field(default=0, init=False, repr=False, compare=False)
    _clear_streak: int = field(default=0, init=False, repr=False, compare=False)

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
            "robust_p95_abs_residual",
            "satellite_count",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "close_satellite_max": float(self.close_satellite_max),
            "close_p95_abs_residual_max_m": float(self.close_p95_abs_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
            "enter_confirm_close_epochs": float(self.enter_confirm_close_epochs),
            "enter_confirm_far_epochs": float(self.enter_confirm_far_epochs),
            "exit_confirm_close_epochs": float(self.exit_confirm_close_epochs),
            "exit_confirm_far_epochs": float(self.exit_confirm_far_epochs),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")
        p95_abs_residual = _feature(context, "robust_p95_abs_residual")
        satellites = _feature(context, "satellite_count")

        close_mode, far_mode, score = _quality_veto_components(
            blocked=blocked,
            positive=positive,
            disagreement=disagreement,
            cb_disagreement=cb_disagreement,
            residual=residual,
            p95_abs_residual=p95_abs_residual,
            satellites=satellites,
            close_blocked_low=self.close_blocked_low,
            close_blocked_high=self.close_blocked_high,
            close_disagreement_max_m=self.close_disagreement_max_m,
            close_cb_max_m=self.close_cb_max_m,
            close_residual_max_m=self.close_residual_max_m,
            close_satellite_max=self.close_satellite_max,
            close_p95_abs_residual_max_m=self.close_p95_abs_residual_max_m,
            far_blocked_max=self.far_blocked_max,
            far_positive_min=self.far_positive_min,
            far_disagreement_min_m=self.far_disagreement_min_m,
            far_cb_min_m=self.far_cb_min_m,
        )
        candidate = close_mode or far_mode
        self._close_streak = self._close_streak + 1 if close_mode else 0
        self._far_streak = self._far_streak + 1 if far_mode else 0
        self._clear_streak = 0 if candidate else self._clear_streak + 1

        if not self._blocked_active:
            if self._close_streak >= self.enter_confirm_close_epochs:
                self._blocked_active = True
                self._active_mode = "close"
            elif self._far_streak >= self.enter_confirm_far_epochs:
                self._blocked_active = True
                self._active_mode = "far"
        else:
            exit_confirm_epochs = (
                self.exit_confirm_far_epochs
                if self._active_mode == "far"
                else self.exit_confirm_close_epochs
            )
            if self._clear_streak >= exit_confirm_epochs:
                self._blocked_active = False
                self._active_mode = ""

        return StrategyDecision(
            use_blocked=self._blocked_active,
            score=score,
            rationale=(
                f"active={self._blocked_active}, mode={self._active_mode or 'clear'}, "
                f"close={close_mode}, far={far_mode}, close_streak={self._close_streak}, "
                f"far_streak={self._far_streak}, clear={self._clear_streak}, "
                f"blocked={blocked:.4f}, p95_abs={p95_abs_residual:.2f}, sats={satellites:.0f}"
            ),
        )


@dataclass
class RescueBranchAwareHysteresisQualityVetoRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    close_satellite_max: float = 9.0
    close_p95_abs_residual_max_m: float = 55.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    enter_confirm_close_epochs: int = 3
    enter_confirm_far_epochs: int = 1
    exit_confirm_close_epochs: int = 3
    exit_confirm_far_epochs: int = 5
    close_rescue_satellite_max: float = 8.0
    close_rescue_p95_abs_residual_max_m: float = 50.0
    close_rescue_cb_min_m: float = 16.0
    name: str = "rescue_branch_aware_hysteresis_quality_veto_regime_gate"
    style: str = "stateful-branch-rescue-hysteresis"
    _blocked_active: bool = field(default=False, init=False, repr=False, compare=False)
    _active_mode: str = field(default="", init=False, repr=False, compare=False)
    _close_streak: int = field(default=0, init=False, repr=False, compare=False)
    _far_streak: int = field(default=0, init=False, repr=False, compare=False)
    _clear_streak: int = field(default=0, init=False, repr=False, compare=False)

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
            "robust_p95_abs_residual",
            "satellite_count",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "close_satellite_max": float(self.close_satellite_max),
            "close_p95_abs_residual_max_m": float(self.close_p95_abs_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
            "enter_confirm_close_epochs": float(self.enter_confirm_close_epochs),
            "enter_confirm_far_epochs": float(self.enter_confirm_far_epochs),
            "exit_confirm_close_epochs": float(self.exit_confirm_close_epochs),
            "exit_confirm_far_epochs": float(self.exit_confirm_far_epochs),
            "close_rescue_satellite_max": float(self.close_rescue_satellite_max),
            "close_rescue_p95_abs_residual_max_m": float(self.close_rescue_p95_abs_residual_max_m),
            "close_rescue_cb_min_m": float(self.close_rescue_cb_min_m),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")
        p95_abs_residual = _feature(context, "robust_p95_abs_residual")
        satellites = _feature(context, "satellite_count")

        close_mode, far_mode, score = _quality_veto_components(
            blocked=blocked,
            positive=positive,
            disagreement=disagreement,
            cb_disagreement=cb_disagreement,
            residual=residual,
            p95_abs_residual=p95_abs_residual,
            satellites=satellites,
            close_blocked_low=self.close_blocked_low,
            close_blocked_high=self.close_blocked_high,
            close_disagreement_max_m=self.close_disagreement_max_m,
            close_cb_max_m=self.close_cb_max_m,
            close_residual_max_m=self.close_residual_max_m,
            close_satellite_max=self.close_satellite_max,
            close_p95_abs_residual_max_m=self.close_p95_abs_residual_max_m,
            far_blocked_max=self.far_blocked_max,
            far_positive_min=self.far_positive_min,
            far_disagreement_min_m=self.far_disagreement_min_m,
            far_cb_min_m=self.far_cb_min_m,
        )
        close_rescue = close_mode and (
            (
                satellites <= self.close_rescue_satellite_max
                and p95_abs_residual <= self.close_rescue_p95_abs_residual_max_m
            )
            or cb_disagreement >= self.close_rescue_cb_min_m
        )
        candidate = close_mode or far_mode
        self._close_streak = self._close_streak + 1 if close_mode else 0
        self._far_streak = self._far_streak + 1 if far_mode else 0
        self._clear_streak = 0 if candidate else self._clear_streak + 1

        if not self._blocked_active:
            if close_rescue:
                self._blocked_active = True
                self._active_mode = "close"
            elif self._close_streak >= self.enter_confirm_close_epochs:
                self._blocked_active = True
                self._active_mode = "close"
            elif self._far_streak >= self.enter_confirm_far_epochs:
                self._blocked_active = True
                self._active_mode = "far"
        else:
            exit_confirm_epochs = (
                self.exit_confirm_far_epochs
                if self._active_mode == "far"
                else self.exit_confirm_close_epochs
            )
            if self._clear_streak >= exit_confirm_epochs:
                self._blocked_active = False
                self._active_mode = ""

        return StrategyDecision(
            use_blocked=self._blocked_active,
            score=score,
            rationale=(
                f"active={self._blocked_active}, mode={self._active_mode or 'clear'}, "
                f"close={close_mode}, far={far_mode}, rescue={close_rescue}, "
                f"close_streak={self._close_streak}, far_streak={self._far_streak}, clear={self._clear_streak}, "
                f"blocked={blocked:.4f}, cb={cb_disagreement:.2f}, p95_abs={p95_abs_residual:.2f}, sats={satellites:.0f}"
            ),
        )


@dataclass
class NegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    close_satellite_max: float = 9.0
    close_p95_abs_residual_max_m: float = 55.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    enter_confirm_close_epochs: int = 3
    enter_confirm_far_epochs: int = 1
    exit_confirm_close_epochs: int = 3
    exit_confirm_far_epochs: int = 5
    close_rescue_satellite_max: float = 8.0
    close_rescue_p95_abs_residual_max_m: float = 50.0
    close_rescue_cb_min_m: float = 16.0
    negative_exit_disagreement_min_m: float = 42.0
    negative_exit_cb_min_m: float = 25.0
    negative_exit_p95_abs_residual_min_m: float = 52.0
    negative_exit_hits_required: int = 1
    name: str = "negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate"
    style: str = "stateful-branch-rescue-negative-exit"
    _blocked_active: bool = field(default=False, init=False, repr=False, compare=False)
    _active_mode: str = field(default="", init=False, repr=False, compare=False)
    _close_streak: int = field(default=0, init=False, repr=False, compare=False)
    _far_streak: int = field(default=0, init=False, repr=False, compare=False)
    _clear_streak: int = field(default=0, init=False, repr=False, compare=False)

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
            "robust_p95_abs_residual",
            "satellite_count",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "close_satellite_max": float(self.close_satellite_max),
            "close_p95_abs_residual_max_m": float(self.close_p95_abs_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
            "enter_confirm_close_epochs": float(self.enter_confirm_close_epochs),
            "enter_confirm_far_epochs": float(self.enter_confirm_far_epochs),
            "exit_confirm_close_epochs": float(self.exit_confirm_close_epochs),
            "exit_confirm_far_epochs": float(self.exit_confirm_far_epochs),
            "close_rescue_satellite_max": float(self.close_rescue_satellite_max),
            "close_rescue_p95_abs_residual_max_m": float(self.close_rescue_p95_abs_residual_max_m),
            "close_rescue_cb_min_m": float(self.close_rescue_cb_min_m),
            "negative_exit_disagreement_min_m": float(self.negative_exit_disagreement_min_m),
            "negative_exit_cb_min_m": float(self.negative_exit_cb_min_m),
            "negative_exit_p95_abs_residual_min_m": float(self.negative_exit_p95_abs_residual_min_m),
            "negative_exit_hits_required": float(self.negative_exit_hits_required),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")
        p95_abs_residual = _feature(context, "robust_p95_abs_residual")
        satellites = _feature(context, "satellite_count")

        close_mode, far_mode, score = _quality_veto_components(
            blocked=blocked,
            positive=positive,
            disagreement=disagreement,
            cb_disagreement=cb_disagreement,
            residual=residual,
            p95_abs_residual=p95_abs_residual,
            satellites=satellites,
            close_blocked_low=self.close_blocked_low,
            close_blocked_high=self.close_blocked_high,
            close_disagreement_max_m=self.close_disagreement_max_m,
            close_cb_max_m=self.close_cb_max_m,
            close_residual_max_m=self.close_residual_max_m,
            close_satellite_max=self.close_satellite_max,
            close_p95_abs_residual_max_m=self.close_p95_abs_residual_max_m,
            far_blocked_max=self.far_blocked_max,
            far_positive_min=self.far_positive_min,
            far_disagreement_min_m=self.far_disagreement_min_m,
            far_cb_min_m=self.far_cb_min_m,
        )
        close_rescue = close_mode and (
            (
                satellites <= self.close_rescue_satellite_max
                and p95_abs_residual <= self.close_rescue_p95_abs_residual_max_m
            )
            or cb_disagreement >= self.close_rescue_cb_min_m
        )
        candidate = close_mode or far_mode
        self._close_streak = self._close_streak + 1 if close_mode else 0
        self._far_streak = self._far_streak + 1 if far_mode else 0
        self._clear_streak = 0 if candidate else self._clear_streak + 1

        if not self._blocked_active:
            if close_rescue:
                self._blocked_active = True
                self._active_mode = "close"
            elif self._close_streak >= self.enter_confirm_close_epochs:
                self._blocked_active = True
                self._active_mode = "close"
            elif self._far_streak >= self.enter_confirm_far_epochs:
                self._blocked_active = True
                self._active_mode = "far"
        else:
            negative_exit_hits = (
                int(disagreement >= self.negative_exit_disagreement_min_m)
                + int(cb_disagreement >= self.negative_exit_cb_min_m)
                + int(p95_abs_residual >= self.negative_exit_p95_abs_residual_min_m)
            )
            if (
                self._active_mode == "close"
                and not candidate
                and negative_exit_hits >= self.negative_exit_hits_required
            ):
                self._blocked_active = False
                self._active_mode = ""
            else:
                exit_confirm_epochs = (
                    self.exit_confirm_far_epochs
                    if self._active_mode == "far"
                    else self.exit_confirm_close_epochs
                )
                if self._clear_streak >= exit_confirm_epochs:
                    self._blocked_active = False
                    self._active_mode = ""

        return StrategyDecision(
            use_blocked=self._blocked_active,
            score=score,
            rationale=(
                f"active={self._blocked_active}, mode={self._active_mode or 'clear'}, "
                f"close={close_mode}, far={far_mode}, rescue={close_rescue}, "
                f"neg_hits={negative_exit_hits if 'negative_exit_hits' in locals() else 0}, "
                f"close_streak={self._close_streak}, far_streak={self._far_streak}, clear={self._clear_streak}, "
                f"blocked={blocked:.4f}, cb={cb_disagreement:.2f}, p95_abs={p95_abs_residual:.2f}, sats={satellites:.0f}"
            ),
        )


@dataclass
class EntryVetoNegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy:
    close_blocked_low: float = 0.10
    close_blocked_high: float = 0.50
    close_disagreement_max_m: float = 40.0
    close_cb_max_m: float = 20.0
    close_residual_max_m: float = 22.0
    close_satellite_max: float = 9.0
    close_p95_abs_residual_max_m: float = 55.0
    far_blocked_max: float = 0.01
    far_positive_min: float = 0.15
    far_disagreement_min_m: float = 90.0
    far_cb_min_m: float = 45.0
    enter_confirm_close_epochs: int = 3
    enter_confirm_far_epochs: int = 1
    exit_confirm_close_epochs: int = 3
    exit_confirm_far_epochs: int = 5
    close_rescue_satellite_max: float = 8.0
    close_rescue_p95_abs_residual_max_m: float = 50.0
    close_rescue_cb_min_m: float = 16.0
    close_entry_p95_abs_residual_max_m: float = 50.0
    negative_exit_disagreement_min_m: float = 42.0
    negative_exit_cb_min_m: float = 25.0
    negative_exit_p95_abs_residual_min_m: float = 52.0
    negative_exit_hits_required: int = 1
    name: str = "entry_veto_negative_exit_rescue_branch_aware_hysteresis_quality_veto_regime_gate"
    style: str = "stateful-branch-entry-veto-rescue-negative-exit"
    _blocked_active: bool = field(default=False, init=False, repr=False, compare=False)
    _active_mode: str = field(default="", init=False, repr=False, compare=False)
    _close_streak: int = field(default=0, init=False, repr=False, compare=False)
    _far_streak: int = field(default=0, init=False, repr=False, compare=False)
    _clear_streak: int = field(default=0, init=False, repr=False, compare=False)

    def required_features(self) -> tuple[str, ...]:
        return (
            "mean_weighted_blocked_frac",
            "blocked_positive_frac_gt5",
            "disagreement_m",
            "cb_disagreement_m",
            "robust_mean_abs_residual",
            "robust_p95_abs_residual",
            "satellite_count",
        )

    def parameters(self) -> dict[str, float]:
        return {
            "close_blocked_low": float(self.close_blocked_low),
            "close_blocked_high": float(self.close_blocked_high),
            "close_disagreement_max_m": float(self.close_disagreement_max_m),
            "close_cb_max_m": float(self.close_cb_max_m),
            "close_residual_max_m": float(self.close_residual_max_m),
            "close_satellite_max": float(self.close_satellite_max),
            "close_p95_abs_residual_max_m": float(self.close_p95_abs_residual_max_m),
            "far_blocked_max": float(self.far_blocked_max),
            "far_positive_min": float(self.far_positive_min),
            "far_disagreement_min_m": float(self.far_disagreement_min_m),
            "far_cb_min_m": float(self.far_cb_min_m),
            "enter_confirm_close_epochs": float(self.enter_confirm_close_epochs),
            "enter_confirm_far_epochs": float(self.enter_confirm_far_epochs),
            "exit_confirm_close_epochs": float(self.exit_confirm_close_epochs),
            "exit_confirm_far_epochs": float(self.exit_confirm_far_epochs),
            "close_rescue_satellite_max": float(self.close_rescue_satellite_max),
            "close_rescue_p95_abs_residual_max_m": float(self.close_rescue_p95_abs_residual_max_m),
            "close_rescue_cb_min_m": float(self.close_rescue_cb_min_m),
            "close_entry_p95_abs_residual_max_m": float(self.close_entry_p95_abs_residual_max_m),
            "negative_exit_disagreement_min_m": float(self.negative_exit_disagreement_min_m),
            "negative_exit_cb_min_m": float(self.negative_exit_cb_min_m),
            "negative_exit_p95_abs_residual_min_m": float(self.negative_exit_p95_abs_residual_min_m),
            "negative_exit_hits_required": float(self.negative_exit_hits_required),
        }

    def decide(self, context: StrategyContext) -> StrategyDecision:
        blocked = _feature(context, "mean_weighted_blocked_frac")
        positive = _feature(context, "blocked_positive_frac_gt5")
        disagreement = _feature(context, "disagreement_m")
        cb_disagreement = _feature(context, "cb_disagreement_m")
        residual = _feature(context, "robust_mean_abs_residual")
        p95_abs_residual = _feature(context, "robust_p95_abs_residual")
        satellites = _feature(context, "satellite_count")

        close_mode, far_mode, score = _quality_veto_components(
            blocked=blocked,
            positive=positive,
            disagreement=disagreement,
            cb_disagreement=cb_disagreement,
            residual=residual,
            p95_abs_residual=p95_abs_residual,
            satellites=satellites,
            close_blocked_low=self.close_blocked_low,
            close_blocked_high=self.close_blocked_high,
            close_disagreement_max_m=self.close_disagreement_max_m,
            close_cb_max_m=self.close_cb_max_m,
            close_residual_max_m=self.close_residual_max_m,
            close_satellite_max=self.close_satellite_max,
            close_p95_abs_residual_max_m=self.close_p95_abs_residual_max_m,
            far_blocked_max=self.far_blocked_max,
            far_positive_min=self.far_positive_min,
            far_disagreement_min_m=self.far_disagreement_min_m,
            far_cb_min_m=self.far_cb_min_m,
        )
        close_rescue = close_mode and (
            (
                satellites <= self.close_rescue_satellite_max
                and p95_abs_residual <= self.close_rescue_p95_abs_residual_max_m
            )
            or cb_disagreement >= self.close_rescue_cb_min_m
        )
        close_entry = close_mode and (
            p95_abs_residual <= self.close_entry_p95_abs_residual_max_m
        )
        candidate = close_mode or far_mode
        self._close_streak = self._close_streak + 1 if close_entry else 0
        self._far_streak = self._far_streak + 1 if far_mode else 0
        self._clear_streak = 0 if candidate else self._clear_streak + 1

        if not self._blocked_active:
            if close_rescue:
                self._blocked_active = True
                self._active_mode = "close"
            elif self._close_streak >= self.enter_confirm_close_epochs:
                self._blocked_active = True
                self._active_mode = "close"
            elif self._far_streak >= self.enter_confirm_far_epochs:
                self._blocked_active = True
                self._active_mode = "far"
        else:
            negative_exit_hits = (
                int(disagreement >= self.negative_exit_disagreement_min_m)
                + int(cb_disagreement >= self.negative_exit_cb_min_m)
                + int(p95_abs_residual >= self.negative_exit_p95_abs_residual_min_m)
            )
            if (
                self._active_mode == "close"
                and not candidate
                and negative_exit_hits >= self.negative_exit_hits_required
            ):
                self._blocked_active = False
                self._active_mode = ""
            else:
                exit_confirm_epochs = (
                    self.exit_confirm_far_epochs
                    if self._active_mode == "far"
                    else self.exit_confirm_close_epochs
                )
                if self._clear_streak >= exit_confirm_epochs:
                    self._blocked_active = False
                    self._active_mode = ""

        return StrategyDecision(
            use_blocked=self._blocked_active,
            score=score,
            rationale=(
                f"active={self._blocked_active}, mode={self._active_mode or 'clear'}, "
                f"close={close_mode}, far={far_mode}, rescue={close_rescue}, "
                f"entry_close={close_entry}, "
                f"neg_hits={negative_exit_hits if 'negative_exit_hits' in locals() else 0}, "
                f"close_streak={self._close_streak}, far_streak={self._far_streak}, clear={self._clear_streak}, "
                f"blocked={blocked:.4f}, cb={cb_disagreement:.2f}, p95_abs={p95_abs_residual:.2f}, sats={satellites:.0f}"
            ),
        )


def default_strategies():
    return (
        AlwaysRobustStrategy(),
        AlwaysBlockedStrategy(),
        DisagreementGateStrategy(disagreement_threshold_m=80.0),
        ClockVetoGateStrategy(
            disagreement_threshold_m=85.0,
            cb_disagreement_threshold_m=30.0,
            blocked_ceiling=0.03,
        ),
        DualModeRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
        ),
        QualityVetoRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            close_satellite_max=9.0,
            close_p95_abs_residual_max_m=55.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
        ),
        HysteresisQualityVetoRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            close_satellite_max=9.0,
            close_p95_abs_residual_max_m=55.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
            enter_confirm_epochs=1,
            exit_confirm_epochs=3,
        ),
        ModeAwareHysteresisQualityVetoRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            close_satellite_max=9.0,
            close_p95_abs_residual_max_m=55.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
            enter_confirm_close_epochs=2,
            enter_confirm_far_epochs=1,
            exit_confirm_epochs=4,
        ),
        BranchAwareHysteresisQualityVetoRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            close_satellite_max=9.0,
            close_p95_abs_residual_max_m=55.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
            enter_confirm_close_epochs=2,
            enter_confirm_far_epochs=1,
            exit_confirm_close_epochs=3,
            exit_confirm_far_epochs=5,
        ),
        RescueBranchAwareHysteresisQualityVetoRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            close_satellite_max=9.0,
            close_p95_abs_residual_max_m=55.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
            enter_confirm_close_epochs=3,
            enter_confirm_far_epochs=1,
            exit_confirm_close_epochs=3,
            exit_confirm_far_epochs=5,
            close_rescue_satellite_max=8.0,
            close_rescue_p95_abs_residual_max_m=50.0,
            close_rescue_cb_min_m=16.0,
        ),
        NegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            close_satellite_max=9.0,
            close_p95_abs_residual_max_m=55.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
            enter_confirm_close_epochs=3,
            enter_confirm_far_epochs=1,
            exit_confirm_close_epochs=3,
            exit_confirm_far_epochs=5,
            close_rescue_satellite_max=8.0,
            close_rescue_p95_abs_residual_max_m=50.0,
            close_rescue_cb_min_m=16.0,
            negative_exit_disagreement_min_m=42.0,
            negative_exit_cb_min_m=25.0,
            negative_exit_p95_abs_residual_min_m=52.0,
            negative_exit_hits_required=1,
        ),
        EntryVetoNegativeExitRescueBranchAwareHysteresisQualityVetoRegimeGateStrategy(
            close_blocked_low=0.10,
            close_blocked_high=0.50,
            close_disagreement_max_m=40.0,
            close_cb_max_m=20.0,
            close_residual_max_m=22.0,
            close_satellite_max=9.0,
            close_p95_abs_residual_max_m=55.0,
            far_blocked_max=0.01,
            far_positive_min=0.15,
            far_disagreement_min_m=90.0,
            far_cb_min_m=45.0,
            enter_confirm_close_epochs=3,
            enter_confirm_far_epochs=1,
            exit_confirm_close_epochs=3,
            exit_confirm_far_epochs=5,
            close_rescue_satellite_max=8.0,
            close_rescue_p95_abs_residual_max_m=50.0,
            close_rescue_cb_min_m=16.0,
            close_entry_p95_abs_residual_max_m=50.0,
            negative_exit_disagreement_min_m=42.0,
            negative_exit_cb_min_m=25.0,
            negative_exit_p95_abs_residual_min_m=52.0,
            negative_exit_hits_required=1,
        ),
        RuleChainGateStrategy(
            blocked_threshold=0.001,
            positive_threshold=0.25,
            disagreement_threshold_m=80.0,
        ),
        WeightedScoreGateStrategy(
            blocked_scale=0.01,
            positive_scale=0.50,
            disagreement_scale_m=80.0,
            threshold=1.60,
        ),
    )
