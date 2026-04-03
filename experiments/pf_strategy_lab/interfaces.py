from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol


@dataclass(frozen=True)
class StrategyContext:
    segment_label: str
    epoch: int
    features: Mapping[str, float]


@dataclass(frozen=True)
class StrategyDecision:
    use_blocked: bool
    score: float
    rationale: str


class GateStrategy(Protocol):
    name: str
    style: str

    def required_features(self) -> tuple[str, ...]:
        ...

    def parameters(self) -> Mapping[str, float]:
        ...

    def decide(self, context: StrategyContext) -> StrategyDecision:
        ...

