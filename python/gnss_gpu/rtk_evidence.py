"""Truth-free RTK evidence provenance and deterministic FIX replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Iterable

from gnss_gpu.rtk_fix_gate import TrustedFixGateDecision, trusted_fix_gate


class EvidenceAuditError(ValueError):
    """Raised when an observation is duplicated or only partly consumed."""


@dataclass(frozen=True)
class EvidenceRecord:
    epoch: int
    target: str
    source: str
    observation_id: str
    stage: int
    beta: float
    n_rows: int
    log_evidence: float | None = None

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.stage < 0 or self.n_rows < 0:
            raise ValueError("epoch, stage, and n_rows must be non-negative")
        if not self.target or not self.source or not self.observation_id:
            raise ValueError("target, source, and observation_id are required")
        if not math.isfinite(self.beta) or not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must be finite and in (0, 1]")
        if self.log_evidence is not None and not math.isfinite(self.log_evidence):
            raise ValueError("log_evidence must be finite when present")

    @property
    def group_key(self) -> tuple[int, str, str, str]:
        return (self.epoch, self.target, self.source, self.observation_id)

    @property
    def stage_key(self) -> tuple[int, str, str, str, int]:
        return (*self.group_key, self.stage)


@dataclass(frozen=True)
class EvidenceAudit:
    n_records: int
    n_updates: int
    beta_error_count: int


class EvidenceLedger:
    """Append-only provenance ledger with strict likelihood-consumption audit."""

    def __init__(self, *, beta_tolerance: float = 1.0e-9) -> None:
        self.beta_tolerance = float(beta_tolerance)
        self._records: list[EvidenceRecord] = []
        self._stage_keys: set[tuple[int, str, str, str, int]] = set()

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def append(self, record: EvidenceRecord) -> None:
        if record.stage_key in self._stage_keys:
            raise EvidenceAuditError(f"duplicate evidence stage: {record.stage_key}")
        self._stage_keys.add(record.stage_key)
        self._records.append(record)

    def record(
        self,
        *,
        epoch: int,
        target: str,
        source: str,
        observation_id: str,
        beta: float = 1.0,
        stage: int = 0,
        n_rows: int = 0,
        log_evidence: float | None = None,
    ) -> None:
        self.append(
            EvidenceRecord(
                epoch=epoch,
                target=target,
                source=source,
                observation_id=observation_id,
                stage=stage,
                beta=beta,
                n_rows=n_rows,
                log_evidence=log_evidence,
            )
        )

    def audit(self, *, require_complete: bool = True) -> EvidenceAudit:
        beta_by_update: dict[tuple[int, str, str, str], float] = {}
        for record in self._records:
            beta_by_update[record.group_key] = (
                beta_by_update.get(record.group_key, 0.0) + record.beta
            )
        failures = {
            key: beta
            for key, beta in beta_by_update.items()
            if beta > 1.0 + self.beta_tolerance
            or (require_complete and abs(beta - 1.0) > self.beta_tolerance)
        }
        if failures:
            preview = ", ".join(f"{key}={beta:.12g}" for key, beta in list(failures.items())[:3])
            raise EvidenceAuditError(f"invalid evidence beta totals: {preview}")
        return EvidenceAudit(
            n_records=len(self._records),
            n_updates=len(beta_by_update),
            beta_error_count=0,
        )

    def rows(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self._records]


@dataclass(frozen=True)
class TrustedFixPolicyConfig:
    gamma_threshold: float = 0.99
    min_streak: int = 3
    min_ambiguities: int = 8
    max_float_separation_m: float = 0.5
    max_ddpr_separation_m: float = 1.75
    min_ddpr_pairs: int = 9
    max_ddpr_age_epochs: int = 4


@dataclass(frozen=True)
class RTKEpochTrace:
    """Compact truth-free inputs and result for one trusted commit decision."""

    epoch: int
    tow: float
    assignment_id: str
    gamma: float
    n_ambiguities: int
    map_float_separation_m: float
    map_ddpr_separation_m: float
    last_ddpr_pairs: int
    ddpr_age_epochs: int
    ecef_x: float
    ecef_y: float
    ecef_z: float
    gamma_eligible: bool
    fix_streak: int
    fixed: bool
    evidence_records: int = 0

    def policy_input(self) -> "TrustedFixPolicyInput":
        return TrustedFixPolicyInput(
            epoch=self.epoch,
            assignment_id=self.assignment_id,
            gamma=self.gamma,
            n_ambiguities=self.n_ambiguities,
            map_float_separation_m=self.map_float_separation_m,
            map_ddpr_separation_m=self.map_ddpr_separation_m,
            last_ddpr_pairs=self.last_ddpr_pairs,
            ddpr_age_epochs=self.ddpr_age_epochs,
        )

    def row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrustedFixPolicyInput:
    epoch: int
    assignment_id: str
    gamma: float
    n_ambiguities: int
    map_float_separation_m: float
    map_ddpr_separation_m: float
    last_ddpr_pairs: int
    ddpr_age_epochs: int


@dataclass(frozen=True)
class TrustedFixPolicyDecision:
    fixed: bool
    gamma_eligible: bool
    fix_streak: int
    gate: TrustedFixGateDecision


class TrustedFixCommitPolicy:
    """Stateful trusted FIX policy whose inputs can be deterministically replayed."""

    def __init__(self, config: TrustedFixPolicyConfig) -> None:
        self.config = config
        self._last_epoch: int | None = None
        self._last_assignment = ""
        self._streak = 0

    def evaluate(self, value: TrustedFixPolicyInput) -> TrustedFixPolicyDecision:
        if self._last_epoch is not None and value.epoch <= self._last_epoch:
            raise ValueError("FIX policy epochs must be strictly increasing")
        eligible = bool(
            value.assignment_id
            and value.n_ambiguities >= self.config.min_ambiguities
            and math.isfinite(value.gamma)
            and value.gamma > self.config.gamma_threshold
        )
        consecutive = self._last_epoch is None or value.epoch == self._last_epoch + 1
        if eligible and consecutive and value.assignment_id == self._last_assignment:
            self._streak += 1
        elif eligible:
            self._streak = 1
        else:
            self._streak = 0
        self._last_epoch = int(value.epoch)
        self._last_assignment = value.assignment_id
        gate = trusted_fix_gate(
            map_float_separation_m=value.map_float_separation_m,
            map_ddpr_separation_m=value.map_ddpr_separation_m,
            last_ddpr_pairs=value.last_ddpr_pairs,
            ddpr_age_epochs=value.ddpr_age_epochs,
            max_float_separation_m=self.config.max_float_separation_m,
            max_ddpr_separation_m=self.config.max_ddpr_separation_m,
            min_ddpr_pairs=self.config.min_ddpr_pairs,
            max_ddpr_age_epochs=self.config.max_ddpr_age_epochs,
        )
        return TrustedFixPolicyDecision(
            fixed=bool(eligible and self._streak >= self.config.min_streak and gate.passed),
            gamma_eligible=eligible,
            fix_streak=self._streak,
            gate=gate,
        )


def replay_fix_decisions(
    traces: Iterable[RTKEpochTrace], config: TrustedFixPolicyConfig
) -> list[TrustedFixPolicyDecision]:
    policy = TrustedFixCommitPolicy(config)
    return [policy.evaluate(trace.policy_input()) for trace in traces]


def ambiguity_assignment_id(assignment: Iterable[object]) -> str:
    """Return a stable compact identity for a canonical versioned assignment."""

    normalized = _normalized_assignment(assignment)
    if not normalized:
        return ""
    payload = json.dumps(normalized, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def ambiguity_assignment_json(assignment: Iterable[object]) -> str:
    """Serialize a canonical assignment without Python-specific repr syntax."""

    return json.dumps(
        _normalized_assignment(assignment), separators=(",", ":"), ensure_ascii=True
    )


def ambiguity_assignment_from_json(payload: str) -> tuple[object, ...]:
    """Deserialize :func:`ambiguity_assignment_json` for temporal replay."""

    values = json.loads(payload)
    return tuple(
        ((((str(ref_sat), str(sat), int(wavelength_nm)), int(generation))), int(integer))
        for ref_sat, sat, wavelength_nm, generation, integer in values
    )


def _normalized_assignment(assignment: Iterable[object]) -> list[list[object]]:
    normalized: list[list[object]] = []
    for item in assignment:
        ((ref_sat, sat, wavelength_nm), generation), integer = item  # type: ignore[misc]
        normalized.append(
            [
                str(ref_sat),
                str(sat),
                int(wavelength_nm),
                int(generation),
                int(integer),
            ]
        )
    return normalized
