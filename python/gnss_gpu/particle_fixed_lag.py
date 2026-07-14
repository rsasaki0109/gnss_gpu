"""Mode-conditioned fixed-lag smoothing through PF resampling genealogy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from gnss_gpu.particle_ffbsi import ffbsi_sample_indices, genealogy_smooth_indices


@dataclass(frozen=True)
class FixedLagSmootherOutput:
    epoch_index: int
    position: np.ndarray
    covariance: np.ndarray
    path_count: int
    unique_oldest_particles: int
    terminal_particle_count: int
    terminal_conditioned: bool
    rewrite_allowed: bool
    smoother_mode: str = "genealogy"


@dataclass
class _HistoryFrame:
    epoch_index: int
    particles: np.ndarray
    log_weights: np.ndarray
    ancestors: np.ndarray
    terminal_mask: np.ndarray | None
    rewrite_allowed: bool
    velocities: np.ndarray
    dt: float
    sigma_pos: float


def nearest_mode_mask(
    particles: np.ndarray,
    mode_positions: np.ndarray,
    selected_mode_index: int,
) -> np.ndarray:
    """Assign every finite particle to its nearest mode and return one mode."""
    xyz = np.asarray(particles, dtype=np.float64)
    modes = np.asarray(mode_positions, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("particles must have shape (N, >=3)")
    if modes.ndim != 2 or modes.shape[1] != 3 or modes.shape[0] == 0:
        raise ValueError("mode_positions must have shape (K, 3), K > 0")
    if not 0 <= int(selected_mode_index) < modes.shape[0]:
        raise ValueError("selected_mode_index is out of range")
    finite = np.all(np.isfinite(xyz[:, :3]), axis=1)
    mask = np.zeros(xyz.shape[0], dtype=bool)
    if not np.any(finite):
        return mask
    distances = np.linalg.norm(
        xyz[finite, None, :3] - modes[None, :, :], axis=2
    )
    labels = np.argmin(distances, axis=1)
    mask[np.flatnonzero(finite)] = labels == int(selected_mode_index)
    return mask


class FixedLagParticleSmoother:
    """Bounded-memory ancestor smoother.

    Each frame must be captured after all measurement updates and before the
    epoch-end systematic resample. ``ancestors`` maps the post-resample slots
    to that frame's pre-resample slots. Doppler is never replayed backward: its
    effect is already represented by the forward particle state and ancestry.
    """

    def __init__(
        self,
        lag_epochs: int = 25,
        n_paths: int = 32,
        seed: int = 20260713,
        *,
        smoother_mode: str = "marginal",
        sigma_cb: float = 50.0,
    ):
        if lag_epochs < 1:
            raise ValueError("lag_epochs must be at least one")
        if n_paths < 1:
            raise ValueError("n_paths must be at least one")
        if smoother_mode not in {"genealogy", "marginal"}:
            raise ValueError("smoother_mode must be 'genealogy' or 'marginal'")
        self.lag_epochs = int(lag_epochs)
        self.n_paths = int(n_paths)
        self.smoother_mode = str(smoother_mode)
        self.sigma_cb = float(sigma_cb)
        self._rng = np.random.default_rng(int(seed))
        self._frames: deque[_HistoryFrame] = deque()

    @property
    def buffered_epochs(self) -> int:
        return len(self._frames)

    def append(
        self,
        epoch_index: int,
        particles: np.ndarray,
        log_weights: np.ndarray,
        ancestors: np.ndarray,
        *,
        terminal_mask: np.ndarray | None = None,
        rewrite_allowed: bool = True,
        velocities: np.ndarray | None = None,
        dt: float = 1.0,
        sigma_pos: float = 1.0,
    ) -> FixedLagSmootherOutput | None:
        particles_arr = np.asarray(particles, dtype=np.float64)
        log_weights_arr = np.asarray(log_weights, dtype=np.float64).reshape(-1)
        ancestors_arr = np.asarray(ancestors, dtype=np.int64).reshape(-1)
        if particles_arr.ndim != 2 or particles_arr.shape[1] < 4:
            raise ValueError("particles must have shape (N, >=4)")
        n_particles = particles_arr.shape[0]
        if log_weights_arr.size != n_particles or ancestors_arr.size != n_particles:
            raise ValueError("particles, log_weights, and ancestors lengths must match")
        if velocities is None:
            velocities_arr = np.zeros((n_particles, 3), dtype=np.float64)
        else:
            velocities_arr = np.asarray(velocities, dtype=np.float64)
            if velocities_arr.shape != (n_particles, 3):
                raise ValueError("velocities must have shape (N, 3)")
        if dt < 0.0 or sigma_pos <= 0.0:
            raise ValueError("dt must be non-negative and sigma_pos must be positive")
        mask_arr = None
        if terminal_mask is not None:
            mask_arr = np.asarray(terminal_mask, dtype=bool).reshape(-1)
            if mask_arr.size != n_particles:
                raise ValueError("terminal_mask length must match particles")
            if not np.any(mask_arr):
                mask_arr = None
                rewrite_allowed = False
        self._frames.append(
            _HistoryFrame(
                int(epoch_index),
                particles_arr.copy(),
                log_weights_arr.copy(),
                ancestors_arr.copy(),
                None if mask_arr is None else mask_arr.copy(),
                bool(rewrite_allowed),
                velocities_arr.copy(),
                float(dt),
                float(sigma_pos),
            )
        )
        if len(self._frames) <= self.lag_epochs:
            return None
        return self._smooth_and_pop_oldest()

    def flush(self) -> list[FixedLagSmootherOutput]:
        outputs: list[FixedLagSmootherOutput] = []
        while self._frames:
            outputs.append(self._smooth_and_pop_oldest())
        return outputs

    def _smooth_and_pop_oldest(self) -> FixedLagSmootherOutput:
        frames = list(self._frames)
        particles = np.stack([frame.particles for frame in frames], axis=0)
        log_weights = np.stack([frame.log_weights for frame in frames], axis=0)
        ancestors = np.stack([frame.ancestors for frame in frames], axis=0)
        velocities = np.stack([frame.velocities for frame in frames], axis=0)
        dt = np.asarray([frame.dt for frame in frames], dtype=np.float64)
        sigma_pos = np.asarray([frame.sigma_pos for frame in frames], dtype=np.float64)
        terminal = frames[-1]

        oldest_indices = np.empty(self.n_paths, dtype=np.int64)
        oldest_positions = np.empty((self.n_paths, 3), dtype=np.float64)
        for path_index in range(self.n_paths):
            if self.smoother_mode == "genealogy":
                indices = genealogy_smooth_indices(
                    log_weights,
                    ancestors,
                    self._rng,
                    terminal_mask=terminal.terminal_mask,
                )
            else:
                indices = ffbsi_sample_indices(
                    log_weights,
                    particles[:, :, :4],
                    velocities,
                    dt,
                    sigma_pos,
                    self.sigma_cb,
                    self._rng,
                    terminal_mask=terminal.terminal_mask,
                )
            oldest_indices[path_index] = indices[0]
            oldest_positions[path_index] = particles[0, indices[0], :3]

        position = np.mean(oldest_positions, axis=0)
        if self.n_paths > 1:
            covariance = np.cov(oldest_positions, rowvar=False, ddof=1)
        else:
            covariance = np.zeros((3, 3), dtype=np.float64)
        output = FixedLagSmootherOutput(
            epoch_index=frames[0].epoch_index,
            position=position,
            covariance=np.asarray(covariance, dtype=np.float64).reshape(3, 3),
            path_count=self.n_paths,
            unique_oldest_particles=int(np.unique(oldest_indices).size),
            terminal_particle_count=(
                int(np.count_nonzero(terminal.terminal_mask))
                if terminal.terminal_mask is not None
                else int(terminal.particles.shape[0])
            ),
            terminal_conditioned=terminal.terminal_mask is not None,
            rewrite_allowed=bool(terminal.rewrite_allowed),
            smoother_mode=self.smoother_mode,
        )
        self._frames.popleft()
        return output


# Backward-compatible name for the original genealogy-only milestone API.
FixedLagGenealogySmoother = FixedLagParticleSmoother
