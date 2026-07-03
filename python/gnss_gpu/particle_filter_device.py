"""GPU-accelerated Particle Filter with persistent device memory.

Particle state lives on GPU. No H2D/D2H transfers except:
- Satellite data per update (small: ~1KB for 8 satellites)
- Estimate output (32 bytes)
- Velocity guide per predict (24 bytes)
- Particle dump for visualization (on-demand)

This eliminates the #1 performance bottleneck: cudaMalloc/cudaFree and
full particle array H2D/D2H transfers on every call.

Implementation is split across ``pf_device_config``, ``pf_device_runtime``,
and ``pf_device_smoother``; this module re-exports the public class.
"""

from __future__ import annotations

from gnss_gpu.pf_device_runtime import ParticleFilterDeviceRuntime
from gnss_gpu.pf_device_smoother import ParticleFilterDeviceSmootherMixin


class ParticleFilterDevice(ParticleFilterDeviceSmootherMixin, ParticleFilterDeviceRuntime):
    """High-performance particle filter with persistent GPU memory."""

    pass


__all__ = ["ParticleFilterDevice"]
