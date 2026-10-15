"""Preserve the original 3D FP64 mixture across chunks and native transfers."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_mixture_coordinates_cases import (
    GPU_COMPONENTS, GPU_COUNTS, check_cpu, check_error_policy, check_gpu, check_rejection, cpu_ids, policy_ids, rejection_arguments,
)
from gnss_gpu.pf_device_runtime import ParticleFilterDeviceRuntime
from gnss_gpu.particle_filter_device import ParticleFilterDevice


@pytest.mark.parametrize('components,count,variant,ieee', list(cpu_ids()))
def test_exact_3d_oracle_across_chunks_and_ieee_bits(components, count, variant, ieee):
    check_cpu(ParticleFilterDeviceRuntime.position_mixture_update, components, count, variant, ieee)


@pytest.mark.parametrize('arguments', list(rejection_arguments()))
def test_original_rejections_preserve_weights(arguments):
    check_rejection(ParticleFilterDeviceRuntime.position_mixture_update, arguments)


@pytest.mark.parametrize('count', GPU_COUNTS)
@pytest.mark.parametrize('components', GPU_COMPONENTS)
@pytest.mark.parametrize('variant', (0, 1))
def test_actual_native_states_weights_queries_match_original(count, components, variant):
    native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    check_gpu(native, ParticleFilterDevice, count, components, variant)


@pytest.mark.parametrize("scenario,mode", list(policy_ids()))
def test_original_strict_numpy_error_and_warning_handler_order(scenario, mode):
    check_error_policy(ParticleFilterDeviceRuntime.position_mixture_update, scenario, mode)
