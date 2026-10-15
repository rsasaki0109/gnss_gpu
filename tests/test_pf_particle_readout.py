"""Particle export ownership, IEEE bits, and subsequent scratch consumers."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_particle_readout_cases import COUNTS, METHODS, check_export, check_history
from gnss_gpu.particle_filter_device import ParticleFilterDevice

native = pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('variant', range(8))
def test_export_preserves_all_bits_weights_ess_and_owned_output(count, variant):
    check_export(native, ParticleFilterDevice, count, variant)


@pytest.mark.parametrize('count', COUNTS)
@pytest.mark.parametrize('common', (False, True))
@pytest.mark.parametrize('method', METHODS)
def test_export_scratch_is_safe_across_queued_pf_history(count, common, method):
    check_history(native, ParticleFilterDevice, count, common, method)


@pytest.mark.parametrize('count', [n for n in COUNTS if n <= 257])
@pytest.mark.parametrize('common', (False, True))
def test_export_scratch_with_systematic_resampling(count, common):
    check_history(native, ParticleFilterDevice, count, common, 'systematic')
