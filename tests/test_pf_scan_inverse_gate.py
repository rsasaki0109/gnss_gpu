"""All16 selected state bits and fallback on each side of the source gate."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_scan_inverse_gate_cases import COUNTS,TARGETS,VARIANTS
from pf_uniform_inverse_cases import check_inverse

@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('target',TARGETS)
@pytest.mark.parametrize('variant',VARIANTS)
def test_exact_selection_across_large_source_gate(count,target,variant):
    native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')
    from gnss_gpu.particle_filter_device import ParticleFilterDevice
    check_inverse(native,ParticleFilterDevice,count,target,variant)
