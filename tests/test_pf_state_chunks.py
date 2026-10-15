"""Exact full-state import/export and scratch lifetime across PF operations."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from pf_state_chunks_cases import COUNTS,METHODS,check_state,check_history,check_representation
from gnss_gpu.particle_filter_device import ParticleFilterDevice

native=pytest.importorskip('gnss_gpu._gnss_gpu_pf_device')


@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('variant',range(8))
def test_all16_import_export_bits_ess_reset_and_owned_arrays(count,variant):
    check_state(native,ParticleFilterDevice,count,variant)


@pytest.mark.parametrize('count',COUNTS)
@pytest.mark.parametrize('common',(False,True))
@pytest.mark.parametrize('method',METHODS)
def test_checkpoint_scratch_with_queued_prediction_gather_kf_geometry(count,common,method):
    check_history(native,ParticleFilterDevice,count,common,method)


@pytest.mark.parametrize('count',[n for n in COUNTS if n<=257])
@pytest.mark.parametrize('common',(False,True))
def test_checkpoint_scratch_with_systematic_resampling(count,common):
    check_history(native,ParticleFilterDevice,count,common,'systematic')


@pytest.mark.parametrize('form',('C','F','strided'))
def test_host_input_representation_and_completion(form):
    check_representation(native,ParticleFilterDevice,form)
