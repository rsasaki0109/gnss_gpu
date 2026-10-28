import numpy as np
import pytest
from experiments.run_cpu_coherent_multipath import synthesize_blocks


def test_nested_coherence_preserves_signal_and_noise_psd():
    signal, blocks, delay = synthesize_blocks(3.6,.8,.1,.7,81)
    assert blocks.shape == (10,len(signal)) and -.3 <= delay <= .3
    noise = blocks-signal
    expected = 1/(10**3.7*.020)
    for n in (1,4,10):
        measured = np.mean(abs(noise[:n].mean(axis=0))**2)
        assert measured == pytest.approx(expected/n,rel=.06)
        np.testing.assert_allclose(blocks[:n].mean(axis=0),signal+noise[:n].mean(axis=0),atol=1e-15)
    # A single arbitrary phase shared by all observations does not destroy coherence.
    np.testing.assert_allclose((blocks*np.exp(.7j)).mean(axis=0),blocks.mean(axis=0)*np.exp(.7j),atol=1e-15)


def test_shared_harmonics_are_exactly_paired_across_bandwidths():
    narrow, small, delay = synthesize_blocks(.9,.8,.5,0.,82)
    wide, large, delay2 = synthesize_blocks(3.6,.8,.5,0.,82)
    start = (len(wide)-len(narrow))//2
    assert delay==delay2
    np.testing.assert_array_equal(small,large[:,start:start+len(narrow)])
    np.testing.assert_array_equal(narrow,wide[start:start+len(narrow)])


def test_distinct_replica_noise_and_exact_replay():
    a=synthesize_blocks(.9,0.,0.,0.,83)
    b=synthesize_blocks(.9,0.,0.,0.,84)
    replay=synthesize_blocks(.9,0.,0.,0.,83)
    np.testing.assert_array_equal(a[1],replay[1])
    assert not np.array_equal(a[1],b[1])
