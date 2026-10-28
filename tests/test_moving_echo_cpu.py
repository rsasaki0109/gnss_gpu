import numpy as np
import pytest
from gnss_gpu.moving_echo_cpu import moving_echo_blocks,range_rate_from_doppler,L1_FREQUENCY_HZ
from gnss_gpu.multipath_information_cpu import LIGHT_SPEED
from experiments.run_cpu_moving_echo_pilot import synthesize


def test_zero_drift_integral_and_motion_sign():
    f=np.array([-7e6,0.,7e6]);fd=7.;rate=range_rate_from_doppler(fd)
    assert rate<0 and np.isclose(-L1_FREQUENCY_HZ*rate/LIGHT_SPEED,fd)
    effective=.3-(L1_FREQUENCY_HZ+f)*rate/LIGHT_SPEED
    centers=(np.arange(40)+.5)*.02
    expected=np.exp(2j*np.pi*centers[:,None]*effective)*np.sinc(effective*.02)
    np.testing.assert_allclose(moving_echo_blocks(f,.3,0.,rate),expected,rtol=1e-12,atol=1e-12)


def test_independent_time_varying_delay_quadrature():
    f=np.array([-7e6,0.,7e6]);rate=range_rate_from_doppler(-7.)
    actual=moving_echo_blocks(f,.2,-.7,rate)
    for k in (0,19,39):
        t=k*.02+(np.arange(50000)+.5)*.02/50000
        delay=rate*t/LIGHT_SPEED
        phase=2j*np.pi*(.2*t[:,None]-.35*t[:,None]**2)-2j*np.pi*(L1_FREQUENCY_HZ+f)*delay[:,None]
        np.testing.assert_allclose(actual[k],np.mean(np.exp(phase),axis=0),rtol=1e-10,atol=1e-11)
    with pytest.raises(ValueError):moving_echo_blocks([],0,0,0)


def test_zero_motion_exact_and_paired_noise():
    fixed=synthesize(('fixed_code',0.,.5,-.3,1330000));moving=synthesize(('moving_code',0.,.5,-.3,1330000))
    for a,b in zip(fixed,moving):np.testing.assert_array_equal(a,b)
    b,s,d=synthesize(('fixed_code',7.,.5,-.3,1330000))
    other,changed,truth=synthesize(('moving_code',7.,.5,-.3,1330000))
    assert truth==d
    np.testing.assert_allclose(other-changed,b-s,rtol=1e-12,atol=1e-15)
    assert np.linalg.norm(changed-s)>0
