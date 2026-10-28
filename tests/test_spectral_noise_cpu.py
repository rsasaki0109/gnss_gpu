import numpy as np
import pytest
from gnss_gpu.spectral_noise_cpu import spectral_noise_shape


@pytest.mark.parametrize('profile',['white','center','edge'])
def test_equal_expected_energy_symmetry_and_measured_gaussian_variance(profile):
    f=np.arange(-3682,3683,dtype=float)*1000
    shape=spectral_noise_shape(f,profile)
    assert np.all(shape>0)
    assert shape.mean()==pytest.approx(1.)
    np.testing.assert_allclose(shape,shape[::-1],rtol=5*np.finfo(float).eps,atol=0)
    rng=np.random.default_rng(728)
    noise=(rng.normal(size=(32,len(f)))+1j*rng.normal(size=(32,len(f))))*np.sqrt(shape/2)
    assert np.mean(abs(noise)**2)==pytest.approx(1.,rel=.015)
    if profile=='white': np.testing.assert_array_equal(shape,np.ones(len(f)))
    if profile=='center': assert shape[len(f)//2]>5*shape[0]
    if profile=='edge': assert shape[0]>5*shape[len(f)//2]


@pytest.mark.parametrize('frequency,profile',[([0,0],'white'),([0,np.nan],'white'),([1],'white'),([[1,2]],'white'),([0,1],'unknown')])
def test_invalid_profiles(frequency,profile):
    with pytest.raises(ValueError): spectral_noise_shape(frequency,profile)
