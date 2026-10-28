import numpy as np
from experiments.diagnose_cpu_path_boundary import objective, describe
from gnss_gpu.multipath_information_cpu import ca_passband_spectrum, CHIP_RATE


def test_loss_gradient_and_noiseless_truth():
    f,s=ca_passband_spectrum(3.6*CHIP_RATE)
    y=s*(np.exp(-2j*np.pi*f*.12/CHIP_RATE)+.7j*np.exp(-2j*np.pi*f*.22/CHIP_RATE))
    solve=objective(y,np.ones(len(y)))
    assert solve((.12,.1))[0]<1e-24
    point=np.array([.11,.08]);step=1e-6
    numerical=[(solve(point+step*np.eye(2)[i])[0]-solve(point-step*np.eye(2)[i])[0])/(2*step) for i in range(2)]
    np.testing.assert_allclose(solve(point)[1],numerical,rtol=1e-7,atol=1e-7)
    assert describe(solve,(.12,.001))['template_condition']>describe(solve,(.12,.1))['template_condition']
