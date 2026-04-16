"""Back-compat shim. Canonical location is `gnss_gpu.e2e_helpers`."""
from gnss_gpu.e2e_helpers import *  # noqa: F401,F403
from gnss_gpu.e2e_helpers import (  # noqa: F401
    C_LIGHT,
    CA_CHIP_RATE,
    GPS_CA_CODE_LENGTH,
    GPS_CA_PERIOD_S,
    GPS_CA_PERIOD_M,
    GPS_L1_FREQ,
    DEFAULT_CODE_LOCK_MAX_ERROR_M,
    compute_e2e_wls_weights,
    acquisition_lag_to_code_phase_chips,
    code_phase_chips_to_acquisition_lag,
    refine_acquisition_code_lag_dll,
    refine_acquisition_code_lags_dll_batch,
    pseudorange_to_code_phase_chips,
    acquisition_code_phase_to_pseudorange,
)
