from dedalus.utils.logger import mylog
from dedalus.utils.parallelism import com_sys
def mcwilliams_spec(k, k0, E0):
    """spectrum from McWilliams (1990), JFM 219:361-385

    UNNORMALIZED!

    """
    spec = k**6./(k + 2.*k0)**18.

    return spec
