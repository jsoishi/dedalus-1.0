from dedalus.utils.logger import mylog
from dedalus.utils.parallelism import com_sys
def mcwilliams_spec(k, k0, E0):
    """spectrum from McWilliams (1990), JFM 219:361-385

    """
    spec = k**6./(k + 2.*k0)**18.
    # normalize by 1D spectrum
    if com_sys.myproc == 0:
        sl = (slice(0,1),slice(0,k.shape[1]/2-1))
        norm = E0/spec[sl].sum()
    else:
        norm = 0.
    com_sys.comm.bcast(norm, root=0)
    return spec*norm
