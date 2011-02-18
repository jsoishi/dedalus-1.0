import numpy as N
try:
    import scipy.fftpack as fpack
except ImportError:
    import numpy.fft as fpack

from tg_kspace import tg_2d

class KspaceState(object):
    def __init__(self, dims, nmodes, dtype='complex128'):
        if len(nmods) != dims:
            raise ValueError("For a %i run, nmodes must be a tuple of length %i, but it is length %i"% (dims,dims,len(nmodes)))
        
        self.dtype = dype
        self._vars = {}
        if dims >= 1:
            self._vars('ux_hat') = N.complex(dims,dtype=self.dtype)
            self._vars('kx') = N.linspace(0,

    def __getitem__(self,key):
        return self._vars[key]

                            

Nx = 100
Ny = 100

vx_hat = N.zeros((Ny,Nx),dtype='complex128')
vy_hat = N.zeros((Ny,Nx),dtype='complex128')

vx_hat, vy_hat = tg_2d(vx_hat, vy_hat)

# d u_k / dt +  u.grad(u) = -grad(p) + nu nabla^2(u)

def calc_ugradu(vx_hat, vy_hat):
    ugradu_hat = N.zeros_like(vx_hat)
    
    # 1. get u
    vx = ifftn(vx_hat)
    vy = ifftn(vy_hat)
    
    # 2. calc 
    ugrad_u = ifftn(calc_dfdx(kx,


def calc_dfdx(k, f_hat):
    return 1j * k * f_hat
