import numpy as na
"""
NB: UNTIL we switch fftw backends, *all* data initializations must be
done using data[:,:] or some other explicit indexing. otherwise, the
pointer will be erased, and FFTs will fail!

"""
def taylor_green(ux, uy):
    if ux.dim == 2:
        ux.data[1,1] = -1j/4.
        ux.data[-1,-1] = 1j/4.
        ux.data[-1,1] = -1j/4.
        ux.data[1,-1] = 1j/4.

        uy.data[1,1] = 1j/4.
        uy.data[-1,-1] = -1j/4.
        uy.data[-1,1] = -1j/4.
        uy.data[1,-1] = 1j/4.

def sin_x(f,ampl=1.):
    f.data[0,1] = ampl*1j
    f.data[0,-1] = -f.data[0,1]

def sin_y(f,ampl=1.):
    f.data[1,0] = ampl*1j
    f.data[-1,0] = -f.data[0,1]

def turb(ux, uy, spec, **kwargs):
    """generate noise with a random phase and a spectrum given by
    the spec function.

    """
    kk = na.zeros(ux.data.shape)
    for k in ux.k.values():
        kk += k**2
    kk = na.sqrt(kk)
    ampl = na.sqrt(spec(kk,**kwargs)/ux.data.ndim)
    
    for u in [ux,uy]:
        ## create random numbers in real space, transform to
        ## complex. this ensures the field will be real, but screws up
        ## the power spectrum.
        #u.data[:,:] = na.random.random(u.data.shape) + 0j
        #u._curr_space='xspace'
        #u.data[:,:] = ampl*u['kspace']


        ## this is the propoer way to do it, but needs to be filtered
        ## to ensure it's real after ifft. currently, that doesn't
        ## work....
        # tantheta = na.tan(2*na.pi*na.random.random(u.data.shape))
        # c = na.sqrt(ampl**2/(1.+tantheta**2))
        # d = c*tantheta
        # u.data[:,:] = c + 1j*d

        u.data[:,:] = na.random.random(u.data.shape) +1j*na.random.random(u.data.shape)

        # enforce symmetry in kspace to ensure data is real in
        # xspace...doesn't work for 2D data yet

        nx = u.data.shape[0]
        u.data[nx/2+1:,:] = u.data[nx/2-1:0:-1,:].conj()
        u.data[nx/2,:].imag = 0.
        u.data[:,nx/2].imag = 0.
        u.data[0,:].imag = 0.
        u.data[:,0].imag = 0.

def mcwilliams_spec(k, k0, ampl):
    spec = k**6./(k + 2.*k0)**18.
    spec *= ampl/spec.sum()
    return spec
