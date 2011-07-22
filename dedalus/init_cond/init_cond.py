"""Initial Condition Generators.

NB: UNTIL we switch fftw backends, *all* data initializations must be
done using data[:,:] or some other explicit indexing. otherwise, the
pointer will be erased, and FFTs will fail!

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi.  All Rights Reserved.

  This file is part of dedalus.

  dedalus is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from dedalus.funcs import insert_ipython
import numpy as na
try:
    from scipy.interpolate import interp1d
    from scipy.integrate import simps
    from scipy.optimize import broyden1
except ImportError:
    print "Warning: Scipy not found. Interpolation won't work."
from dedalus.data_objects import hermitianize

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
    f.data[-1,0] = -f.data[1,0]

def sin_k(f, kindex, ampl=1.):
    f[tuple(kindex)] = ampl*1j/2.
    f[tuple(-1*na.array(kindex))] = f[tuple(kindex)].conjugate()

def cos_k(f, kindex, ampl=1.):
    f[tuple(kindex)] = ampl/2.
    f[tuple(-1*na.array(kindex))] = f[tuple(kindex)].conjugate()


def alfven(data, k=(1, 0, 0)):
    """
    Generate conditions for simulating Alfven waves in MHD.
    For 2d, must have k and B0 in same direction
    """
    
    N = len(k)
    
    # Field setup and calculation
    B0 = na.array([1., 0., 0.])[:N] * 5.0
    B0mag = na.linalg.norm(B0)
    
    kmag = na.linalg.norm(k)
        
    # Alfven speed and wave frequency
    cA = B0mag / na.sqrt(4 * na.pi * data.parameters['rho0'])
    omega = na.abs(cA * na.dot(k, B0) / B0mag)
    print '-' * 20
    print 'cA = ', cA
    print 'cA * cos(theta) = ', cA * na.dot(k, B0) / B0mag / kmag
    print 'Max dt (CFL=1) = ', na.min(na.array(data['u']['x'].length) / 
                                      na.array(data['u']['x'].shape)) / cA
    print '-' * 20
    
    # u and B perturbations
    u1 = na.array([0., 0., 1.])[3-N:] * 5e-6
    u1mag = na.linalg.norm(u1)
    
    B1 = (na.dot(k, u1) * B0 - na.dot(k, B0) * u1) / omega
    B1mag = na.linalg.norm(B1)
    
    for i in xrange(data['u'].ndim):
        sin_k(data['u'][i].data, k[::-1], ampl=u1[i])
        sin_k(data['B'][i].data, k[::-1], ampl=B1[i])
    
        data['u'][i]._curr_space = 'kspace'
        data['B'][i]._curr_space = 'kspace'
    
    # Background magnetic field
    for i in xrange(data['B'].ndim):
        data['B'][i]['xspace'] += B0[i]

def turb(ux, uy, spec, tot_en=0.5, **kwargs):
    """generate noise with a random phase and a spectrum given by
    the spec function.

    """
    kk = na.zeros(ux.data.shape)
    for k in ux.k.values():
        kk += k**2
    kk = na.sqrt(kk)
    sp = spec(kk,**kwargs)
    ampl = na.sqrt(sp/ux.data.ndim/sp.sum())

    for u in [ux,uy]:
        u.data[:,:] = ampl*na.exp(1j*2*na.pi*na.random.random(u.data.shape))

        # enforce symmetry in kspace to ensure data is real in
        # xspace.
        nx = u.data.shape[0]
        u.data[-1:nx/2:-1,-1:nx/2:-1] = u.data[1:nx/2,1:nx/2].conj()
        u.data[nx/2-1:0:-1, -1:nx/2:-1] = u.data[nx/2+1:,1:nx/2].conj()

        u.data[0,nx/2+1:] = u.data[0,nx/2-1:0:-1].conj()
        u.data[nx/2+1:,0] = u.data[nx/2-1:0:-1,0].conj()
        u.data[nx/2,nx/2+1:] = u.data[nx/2,nx/2-1:0:-1].conj()
        u.data[nx/2+1:,nx/2] = u.data[nx/2-1:0:-1,nx/2].conj()

        u.data[0:1,0:1].imag= 0.
        u.data[nx/2:nx/2+1,nx/2:nx/2+1].imag = 0
        u.data[0:1,nx/2:nx/2+1].imag = 0
        u.data[nx/2:nx/2+1,0:1].imag = 0
        u.data[0,0] = 0.

    remove_compressible(ux,uy)


def remove_compressible(ux,uy, renorm=False):
    """project off compressible parts of velocity fields. if renorm is
    set, will renormalize so ux and uy have the same normalization as
    before (not implemented yet.
    """
    if renorm:
        raise NotImplementedError
    ku = na.zeros_like(ux.data)
    for k, f in zip(('x','y'),(ux,uy)):
        ku += f.data * f.k[k]

    ux.data[:,:] -= ku * ux.k['x']/ux.k2(no_zero=True)
    uy.data[:,:] -= ku * uy.k['y']/uy.k2(no_zero=True)

def MIT_vortices(data):
    """define the 2D vortices from the MIT 18.336 sample matlab script at 

    http://math.mit.edu/cse/codes/mit18336_spectral_ns2d.m

    """
    
    sh = data['u']['x']['kspace'].shape
    x, y = na.meshgrid(na.r_[0:sh[0]]*2*na.pi/sh[0],na.r_[0:sh[1]]*2*na.pi/sh[1])
    aux = data.clone()
    aux.add_field('w','scalar')
    aux.add_field('psi','scalar')
    aux['w']['xspace']=na.exp(-((x-na.pi)**2+(y-na.pi+na.pi/4)**2)/(0.2)) \
        +na.exp(-((x-na.pi)**2+(y-na.pi-na.pi/4)**2)/(0.2)) \
        -0.5*na.exp(-((x-na.pi-na.pi/4)**2+(y-na.pi-na.pi/4)**2)/(0.4))

    aux['psi']['kspace'] = aux['w']['kspace']/aux['w'].k2(no_zero=True)

    data['u']['x']['kspace'] = aux['psi'].deriv('y')
    data['u']['y']['kspace'] = -aux['psi'].deriv('x')

def shearing_wave(data, wampl, kinit):
    """Lithwick (2007) 2D shearing wave. 

    inputs
    ------
    data -- data object
    wampl -- z vorticity amplitude
    kinit -- initial wave vector in index space
    """
    aux = data.clone()
    aux.add_field('w','scalar')
    aux.add_field('psi','scalar')
    cos_k(aux['w']['kspace'],kinit,ampl=wampl)
    aux['psi']['kspace'] = aux['w']['kspace']/aux['w'].k2(no_zero=True)

    data['u']['x']['kspace'] = aux['psi'].deriv('y')
    data['u']['y']['kspace'] = -aux['psi'].deriv('x')

def zeldovich(data, ampl, a_ini, a_cross):
    """velocity wave IC, for testing nonlinear collisionless cosmology
    against the Zeldovich approximation
    """
    k = 2*na.pi/data.length[0]
    N = data['delta'].shape[0]
    D = 1.
    A = a_ini/(a_cross * k)
    x = na.array([i - N/2 for i in xrange(N)])*data.length[0]/N
    q = na.array(broyden1(lambda y: na.array(y) + D*A*na.sin(k*na.array(y)) - x, x))
    delta1d = 1./(1. + D*A*k*na.cos(k*q)) - 1.
    for i in xrange(N):
        for j in xrange(N):
            data['delta']['xspace'][i,j,:] = delta1d
    data['u'][0]['kspace'][0,0,1] = ampl * 1j / 2
    data['u'][0]['kspace'][0,0,-1] = -data['u'][0]['kspace'][0,0,1]

def get_ic_data(fname, ak, deltacp, thetac):
    """read certain values from a linger output file

    """

    infile = open(fname)
    for i,line in enumerate(infile):
        values = line.split()
        ak.append(float(values[1]))
        deltacp.append(float(values[6]))
        thetac.append(float(values[11]))
    infile.close()

def get_norm_data(fname, ak_trans, Ttot0):
    """read certain values from a transfer-mode linger++ output file

    """
    infile = open(fname)
    for i,line in enumerate(infile):
        values = line.split()
        ak_trans.append(float(values[0]))
        Ttot0.append(float(values[6]))
    infile.close()

def sig2_integrand(Ttot0, ak, nspect):
    """calculate the integrand in the expression for the mean square
    of the overdensity field, smoothed with a top-hat window function W
    at 8 Mpc.

    """
    R = 8.
    x = ak*R
    w = 3. * (na.sin(x) - x*na.cos(x))/(x*x*x)
    Pk = (ak**nspect) * (Ttot0*Ttot0)
    return (w*w) * Pk * (ak*ak)

def get_normalization(Ttot0, ak, sigma_8, nspect):
    """calculate the normalization for delta_tot using sigma_8

    """
    integrand = sig2_integrand(Ttot0, ak, nspect)
    sig2 = 4.*na.pi*simps(integrand, ak)
    ampl = sigma_8/na.sqrt(sig2)
    return ampl

def create_cosmo_field(field, spec, freq, mean=0., stdev=1.):
    """fill 3-d field with values sampled from gaussians with amplitudes 
    given by spec.

    """
    shape = spec.shape
    field[:,:,:] = spec * (na.random.normal(mean, stdev, shape) + 1j*na.random.normal(mean, stdev, shape)) # Use field[:,:,:] so we don't lose the pointer
    hermitianize.enforce_hermitian(field)
    hermitianize.zero_nyquist(field)

def cosmology(data, ic_fname, norm_fname, nspect=0.961, sigma_8=0.811):
    """generate realization of initial conditions in CDM overdensity
    and velocity from linger++ output. Assumes 3-dimensional fields.

    (create_cosmo_field actually creates the realizations from
    spectra; the rest is generating the spectra from input, which
    might reasonably belong somewhere else)

    ***** NEEDS TO CONSIDER RANGE OF WAVENUMBERS *****

    """

    ak = []
    deltacp = []
    thetac = []
    Ttot0 = []
    ak_trans = [] # the k-values that go with Ttot0

    get_ic_data(ic_fname, ak, deltacp, thetac)
    get_norm_data(norm_fname, ak_trans, Ttot0)
    ak = na.array(ak)
    deltacp = na.array(deltacp)
    thetac = na.array(thetac)
    ak_trans = na.array(ak_trans)
    Ttot0 = na.array(Ttot0)
    
    # normalize
    ampl = get_normalization(Ttot0, ak_trans, sigma_8, nspect)

    deltacp = deltacp*ampl
    thetac = thetac*ampl

    shape = data['delta'].shape
   
    nk = shape[0]
    freq = data['delta'].k['x']
   
    kk = na.sqrt(data['delta'].k2(no_zero=True))

    maxkk = kk[nk/2, nk/2, nk/2]
    maxkinput = max(ak)
    if maxkk > maxkinput:
        print 'cannot interpolate: some grid wavenumbers larger than input wavenumbers; ICs for those modes are wrong'
        # any |k| larger than the max input k is replaced by max input k
        kk[:,:,:] = na.minimum(kk, maxkinput*na.ones_like(kk))

    # calculate spectra
    f_deltacp = interp1d(ak[::-1], deltacp[::-1], kind='cubic')
    spec_delta = kk**(nspect/2.)*f_deltacp(kk)
    spec_delta[0,0,0] = 0

    f_thetac = interp1d(ak[::-1], thetac[::-1], kind='cubic')
    spec_theta = kk**(nspect/2. -1.)*f_thetac(kk)
    spec_theta[0,0,0] = 0

    # create realizations
    create_cosmo_field(data['delta']['kspace'], spec_delta, freq)
    for i in xrange(3):
        create_cosmo_field(data['u'][i]['kspace'], spec_theta, freq)
