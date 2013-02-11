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
from dedalus.utils.parallelism import com_sys, pickle
from dedalus.utils.logger import mylog
from dedalus.funcs import insert_ipython
import numpy as na
from dedalus.utils.misc_numeric import find_zero, integrate_quad, interp_linear
from dedalus.analysis.volume_average import volume_average
def taylor_green(data):
    """The famous Taylor-Green vortex in 2 or 3D.

    Inputs:
        data: StateData object

    """
    mylog.info("Initializing Taylor Green Vortex.")
    ndim = data['u'].ndim
    if ndim == 2:
        data['u']['x'].data[1,1] = -1j/4.
        data['u']['x'].data[-1,-1] = 1j/4.
        data['u']['x'].data[-1,1] = -1j/4.
        data['u']['x'].data[1,-1] = 1j/4.

        data['u']['y'].data[1,1] = 1j/4.
        data['u']['y'].data[-1,-1] = -1j/4.
        data['u']['y'].data[-1,1] = -1j/4.
        data['u']['y'].data[1,-1] = 1j/4.
    elif ndim == 3:
        if data['u']['x'].find_mode((1,1,1)):
            data['u']['x']['kspace'][1,1,1] = -1j/8.
        if data['u']['x'].find_mode((1,1,-1)):
            data['u']['x']['kspace'][1,1,-1] = 1j/8.
        if data['u']['x'].find_mode((-1,1,1)):
            data['u']['x']['kspace'][-1,1,1] = -1j/8.
        if data['u']['x'].find_mode((-1,1,-1)):
            data['u']['x']['kspace'][-1,1,-1] = 1j/8.
        if data['u']['x'].find_mode((1,-1,1)):
            data['u']['x']['kspace'][1,-1,1] = -1j/8.
        if data['u']['x'].find_mode((1,-1,-1)):
            data['u']['x']['kspace'][1,-1,-1] = 1j/8.
        if data['u']['x'].find_mode((-1,-1,1)):
            data['u']['x']['kspace'][-1,-1,1] = -1j/8.
        if data['u']['x'].find_mode((-1,-1,-1)):
            data['u']['x']['kspace'][-1,-1,-1] = 1j/8.

        if data['u']['y'].find_mode((1,1,1)):
            data['u']['y']['kspace'][1,1,1] = 1j/8.
        if data['u']['y'].find_mode((1,1,-1)):
            data['u']['y']['kspace'][1,1,-1] = 1j/8.
        if data['u']['y'].find_mode((-1,1,1)):
            data['u']['y']['kspace'][-1,1,1] = -1j/8.
        if data['u']['y'].find_mode((-1,1,-1)):
            data['u']['y']['kspace'][-1,1,-1] = -1j/8.
        if data['u']['y'].find_mode((1,-1,1)):
            data['u']['y']['kspace'][1,-1,1] = 1j/8.
        if data['u']['y'].find_mode((1,-1,-1)):
            data['u']['y']['kspace'][1,-1,-1] = 1j/8.
        if data['u']['y'].find_mode((-1,-1,1)):
            data['u']['y']['kspace'][-1,-1,1] = -1j/8.
        if data['u']['y'].find_mode((-1,-1,-1)):
            data['u']['y']['kspace'][-1,-1,-1] = -1j/8.

def kida_vortex(data, a, chi=None, smooth=False):
    """Generate ICs for the Kida steady-state vortex.

    Inputs:
        data: StateData object
        a: semi-minor axis of ellipse
        chi: (default None) aspect ratio. if None, will use the aspect ratio of the x-y plane to set the vortex

    """
    try:
        Omega = data.parameters['Omega']
    except KeyError:
        raise KeyError("Kida vortex requires shearing box!")

    if chi is None:
        if hasattr(data, 'local_shape'):
            x = data.local_shape['kspace'][0]
            y = data.local_shape['kspace'][1]
        else:
            x = data.shape[1]
            y = data.shape[0]
        chi = float(y/x)

    b = chi * a
    vort_ampl = data.parameters['S'] * (1 + chi) * Omega/(chi * (chi - 1.))
    mylog.info("adding vortex with vorticity = %10.5f" % vort_ampl)
    sh = data['u']['x']['kspace'].shape
    le = data['u']['x'].length
    aux = data.clone()
    aux.add_field('w','ScalarField')
    aux.add_field('psi','ScalarField')


    xx, yy = na.meshgrid(na.r_[0:sh[1]]*le[1]/sh[1],na.r_[0:sh[0]]*le[0]/sh[0])
    ff = (xx - le[1]/2.)**2/a**2 + (yy - le[0]/2.)**2/b**2 - 1

    if smooth:
        # use tanh to smooth vortex edge...
        aux['w']['xspace'] = vort_ampl*(1-na.tanh(ff/0.3))/2.
    else:
        # or not...
        v = na.zeros_like(ff)
        v[ff < 0] = vort_ampl
        aux['w']['xspace'] = v

    aux['psi']['kspace'] = aux['w']['kspace']/aux['w'].k2(no_zero=True)

    data['u']['x']['kspace'] = aux['psi'].deriv('y')
    data['u']['y']['kspace'] = -aux['psi'].deriv('x')

def sin_k(f, kindex, ampl=1.):
    f[tuple(kindex)] = ampl*1j/2.
    f[tuple(-1*na.array(kindex))] = f[tuple(kindex)].conjugate()

def cos_k(f, kindex, ampl=1.):
    f[tuple(kindex)] = ampl/2.
    f[tuple(-1*na.array(kindex))] = f[tuple(kindex)].conjugate()


def alfven(data, k=(1, 0, 0), B0mag=5.0, u1mag=5e-6, p_vec = [0., 1., 0.]):
    """
    Generate conditions for simulating Alfven waves in MHD.
    For 2d, must have k and B0 in same direction.

    Inputs:
        data        StateData object
        k           k-index tuple, (kz, ky, kx), adds sine wave here in u1 and B1
        B0mag       B0 magnitude along x-direction
        u1mag       u1 magnitude for k-wave being added
        p_vec       polarization vector. USER is responsible for making sure this is perp to x
    """


    N = len(k)
    if N != 3:
        raise ValueError('Only setup for 3d')
    kmag = na.linalg.norm(k)

    # Field setup and calculation
    B0 = na.array([1., 0., 0.])[:N] * B0mag

    p_vec = na.array(p_vec)

    # Background magnetic field
    if (0 in data['u']['x'].k['z'] and
        0 in data['u']['x'].k['y'] and
        0 in data['u']['x'].k['x']):
        for i in xrange(data['B'].ndim):
            k0 = (0,) * N
            data['B'][i]['kspace'][k0] = B0[i]

    # Assign modes for k and -k
    if (k[2] in data['u']['x'].k['z'] and
        k[1] in data['u']['x'].k['y'] and
        k[0] in data['u']['x'].k['x']):
        kiz = na.array(na.where(data['u']['x'].k['z'] == k[2]))
        kiy = na.array(na.where(data['u']['x'].k['y'] == k[1]))
        kix = na.array(na.where(data['u']['x'].k['x'] == k[0]))

        kindex = tuple(kiz + kiy + kix)
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
        u1 = p_vec[3-N:] * u1mag
        B1 = (na.dot(k, u1) * B0 - na.dot(k, B0) * u1) / omega
        B1mag = na.linalg.norm(B1)
        for i in xrange(data['u'].ndim):
            data['u'][i]['kspace']
            data['B'][i]['kspace']

            data['u'][i].data[kindex] = u1[i] * 1j / 2.
            data['B'][i].data[kindex] = B1[i] * 1j / 2.

    if (-k[2] in data['u']['x'].k['z'] and
         -k[1] in data['u']['x'].k['y'] and
         -k[0] in data['u']['x'].k['x']):
        # Find correct mode
        kiz = na.array(na.where(data['u']['x'].k['z'] == -k[2]))
        kiy = na.array(na.where(data['u']['x'].k['y'] == -k[1]))
        kix = na.array(na.where(data['u']['x'].k['x'] == -k[0]))

        kindex = tuple(kiz + kiy + kix)
        # Alfven speed and wave frequency
        cA = B0mag / na.sqrt(4 * na.pi * data.parameters['rho0'])
        omega = na.abs(cA * na.dot(k, B0) / B0mag)

        # u and B perturbations
        u1 = p_vec[3-N:] * u1mag

        B1 = (na.dot(k, u1) * B0 - na.dot(k, B0) * u1) / omega
        B1mag = na.linalg.norm(B1)

        for i in xrange(data['u'].ndim):
            data['u'][i]['kspace']
            data['B'][i]['kspace']

            data['u'][i].data[kindex] = -u1[i] * 1j / 2.
            data['B'][i].data[kindex] = -B1[i] * 1j / 2.


def alfven_old(data, k=(1, 0, 0), B0mag=5.0, u1mag=5e-6):
    """
    Generate conditions for simulating Alfven waves in MHD.
    For 2d, must have k and B0 in same direction.

    Inputs:
        data        StateData object
        k           k-index tuple, (kx, ky, kz), adds sine wave here in u1 and B1
        B0mag       B0 magnitude along x-direction
        u1mag       u1 magnitude for k-wave being added
    """

    N = len(k)
    kmag = na.linalg.norm(k)

    # Field setup and calculation
    B0 = na.array([1., 0., 0.])[:N] * B0mag

    # Alfven speed and wave frequency
    cA = B0mag / na.sqrt(4 * na.pi * data.parameters['rho0'])
    omega = na.abs(cA * na.dot(k, B0) / B0mag)
    print '-' * 20
    print 'cA = ', cA
    print 'cA * cos(theta) = ', cA * na.dot(k, B0) / B0mag / kmag
    print 'Max dt (CFL=1) = ', na.min(na.array(data['u']['x'].length) /
                                      na.array(data['u']['x'].shape)) / cA
    print '-' * 20

    # Background magnetic field
    for i in xrange(data['B'].ndim):
        k0 = (0,) * N
        data['B'][i]['kspace'][k0] = B0[i]

    # u and B perturbations
    u1 = na.array([0., 1., 0.])[3-N:] * u1mag

    B1 = (na.dot(k, u1) * B0 - na.dot(k, B0) * u1) / omega
    B1mag = na.linalg.norm(B1)

    for i in xrange(data['u'].ndim):
        data['u'][i]['kspace']
        data['B'][i]['kspace']

        sin_k(data['u'][i].data, k[::-1], ampl=u1[i])
        sin_k(data['B'][i].data, k[::-1], ampl=B1[i])

def turb_new(data, spec, tot_en=0.5, **kwargs):
    """generate noise with a random phase and a spectrum given by
    the spec function.

    Uses the Rogallo (1981) algorithm.

    MODERINZED, PARALLEL, MULTI-D VERSION

    """
    kk = na.sqrt(data['u'][0].k2())
    kx = data['u'][0].k['x']
    ky = data['u'][0].k['y']

    aux = data.clone()
    aux.add_field('ampl','ScalarField')
    sp = spec(kk, **kwargs)
    kk[kk==0] = 1.
    if data.ndim == 2:
        aux['ampl']['kspace'] = sp/(2.*na.pi*kk)
    elif data.ndim == 3:
        aux['ampl']['kspace'] = sp/(4*na.pi*kk**2)

    aux['ampl'].dealias()
    norm = volume_average(aux['ampl']['kspace'],kdict=data['u'][0].k,reduce_all=True)
    aux['ampl']['kspace'] *= tot_en/norm
    aux['ampl']['kspace'] = na.sqrt(2.*aux['ampl']['kspace'])

    eps = na.finfo(data['u']['x']['kspace'].dtype).eps

    data['u']['x']['xspace'] = na.random.random(data['u'][0]['xspace'].shape)
    theta1 = data['u']['x']['kspace'].copy()
    theta1 /= na.abs(theta1 + eps)

    data['u']['x']['xspace'] = na.random.random(data['u'][0]['xspace'].shape)
    theta2 = data['u']['x']['kspace'].copy()
    theta2 /= na.abs(theta2 + eps)

    data['u']['x']['xspace'] = na.random.random(data['u'][0]['xspace'].shape)
    phi    = data['u']['x']['kspace'].copy()
    phi    /= na.abs(phi + eps)
    phi = na.arctan2(phi.imag, phi.real)
    alpha = aux['ampl']['kspace'] * theta1
    if data.ndim == 2:
        data['u']['x']['kspace'] = alpha * ky/kk
        data['u']['y']['kspace'] = -alpha * kx/kk
        if com_sys.myproc == 0:
            # force hermitian symmetry and zero the nyquist mode.
            kshape = data['u']['x']['kspace'].shape[1]
            kylim = kshape/2 + 1
            data['u']['x']['kspace'][0,:] = alpha[0,:] * na.abs(ky[0,:])/kk[0,:]
            data['u']['x']['kspace'][0,kylim] = 0.
    elif data.ndim == 3:
        kz = data['u'][0].k['z']
        k2 = na.sqrt(kx**2 + ky**2)
        k2[k2 == 0] = 1.
        alpha *= na.cos(phi)
        beta = aux['ampl']['kspace'] * theta2 * na.sin(phi)

        data['u']['x']['kspace'] = (alpha * kk * ky + beta * kx * kz)/(kk * k2)
        data['u']['y']['kspace'] = (beta * ky * kz - alpha * kk*kx)/(kk * k2)
        data['u']['z']['kspace'] = -(beta * k2)/kk

def turb(ux, uy, spec, tot_en=0.5, **kwargs):
    """generate noise with a random phase and a spectrum given by
    the spec function.

    """
    kk = na.zeros(ux.data.shape)
    print ux.local_shape
    for i,k in ux.k.iteritems():
        print k.shape
        kk += k**2

    kk = na.sqrt(kk)
    k2 = na.sqrt(ux.k['x']**2 + ux.k['y']**2)
    k2[k2==0] = 1.
    sp = spec(kk,**kwargs)

    kk[kk==0] = 1.
    ampl = na.sqrt(sp/(2*na.pi*kk)) # for 2D

    # Rogallo alorithm for random-phase, incompressible motions
    alpha = ampl * na.exp(1j*2*na.pi*na.random.random(ux.data.shape)) #* na.cos(2*na.pi*na.random.random(ux.data.shape))
    ux.data[:,:] = alpha*kk*ux.k['y']/(kk*k2)
    uy.data[:,:] = -alpha*kk*ux.k['x']/(kk*k2)

    # fix hermitian
    nh = ux.data.shape[1]/2 + 1
    if ux.data.shape[1] % 2 == 0:
        start = nh - 2
        ux.data[0,nh-1] = 0.
    else:
        start = nh - 1
    ux.data[0,nh:] = ux.data[0,start:0:-1].conj()
    uy.data[0,nh:] = uy.data[0,start:0:-1].conj()

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
    y, x = data['u']['x'].xspace_grid()
    aux = data.clone()
    aux.add_field('w','ScalarField')
    aux.add_field('psi','ScalarField')
    aux['w']['xspace']=na.exp(-((x-na.pi)**2+(y-na.pi+na.pi/4)**2)/(0.2)) \
        +na.exp(-((x-na.pi)**2+(y-na.pi-na.pi/4)**2)/(0.2)) \
        -0.5*na.exp(-((x-na.pi-na.pi/4)**2+(y-na.pi-na.pi/4)**2)/(0.4))

    aux['psi']['kspace'] = aux['w']['kspace']/aux['w'].k2(no_zero=True)

    data['u']['x']['kspace'] = aux['psi'].deriv('y')
    data['u']['y']['kspace'] = -aux['psi'].deriv('x')

def vorticity_wave(data, mode, w_amp):
    """
    Initialize vorticity wave

    Parameters
    ----------
    data : StateData object
        Dataset
    mode : tuple of floats
        Wavevector
    w_amp : complex
        Complex z-vorticity amplitude for specified mode

    """

    aux = data.clone()
    aux.add_field('w','ScalarField')
    aux.add_field('psi','ScalarField')

    index0 = data['u']['x'].find_mode(mode)
    if index0:
        aux['w']['kspace'][index0] = w_amp / 2.
    index1 = data['u']['x'].find_mode(-1 * na.array(mode))
    if index1:
        aux['w']['kspace'][index1] = na.conj(w_amp.conj) / 2.
    if index0 or index1:
        aux['psi']['kspace'] = aux['w']['kspace'] / aux['w'].k2(no_zero=True)
        data['u']['x']['kspace'] = aux['psi'].deriv('y')
        data['u']['y']['kspace'] = -aux['psi'].deriv('x')

def add_gaussian_white_noise(comp, std):
    """
    Add Gaussian white noise to specified component.  Noise is constructed by
    adding a phasor of specified amplitude (determined by desired x-space
    standard deviation) and random phase to each mode.  When dealiasing is used,
    true white noise is not possible: the signal can only be uncorrelated down
    to the dealiasing scale, instead of all the way to the Nyquist scale.

    Parameters
    ----------
    comp : Representation object
        Component to be augmented by noise.
    std : float
        x-space standard deviation.

    """

    phase = 2 * na.pi * na.random.random(comp.kdata.shape)
    amp = std / na.sqrt(comp.nmodes - 1)
    noise = amp * na.exp(1j * phase)
    zero_index = comp.find_mode([0.,] * comp.ndim)
    if zero_index:
        noise[zero_index] = 0.
    comp['kspace'] += noise
    comp.enforce_hermitian()


