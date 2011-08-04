"""Representation object for fourier transformable fields

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
import numpy as na
import numpy.fft as fpack
import fftw3

try: 
    from mpi4py import MPI
except:
    print "mpi4py not found. Parallelization will not work."

class Representation(object):
    """a representation of a field. it stores data and provides
    spatial derivatives.

    """

    def __init__(self, sd, shape, length):
        pass

class FourierRepresentation(Representation):
    """Container for data that can be Fourier transformed. Includes a
    wrapped and specifiable method for performing the FFT. 

    Parallelization will go here?

    """

    def __init__(self, sd, shape, length, dtype='complex128', method='fftw',
                 dealiasing='2/3'):
        """
        Inputs:
            sd          state data object
            shape       The shape of the data, tuple of ints
            length      The length of the data, tuple of floats (default: 2 pi)
            
        """
        self.sd = sd
        self.shape = shape
        self.length = length
        self.ndim = len(self.shape)
        self.data = na.zeros(self.shape, dtype=dtype)
        self.__eps = na.finfo(dtype).eps
        self._curr_space = 'kspace'
        self.trans = {'x': 0, 'y': 1, 'z': 2,
                      0:'x', 1:'y', 2:'z'} # for vector fields
        
        self._setup_k()
        self.set_fft(method)
        self.set_dealiasing(dealiasing)

    def __getitem__(self,space):
        """returns data in either xspace or kspace, transforming as necessary.

        """
        if space == self._curr_space:
            pass
        elif space == 'xspace':
            self.backward()
        elif space == 'kspace':
            self.forward()
        else:
            raise KeyError("space must be either xspace or kspace")
        
        return self.data

    def __setitem__(self, space, data):
        """this needs to ensure the pointer for the field's data
        member doesn't change for FFTW. Currently, we do that by
        slicing the entire data array. 
        """
        if data.size < self.data.size:
            sli = [slice(i/4+1,i/4+i+1) for i in data.shape]
            self.data[sli] = data
        else:
            sli = [slice(i) for i in self.data.shape]
            self.data[:] = data[sli]

        self._curr_space = space

    def _setup_k(self):
        # Get Nyquist wavenumbers
        self.kny = na.pi * na.array(self.shape) / na.array(self.length)

        # Setup wavenumbers
        self.k = []
        for i,S in enumerate(self.shape):
            kshape = i * (1,) + (S,) + (self.ndim - i - 1) * (1,)
            ki = fpack.fftfreq(S) * 2. * self.kny[i]
            ki.resize(kshape)
            self.k.append(ki)
        self.k = dict(zip(['z','y','x'][3-self.ndim:], self.k))

    def set_fft(self, method):
        if method == 'fftw':
            self.fplan = fftw3.Plan(self.data, direction='forward', flags=['measure'])
            self.rplan = fftw3.Plan(self.data, direction='backward', flags=['measure'])
            self.fft = self.fwd_fftw
            self.ifft = self.rev_fftw
        if method == 'numpy':
            self.fft = self.fwd_np
            self.ifft = self.rev_np
            
    def set_dealiasing(self, dealiasing):
        if dealiasing == '2/3':
            self.dealias = self.dealias_23
        elif dealiasing == '2/3spherical':
            self.dealias = self.dealias_spherical_23
        else:
            self.dealias = self.zero_nyquist

    def fwd_fftw(self):
        self.fplan()
        self.data /= self.data.size

    def rev_fftw(self):
        self.rplan()
        self.data.imag = 0.

    def fwd_np(self):
        self.data = fpack.fftn(self.data)

    def rev_np(self):
        self.data = fpack.ifftn(self.data)
        self.data.imag = 0

    def forward(self):
        """FFT method to go from xspace to kspace."""
        
        self.fft()
        self._curr_space = 'kspace'
        self.dealias()
        #self.zero_under_eps()

    def backward(self):
        """IFFT method to go from kspace to xspace."""
        
        self.dealias()
        self.ifft()
        self._curr_space = 'xspace'

    def dealias_23(self):
        """Orszag 2/3 dealias rule"""
        
        # Zeroing mask   
        dmask = ((na.abs(self.k['x']) >= 2/3. * self.kny[-1]) | 
                 (na.abs(self.k['y']) >= 2/3. * self.kny[-2]))

        if self.ndim == 3:
            dmask = dmask | (na.abs(self.k['z']) >= 2/3. * self.kny[-3])

        self['kspace'] # Dummy call to switch spaces
        self.data[dmask] = 0.
        
    def dealias_spherical_23(self):
        """Spherical 2/3 dealiasing rule."""
        
        # Zeroing mask   
        dmask = na.sqrt(self.k2()) >= 2/3. * na.min(self.kny)

        self['kspace'] # Dummy call to switch spaces
        self.data[dmask] = 0.
        
    def deriv(self,dim):
        """take a derivative along dim"""
        if self._curr_space == 'xspace':
            self.forward()
        der = self.data * 1j*self.k[dim]
        return der

    def k2(self, no_zero=False):
        """returns k**2. if no_zero is set, will set the mean mode to
        1. useful for division when the mean is not important.
        """
        k2 = na.zeros(self.data.shape)
        for k in self.k.values():
            k2 += k**2
        if no_zero:
            k2[k2 == 0] = 1.
        return k2
        
    def zero_nyquist(self):
        """Zero out the Nyquist space in each dimension."""
        
        self['kspace']  # Dummy call to ensure in kspace
        nyspace = [slice(None)] * self.ndim 
        
        # Pick out Nyquist space for each dimension and set to zero
        for i in xrange(self.ndim):
            nyspace[i] = self.shape[i] / 2
            self.data[nyspace] = 0.
            nyspace[i] = slice(None)

    def zero_under_eps(self):
        """Zero out any modes with coefficients smaller than machine epsilon."""
        
        self['kspace']  # Dummy call to ensure in kspace
        self.data[na.abs(self.data) < self.__eps] = 0.

    def save(self, dataset):
        """save data to HDF5 dataset

        inputs 
        ------
        dataset -- an h5py dataset opbject

        """
        dataset[:] = self.data
        dataset.attrs['space'] = self._curr_space

class FourierShearRepresentation(FourierRepresentation):
    """

    """
    def __init__(self, sd, shape, length, **kwargs):    
        """
        Shearing box implementation.

        Parameters
        ----------
        S : float
            the *dimensional* shear rate in 1/t units

        """
        
        FourierRepresentation.__init__(self, sd, shape, length, **kwargs)
        self.shear_rate = self.sd.parameters['S'] * self.sd.parameters['Omega']
        self.kx = self.k['x'].copy()
        
        # For now, only numpy's fft is supported
        self.fft = fpack.fft
        self.ifft = fpack.ifft


    def backward(self):
        """IFFT method to go from kspace to xspace."""
        
        deltay = self.shear_rate * self.sd.time 
        x = na.linspace(0., self.length[-1], self.shape[-1], endpoint=False)
        
        self.dealias()
        
        # Do x fft
        self.data = self.ifft(self.data, axis=-1)
        
        # Phase shift
        self.data *= na.exp(-1j * self.k['y'] * x * deltay)
        
        # Do y fft
        self.data = self.ifft(self.data, axis=-2)
        
        # Do z fft
        if self.ndim == 3:
            self.data = self.ifft(self.data, axis=0)
        
        self._curr_space = 'xspace'

    def forward(self):
        """FFT method to go from xspace to kspace."""
        
        deltay = self.shear_rate * self.sd.time
        x = (na.linspace(0., self.length[-1], self.shape[-1], endpoint=False) +
             na.zeros(self.shape))

        # Do z fft
        if self.ndim == 3:
            self.data = self.fft(self.data, axis=0)

        # Do y fft
        self.data = self.fft(self.data, axis=-2)

        # Phase shift
        self.data *= na.exp(1j * self.k['y'] * x * deltay)
        
        # Do x fft
        self.data = self.fft(self.data, axis=-1)
        
        self._curr_space = 'kspace'
        self.dealias()
        
    def _update_k(self):
        """Evolve modes due to shear."""

        self.k['x'] = self.kx - self.shear_rate * self.sd.time * self.k['y']
        while self.k['x'].min() < -self.kny[-1]:
            self.k['x'][self.k['x'] < -self.kny[-1]] += 2 * self.kny[-1]
        while self.k['x'].max() >= self.kny[-1]:
            self.k['x'][self.k['x'] >= self.kny[-1]] -= 2 * self.kny[-1]
            
class ParallelFourierRepresentation(FourierRepresentation):
    def __init__(self, sd, shape, length, comm=None, dtype='complex128', method='fftw',
                 dealiasing='2/3'):
        """
        Inputs:
            sd          state data object
            shape       The shape of the data, tuple of ints
            length      The length of the data, tuple of floats (default: 2 pi)
            comm        MPI communicator (defaults to COMM_WORLD)
        """
        if comm == None:
            comm = MPI.COMM_WORLD
        self.comm = comm

        self.offset = na.array([0, 0, comm.Get_rank() * shape[2]])
        FourierRepresentation.__init__(self, sd, shape, length, dtype=dtype, method=method, dealiasing=dealiasing)

    def _setup_k(self):
        nproc = self.comm.Get_size()
        myproc = self.comm.Get_rank()

        global_shape = na.array(self.shape)*na.array([nproc, 1, 1])
        global_length = na.array(self.length)*na.array([nproc, 1, 1])
        # Get Nyquist wavenumbers
        self.kny = na.pi * na.array(self.shape) / na.array(self.length)

        # Setup wavenumbers
        self.k = []
        for i,S in enumerate(global_shape):
            kshape = i * (1,) + (S,) + (self.ndim - i - 1) * (1,)
            ki = fpack.fftfreq(S) * 2. * self.kny[i]
            ki.resize(kshape)
            self.k.append(ki)
        self.k = dict(zip(['z','y','x'][3-self.ndim:], self.k))
        self.k['z'] = self.k['z'][myproc*self.shape[0]:(myproc+1)*self.shape[0]]


class SphericalHarmonicRepresentation(FourierRepresentation):
    """Dedalus should eventually support spherical and cylindrical geometries.

    """
    pass
