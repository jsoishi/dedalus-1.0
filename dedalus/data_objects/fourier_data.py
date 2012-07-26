"""
Representation object for transformable fields.

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
import numpy.lib.stride_tricks as st
import numpy.fft as fpack
from dedalus.config import decfg
from dedalus.utils.parallelism import com_sys, swap_indices
from dedalus.utils.logger import mylog
from dedalus.utils.fftw import fftw
from dedalus.utils.timer import Timer
try:
    import pycuda.autoinit
    import pycuda.gpuarray as gpuarray
    import scikits.cuda.fft as cu_fft
except ImportError:
    mylog.warning("CUDA cannot be imported. Must use FFTW or numpy FFT.")

class Representation(object):
    """
    Representation of a field component. Stores data and provides spatial 
    derivatives.

    """

    def __init__(self, sd, shape, length):
        pass

class FourierRepresentation(Representation):
    """
    Representation of a field component that can be Fourier transformed 
    in all directions (i.e. periodic across the domain).
    
    """
    
    timer = Timer()
    
    def __init__(self, sd, shape, length):
        """
        Representation of a field component that can be Fourier transformed 
        in all directions (i.e. periodic across the domain).
        
        Parameters
        ----------
        sd : StateData object
        shape : tuple of ints
            The shape of the data in xspace: (z, y, x) or (y, x)
        length : tuple of floats
            The length of the data in xspace: (z, y, x) or (y, x)
            
        Notes
        -----
        When dealing with data, keep in mind that for parallelism the data 
        is transposed between k and x spaces following FFTW's internal MPI 
        parallelism, in which the first two indexed dimensions are swapped 
        when in k-space:

        In 3D:
            x-space: z, y, x
            k-space: y, z, x
    
        In 2D:
            x-space: y, x
            k-space: x, y
            
        """
        
        # Store inputs
        self.sd = sd
        self.global_shape = {'xspace': na.array(shape)}
        self.length = na.asfarray(length)
        self.ndim = len(shape)
        
        # Flexible datatypes not currently supported
        self.dtype = 'complex128'
        self.__eps = na.finfo(self.dtype).eps
        
        # Make sure all dimensions make sense
        if self.ndim not in (2, 3):
            raise ValueError("Must use either 2 or 3 dimensions.")
        if self.ndim != len(self.length):
            raise ValueError("Shape and Length must have same dimensions.")
        
        # Retrieve FFT method and dealiasing method from config
        method = decfg.get('FFT', 'method')
        dealiasing = decfg.get('FFT', 'dealiasing')
        
        # Translation tables
        if self.ndim == 2:
            self.xtrans = {'x':1, 1:'x', 'y':0, 0:'y'}
            self.ktrans = {'x':0, 0:'x', 'y':1, 1:'y'}
        else:
            self.xtrans = {'x':2, 2:'x', 'y':1, 1:'y', 'z':0, 0:'z'}
            self.ktrans = {'x':2, 2:'x', 'y':0, 0:'y', 'z':1, 1:'z'}
            
        # Complete setup
        self._allocate_memory(method)
        self._setup_k()
        self._curr_space = 'kspace'
        self.data = self.kdata
        self.set_fft(method)
        self.set_dealiasing(dealiasing)

        # Set transform counters
        self.fwd_count = 0
        self.rev_count = 0

    @timer
    def _allocate_memory(self, method):
        """Allocate memory for data and derivative."""
    
        # Compute global kspace shape for R2C FFT with transpose
        self.global_shape['kspace'] = self.global_shape['xspace'].copy()
        self.global_shape['kspace'][-1]  = self.global_shape['kspace'][-1] / 2 + 1
        self.global_shape['kspace'] = swap_indices(self.global_shape['kspace'])
        
        # Assign data arrays and compute local shapes and offsets
        if method == 'fftw':
            self.kdata, self.xdata, local_n0, local_n0_start, local_n1, local_n1_start = fftw.create_data(self.global_shape['xspace'], com_sys)
            self.local_shape = {'kspace': self.global_shape['kspace'].copy(),
                                'xspace': self.global_shape['xspace'].copy()}
            self.local_shape['kspace'][0] = local_n1
            self.local_shape['xspace'][0] = local_n0
            self.offset = {'xspace': local_n0_start,
                           'kspace': local_n1_start}
        elif method == 'numpy':
            self.kdata = na.zeros(self.global_shape['kspace'], dtype=self.dtype)
            self.xdata = na.zeros(self.global_shape['xspace'])
            self.local_shape = {'kspace': self.global_shape['kspace'].copy(),
                                'xspace': self.global_shape['xspace'].copy()}
            self.offset = {'xspace': 0,
                           'kspace': 0}
        
        else:
            raise NotImplementedError("Specified FFT method not implemented.")

        # Allocate a temp array to hold derivatives
        self.deriv_data = na.zeros_like(self.kdata)

    def __getitem__(self,space):
        """Returns data in specified space, transforming as necessary."""
        
        if space == self._curr_space:
            pass
        elif space == 'xspace':
            self.backward()
        elif space == 'kspace':
            self.forward()
        else:
            raise KeyError("space must be either xspace or kspace.")
        
        return self.data

    def __setitem__(self, space, data):
        """
        This needs to ensure the pointer for the field's data member doesn't 
        change for FFTW. Currently, we do that by slicing the entire data array. 
        
        """
        
        if space == 'xspace':
            self.data = self.xdata
        elif space == 'kspace':
            self.data = self.kdata
        else:
            raise KeyError("space must be either xspace or kspace.")

        if type(data) == float or type(data) == complex:
            self.data[:] = data
        elif data.size < self.data.size:
            mylog.warning("Size of assignment and data don't agree. This may be disallowed in future versions.")
            sli = [slice(i/4+1,i/4+i+1) for i in data.shape]
            self.data[sli] = data
        else:
            sli = [slice(i) for i in self.data.shape]
            self.data[:] = data[sli]

        self._curr_space = space

    def _setup_k(self):
        """Create local wavenumber arrays."""
    
        # Get Nyquist wavenumbers
        self.kny = na.pi * self.global_shape['xspace'] / self.length
        self.kny = swap_indices(self.kny)
        
        # Setup global wavenumber arrays
        self.k = []
  
        for i,ksize in enumerate(self.global_shape['kspace']):
            xsize = swap_indices(self.global_shape['xspace'])[i]
            if ksize == xsize:
                ki = fpack.fftfreq(ksize) * 2. * self.kny[i]
                if xsize % 2 == 0:
                    ki[ksize / 2] *= -1.
            else:
                ki = fpack.fftfreq(xsize)[:ksize] * 2. * self.kny[i]
                if xsize % 2 == 0:
                    ki[-1] *= -1.
            kshape = i * (1,) + (ksize,) + (self.ndim - i - 1) * (1,)
            ki.resize(kshape)
            self.k.append(ki)
        
        names = ['z','y','x'][3-self.ndim:]
        names = swap_indices(names)
        self.k = dict(zip(names, self.k))

        # Restrict to local
        if self.ndim == 2:
            self.k['x'] = self.k['x'][self.offset['kspace']:self.offset['kspace'] + self.local_shape['kspace'][0]]
        else:
            self.k['y'] = self.k['y'][self.offset['kspace']:self.offset['kspace'] + self.local_shape['kspace'][0]]

    def find_mode(self, mode):
        """
        Test if object has a given mode, return index for closest mode if so.

        Parameters
        ----------
        mode : tuple of ints or floats
            Tuple describing physical wavevector for which to search.  Recall 
            kspace ordering (ky, kz, kx) for 3D, (kx, ky) for 2D.

        Returns
        -------
        index : tuple of ints, or bool
            Tuple of indexes to the closest mode, k, if input mode is present
            in (k - dk/2 <= mode < k + dk/2) in all dimensions. None otherwise.

        """
        
        # Get k-mode spacing
        dk = 2 * na.pi / swap_indices(self.length)
        
        # Intersect applicable mode sets by dimension
        index = [None] * self.ndim
        for name, kn in self.k.iteritems():
            i = self.ktrans[name]
            ilist = na.where((mode[i] >= kn - dk[i] / 2.) & 
                             (mode[i] <  kn + dk[i] / 2.))
            if ilist[i].size == 0:
                return None
            else:
                index[i] = ilist[i][0]
                
        return index

    def set_fft(self, method):
        """Assign fft method."""
        
        mylog.debug("Setting FFT method to %s." % method)
        
        if method == 'fftw':
            self.fplan = fftw.rPlan(self.xdata, self.kdata, com_sys, shape=self.global_shape['xspace'], direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])
            self.rplan = fftw.rPlan(self.xdata, self.kdata, com_sys, shape=self.global_shape['xspace'], direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
            self.fft = self.fwd_fftw
            self.ifft = self.rev_fftw
            
        elif method == 'numpy':
            if com_sys.nproc > 1:
                raise NotImplementedError("Numpy FFT not implemented in parallel.")
            self.fft = self.fwd_np
            self.ifft = self.rev_np
            
        else:
            raise NotImplementedError("Specified FFT method not implemented.")
            
    def fwd_fftw(self):
        self.fplan()
        self.kdata /= self.global_shape['xspace'].prod()
        
    def rev_fftw(self):
        self.rplan()

    def fwd_np(self):
        tr = [1, 0, 2][:self.ndim]
        self.kdata = na.transpose(fpack.rfftn(self.xdata / self.global_shape['xspace'].prod()), tr)

    def rev_np(self):
        tr = [1, 0, 2][:self.ndim]
        self.xdata = fpack.irfftn(na.transpose(self.kdata, tr)) * self.global_shape['xspace'].prod()

    def forward(self):
        """FFT method to go from xspace to kspace."""
        
        if self._curr_space == 'kspace':
            raise ValueError("Forward transform cannot be called from kspace.")
        
        self.fft()
        self.data = self.kdata
        self._curr_space = 'kspace'
        self.dealias()
        self.fwd_count += 1

    def backward(self):
        """IFFT method to go from kspace to xspace."""
        
        if self._curr_space == 'xspace':
            raise ValueError("Backward transform cannot be called from xspace.")
        
        self.dealias()
        self.ifft()
        self.data = self.xdata
        self._curr_space = 'xspace'
        self.rev_count += 1

    def set_dealiasing(self, dealiasing):
        """Assign dealiasing method."""
    
        mylog.debug("Setting dealiasing method to %s." % dealiasing)
        
        if dealiasing == '2/3':
            self.dealias = self.dealias_23
        elif dealiasing == '2/3 cython':
            if self.ndim == 2:
                from dealias_cy_2d import dealias_23
            else:
                from dealias_cy_3d import dealias_23 
            self._cython_dealias_function = dealias_23
            self.dealias = self.dealias_23_cython
        elif dealiasing == '2/3 spherical':
            self.dealias = self.dealias_23_spherical
        elif dealiasing == 'None':
            self.dealias = self.zero_nyquist
        else:
            raise NotImplementedError("Specified dealiasing method not implemented.")

    def dealias_23(self):
        """Orszag 2/3 dealiasing rule."""
        
        # Zeroing mask   
        if self.ndim ==2:
            dmask = ((na.abs(self.k['x']) >= 2 / 3. * self.kny[0]) | 
                     (na.abs(self.k['y']) >= 2 / 3. * self.kny[1]))
        else:
            dmask = ((na.abs(self.k['x']) >= 2 / 3. * self.kny[2]) | 
                     (na.abs(self.k['y']) >= 2 / 3. * self.kny[0]) |
                     (na.abs(self.k['z']) >= 2 / 3. * self.kny[1]))

        if self._curr_space == 'xspace': 
            self.forward()
        self.data[dmask] = 0.
        
    def dealias_23_cython(self):
        """Orszag 2/3 dealiasing rule implemented in cython."""
        
        if self._curr_space == 'xspace': 
            self.forward()
        
        if self.ndim == 2:
            self._cython_dealias_function(self.data, self.k['x'], self.k['y'], self.kny)
        else:
            self._cython_dealias_function(self.data, self.k['x'], self.k['y'], self.k['z'], self.kny)

    def dealias_23_spherical(self):
        """Spherical 2/3 dealiasing rule."""
        
        # Zeroing mask   
        dmask = (na.sqrt(self.k2()) >= 2 / 3. * na.min(self.kny))

        if self._curr_space == 'xspace': 
            self.forward()
        self.data[dmask] = 0.
        
    def deriv(self, dim):
        """Calculate derivative along specified dimension."""
        
        if self._curr_space == 'xspace': 
            self.forward()
        na.multiply(self.data, 1j*self.k[dim], self.deriv_data)
        return self.deriv_data

    def k2(self, no_zero=False):
        """
        Calculate wavenumber magnitudes squared.  If keyword 'no_zero' is True, 
        set the mean mode amplitude to 1 (useful for division).
        
        """
        
        k2 = na.zeros(self.local_shape['kspace'])
        for k in self.k.values():
            k2 += k ** 2
        if no_zero:
            k2[k2 == 0] = 1.
            
        return k2
        
    def zero_nyquist(self):
        """Zero out the Nyquist space in each dimension."""
        
        # Zeroing mask   
        if self.ndim ==2:
            dmask = ((na.abs(self.k['x']) == self.kny[0]) | 
                     (na.abs(self.k['y']) == self.kny[1]))
        else:
            dmask = ((na.abs(self.k['x']) == self.kny[2]) | 
                     (na.abs(self.k['y']) == self.kny[0]) |
                     (na.abs(self.k['z']) == self.kny[1]))

        if self._curr_space == 'xspace': 
            self.forward()
        self.data[dmask] = 0.

    def zero_under_eps(self):
        """Zero out any modes with coefficients smaller than machine epsilon."""
        
        if self._curr_space == 'xspace': 
            self.forward()
        self.data[na.abs(self.data) < self.__eps] = 0.

    def save(self, dataset):
        """
        Save data to HDF5 dataset.

        Parameters
        ----------
        dataset : h5py dataset object

        """
        if self._curr_space == 'xspace':
            dataset[:] = st.as_strided(self.data,
                                       shape = self.data.shape,
                                       strides = self.data.strides)
        else:
            dataset[:] = self.data
        dataset.attrs['space'] = self._curr_space

    def xspace_grid(self):
        """Return the xspace grid for the local processor."""
        
        # Create integer array based on local shape and offset
        grid = na.mgrid[[slice(i) for i in self.local_shape['xspace']]]
        grid = na.asfarray(grid)
        grid[0] += self.offset['xspace']
        
        # Multiply integer array by grid spacing
        dx = self.length / self.global_shape['xspace']
        for i in xrange(self.ndim):
            grid[i] *= dx[i]

        return grid


class FourierShearRepresentation(FourierRepresentation):
    """

    """
    def __init__(self, sd, shape, length, left_edge=[0,0,-na.pi],**kwargs):    
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
        self.left_edge = left_edge

    def set_fft(self, method):
        if method == 'fftw':
            self.fplan_yz = fftw.PlanPlane(self.data, 
                                           direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])
            self.rplan_yz = fftw.PlanPlane(self.data, 
                                           direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
            self.fplan_x = fftw.PlanPencil(self.data, 
                                           direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])
            self.rplan_x = fftw.PlanPencil(self.data, 
                                           direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])

            self.fft = self.fwd_fftw
            self.ifft = self.rev_fftw
        if method == 'numpy':
            self.fft = self.fwd_np
            self.ifft = self.rev_np

    def rev_np(self):
        """IFFT method to go from kspace to xspace."""
        
        deltay = self.shear_rate * self.sd.time 
        x = na.linspace(self.left_edge[-1], self.left_edge[-1]+self.length[-1], self.shape[-1], endpoint=False)
        
        # Do x fft
        self.data = fpack.ifft(self.data, axis=-1) * na.sqrt(self.shape[-1])
        
        # Phase shift
        self.data *= na.exp(-1j * self.k['y'] * x * deltay)
        
        # Do y fft
        self.data = fpack.ifft(self.data, axis=-2) * na.sqrt(self.shape[-2])
        
        # Do z fft
        if self.ndim == 3:
            self.data = fpack.ifft(self.data, axis=0) * na.sqrt(self.shape[0])
        
    def fwd_np(self):
        """FFT method to go from xspace to kspace."""
        
        deltay = self.shear_rate * self.sd.time
        x = (na.linspace(self.left_edge[-1], self.left_edge[-1]+self.length[-1], self.shape[-1], endpoint=False) +
             na.zeros(self.shape))

        # Do z fft
        if self.ndim == 3:
            self.data = fpack.fft(self.data / na.sqrt(self.shape[0]), axis=0)

        # Do y fft
        self.data = fpack.fft(self.data / na.sqrt(self.shape[-2]), axis=-2)

        # Phase shift
        self.data *= na.exp(1j * self.k['y'] * x * deltay)
        
        # Do x fft
        self.data = fpack.fft(self.data / na.sqrt(self.shape[-1]), axis=-1)

    def fwd_fftw(self):
        deltay = self.shear_rate * self.sd.time
        x = (na.linspace(self.left_edge[-1], self.left_edge[-1]+self.length[-1], self.shape[-1], endpoint=False) +
             na.zeros(self.shape))

        # do y-z fft
        self.fplan_yz()

        # Phase shift
        self.data *= na.exp(1j * self.k['y'] * x * deltay)
        
        # Do x fft
        self.fplan_x()
        self.data /= (self.data.size * com_sys.nproc)

    def rev_fftw(self):
        deltay = self.shear_rate * self.sd.time 
        x = na.linspace(self.left_edge[-1], self.left_edge[-1]+self.length[-1], self.shape[-1], endpoint=False)
        
        # Do x fft
        self.rplan_x()

        # Phase shift
        self.data *= na.exp(-1j * self.k['y'] * x * deltay)
        
        # Do y-z fft
        self.rplan_yz()
        self.data.imag = 0.

    def _update_k(self):
        """Evolve modes due to shear."""

        self.k['x'] = self.kx - self.shear_rate * self.sd.time * self.k['y']
        while self.k['x'].min() < -self.kny[-1]:
            self.k['x'][self.k['x'] < -self.kny[-1]] += 2 * self.kny[-1]
        while self.k['x'].max() >= self.kny[-1]:
            self.k['x'][self.k['x'] >= self.kny[-1]] -= 2 * self.kny[-1]
            
# class ParallelFourierShearRepresentation(ParallelFourierRepresentation, FourierShearRepresentation):
#     def __init__(self, sd, shape, length, dtype='complex128', method='fftw',
#                  dealiasing='2/3 cython'):
#         ParallelFourierRepresentation.__init__(self, sd, shape, length, dtype=dtype, method=method, dealiasing=dealiasing)
#         FourierShearRepresentation.__init__(self, sd, shape, length, dtype=dtype, method=method, dealiasing=dealiasing)
#         self.nproc = com_sys.nproc
#         self.myproc = com_sys.myproc
#         self.offset = na.array([0, 0, self.myproc * shape[2]])

#     def fwd_np(self):
#         """FFT method to go from xspace to kspace."""
        
#         deltay = self.shear_rate * self.sd.time
#         x = na.linspace(self.left_edge[-1], self.left_edge[-1]+self._length['kspace'][-1], self._shape['kspace'][-1], endpoint=False)

#         # Do y-z fft
#         self.data = fpack.fftn(self.data, axes=(0,1))

#         # communicate
#         recvbuf = self.communicate('forward')

#         # Phase shift
#         recvbuf *= na.exp(1j * self.k['y'] * x * deltay)
        
#         # Do x fft
#         self.data = fpack.fft(recvbuf, axis=2)
#         self.data /= (self.data.size * com_sys.nproc)

#     def rev_np(self):
#         """IFFT method to go from kspace to xspace."""
#         deltay = self.shear_rate * self.sd.time 
#         x = na.linspace(self.left_edge[-1], self.left_edge[-1]+self.length[-1], self.shape[-1], endpoint=False)

#         # Do x fft
#         self.data = fpack.ifft(self.data, axis=2)
        
#         # Phase shift
#         self.data *= na.exp(-1j * self.k['y'] * x * deltay)

#         # communicate
#         recvbuf = self.communicate('backward')
        
#         # Do y-z fft
#         self.data = fpack.ifftn(recvbuf, axes=(0,1))
#         self.data *= (self.data.size * com_sys.nproc)

#     def fwd_fftw(self):
#         deltay = self.shear_rate * self.sd.time
#         x = na.linspace(self.left_edge[-1], self.left_edge[-1] + self._length['kspace'][-1], self._shape['kspace'][-1], endpoint=False)

#         # do y-z fft
#         self.fplan_yz()
#         a = self.communicate('forward')
#         self.data.shape = self._shape['kspace']
#         self.data[:] = a[:]

#         # Phase shift
#         self.data *= na.exp(1j * self.k['y'] * x * deltay)
        
#         # Do x fft
#         self.fplan_x()
#         self.data /= (self.data.size * com_sys.nproc)

#     def rev_fftw(self):
#         deltay = self.shear_rate * self.sd.time
#         x = na.linspace(self.left_edge[-1], self.left_edge[-1]+self.length[-1], self.shape[-1], endpoint=False)

#         # Do x fft
#         self.rplan_x()

#         # Phase shift
#         self.data *= na.exp(-1j * self.k['y'] * x * deltay)
        
#         # Communicate
#         a = self.communicate('backward')
#         self.data.shape = self._shape['xspace']
#         self.data[:] = a[:]

#         # Do y-z fft
#         self.rplan_yz()
#         self.data.imag = 0.

class SphericalHarmonicRepresentation(FourierRepresentation):
    """Dedalus should eventually support spherical and cylindrical geometries.

    """
    pass
