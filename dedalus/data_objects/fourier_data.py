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
        
        # Make sure all dimensions make sense
        if self.ndim not in (2, 3):
            raise ValueError("Must use either 2 or 3 dimensions.")
        if len(shape) != len(length):
            raise ValueError("Shape and Length must have same dimensions.")
        
        # Flexible datatypes not currently supported
        self.dtype = {'kspace': 'complex128', 
                      'xspace': 'float64'}
        self._eps = {'kspace': na.finfo(self.dtype['kspace']).eps,
                      'xspace': na.finfo(self.dtype['xspace']).eps}
        
        # Retrieve FFT method and dealiasing method from config
        method = decfg.get('FFT', 'method')
        dealiasing = decfg.get('FFT', 'dealiasing')
        
        # Translation tables
        if self.ndim == 2:
            self.ktrans = {'x':0, 0:'x', 'y':1, 1:'y'}
            self.xtrans = {'x':1, 1:'x', 'y':0, 0:'y'}
        else:
            self.ktrans = {'x':2, 2:'x', 'y':0, 0:'y', 'z':1, 1:'z'}
            self.xtrans = {'x':2, 2:'x', 'y':1, 1:'y', 'z':0, 0:'z'}
            
        # Complete setup
        self.set_fft(method)
        self.set_dealiasing(dealiasing)
        self._setup_k()
        self._curr_space = 'kspace'
        self.data = self.kdata

        # Set transform counters
        self.fwd_count = 0
        self.rev_count = 0
    
    def __getitem__(self, space):
        """Return data in specified space, transforming as necessary."""
        
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
        Set data while ensuring the pointer for the field's data member doesn't 
        change for FFTW. Currently done by slicing the entire data array. 
        
        """
        
        if space == 'xspace':
            self.data = self.xdata
        elif space == 'kspace':
            self.data = self.kdata
        else:
            raise KeyError("space must be either xspace or kspace.")

        # Scalar assignment
        if type(data) == float or type(data) == complex:
            self.data[:] = data
        # Mis-matched size assignment
        elif data.size < self.data.size:
            mylog.warning("Size of assignment and data don't agree. This may be disallowed in future versions.")
            sli = [slice(i/4+1,i/4+i+1) for i in data.shape]
            self.data[sli] = data
        # Regular assignment by slicing
        else:
            sli = [slice(i) for i in self.data.shape]
            self.data[:] = data[sli]

        self._curr_space = space

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
            self.kdata = na.zeros(self.global_shape['kspace'], dtype=self.dtype['kspace'])
            self.xdata = na.zeros(self.global_shape['xspace'], dtype=self.dtype['xspace'])
            self.local_shape = {'kspace': self.global_shape['kspace'].copy(),
                                'xspace': self.global_shape['xspace'].copy()}
            self.offset = {'xspace': 0,
                           'kspace': 0}

        # Allocate an array to hold derivatives
        self.deriv_data = na.zeros_like(self.kdata)
        
        # Incorporate ability to add extra views, etc.
        self._additional_allocation(method)
        
    def _additional_allocation(self, method):
        pass

    def _setup_k(self):
        """Create local wavenumber arrays."""
    
        # Get Nyquist wavenumbers
        self.kny = na.pi * self.global_shape['xspace'] / self.length
        self.kny = swap_indices(self.kny)
        
        # Setup global wavenumber arrays
        rcomp = self.ktrans['x']
        self.k = {}

        for i,ksize in enumerate(self.global_shape['kspace']):
            xsize = swap_indices(self.global_shape['xspace'])[i]
            if i == rcomp:
                ki = fpack.fftfreq(xsize)[:ksize] * 2. * self.kny[i]
                if xsize % 2 == 0:
                    ki[-1] *= -1.
            else:
                ki = fpack.fftfreq(ksize) * 2. * self.kny[i]
                if xsize % 2 == 0:
                    ki[ksize / 2] *= -1.
            kshape = i * (1,) + (ksize,) + (self.ndim - i - 1) * (1,)
            ki.resize(kshape)
            self.k[self.ktrans[i]] = ki

        # Restrict to local
        scomp = self.ktrans[0]
        self.k[scomp] = self.k[scomp][self.offset['kspace']:self.offset['kspace'] + self.local_shape['kspace'][0]]

    def require_space(self, space):
        """Transform to required space if not there already."""
        
        if self._curr_space == space:
            pass
        elif space == 'xspace':
            self.backward()
        elif space == 'kspace':
            self.forward()
        else:
            raise ValueError("space must be either xspace or kspace.")

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
        
        # Allocate memory
        self._allocate_memory(method)
        
        if method == 'fftw':
            self.create_fftw_plans()
            self.fft = self.fwd_fftw
            self.ifft = self.rev_fftw
            
        elif method == 'numpy':
            if com_sys.nproc > 1:
                raise NotImplementedError("Numpy FFT not implemented in parallel.")
            self.fft = self.fwd_np
            self.ifft = self.rev_np
            
        else:
            raise NotImplementedError("Specified FFT method not implemented.")
            
    def create_fftw_plans(self):
        self.fplan = fftw.rPlan(self.xdata, self.kdata, com_sys, shape=self.global_shape['xspace'], direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])
        self.rplan = fftw.rPlan(self.xdata, self.kdata, com_sys, shape=self.global_shape['xspace'], direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
            
    def fwd_fftw(self):
        self.fplan()
        self.kdata /= self.global_shape['xspace'].prod()
        
    def rev_fftw(self):
        self.rplan()

    def fwd_np(self):
        tr = [1, 0, 2][:self.ndim]
        self.kdata[:] = na.transpose(fpack.rfftn(self.xdata / self.global_shape['xspace'].prod()), tr)

    def rev_np(self):
        tr = [1, 0, 2][:self.ndim]
        self.xdata[:] = fpack.irfftn(na.transpose(self.kdata, tr)) * self.global_shape['xspace'].prod()

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
        na.multiply(self.data, 1j * self.k[dim], self.deriv_data)
        
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
        if self.ndim == 2:
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
        self.data[na.abs(self.data) < self._eps['kspace']] = 0.

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

    def xspace_grid(self, open=False):
        """Return the xspace grid for the local processor."""
        
        if open:
            refgrid = na.ogrid
        else:
            refgrid = na.mgrid
        
        # Create integer array based on local shape and offset
        grid = refgrid[[slice(i) for i in na.asfarray(self.local_shape['xspace'])]]
        grid[0] += self.offset['xspace']
        
        # Multiply integer array by grid spacing
        dx = self.length / self.global_shape['xspace']
        for i in xrange(self.ndim):
            grid[i] *= dx[i]

        return grid


class FourierShearRepresentation(FourierRepresentation):
    """
    Fourier representation in a shearing-box domain.
        
    """
    
    timer = Timer()
    
    def __init__(self, sd, shape, length):    
        """
        Fourier representation in a shearing-box domain.
        
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
        
        # Fourier initialization
        FourierRepresentation.__init__(self, sd, shape, length)
        
        # Store initial copy of ky, allocate a fleshed-out ky
        self._ky = self.k['y'].copy()
        self.k['y'] = self.k['y'] * na.ones_like(self.k['x'])
        
        # Calculate phase rate for use in x-orientation transpose
        ksize = self.global_shape['kspace'][self.ktrans['x']]
        xsize = self.global_shape['xspace'][self.xtrans['x']]
        xkx = fpack.fftfreq(xsize)[:ksize] * 2. * self.kny[self.ktrans['x']]
        if xsize % 2 == 0:
            xkx[-1] *= -1.
        xkx.resize((self.ndim - 1) * (1,) + (ksize,))
        y = self.xspace_grid(open=True)[self.xtrans['y']]
        self._phase_rate = -self.sd.parameters['S'] * self.sd.parameters['Omega'] * xkx * y
        
        # Calculate wave rate for use in wavenumber update
        self._wave_rate = -self.sd.parameters['S'] * self.sd.parameters['Omega'] * self.k['x']
        
        # Update wavenumbers in case component is initialized at non-zero time
        self._update_k()
        
    def _additional_allocation(self, method):
        """Allocate memory for intermediate transformation arrays."""

        if method == 'fftw':
            pass
        elif method == 'numpy':
            tempshape = self.global_shape['xspace'].copy()
            tempshape[-1] = tempshape[-1] / 2 + 1
            self._tempdata = na.zeros(tempshape, dtype= self.dtype['kspace'])
    
    def _update_k(self):
        """Evolve wavenumbers due to shear."""

        # Wavenumber shift
        #self.k['y'] = self._ky - self._wave_rate * self.sd.time
        na.subtract(self._ky, self._wave_rate * self.sd.time , self.k['y'])
    
        # Wrap wavenumbers past Nyquist value
        kny_y = self.kny[3 - self.ndim]
        while self.k['y'].min() <= -kny_y:
            self.k['y'][self.k['y'] <= -kny_y] += 2 * kny_y
        while self.k['y'].max() > kny_y:
            self.k['y'][self.k['y'] > kny_y] -= 2 * kny_y
            
        # Dealias
        self.dealias()
        
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
        
        raise NotImplementedError("Find_mode for shear rep is more advanced and needs to be done")

    def create_fftw_plans(self):
        raise NotImplementedError("TO DO")
    
        self.fplan_yz = fftw.PlanPlane(self.data, 
                                           direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])
        self.rplan_yz = fftw.PlanPlane(self.data, 
                                           direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
        self.fplan_x = fftw.PlanPencil(self.data, 
                                           direction='FFTW_FORWARD', flags=['FFTW_MEASURE'])
        self.rplan_x = fftw.PlanPencil(self.data, 
                                           direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
                                           
    def fwd_fftw(self):
        raise NotImplementedError("TO DO")
    
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
        raise NotImplementedError("TO DO")
    
        deltay = self.shear_rate * self.sd.time 
        x = na.linspace(self.left_edge[-1], self.left_edge[-1]+self.length[-1], self.shape[-1], endpoint=False)
        
        # Do x fft
        self.rplan_x()

        # Phase shift
        self.data *= na.exp(-1j * self.k['y'] * x * deltay)
        
        # Do y-z fft
        self.rplan_yz()
        self.data.imag = 0.
        
    def fwd_np(self):

        # Do x fft
        self._tempdata[:] = fpack.rfft(self.xdata / self.global_shape['xspace'].prod(), axis=-1)
        
        # Phase shift
        self._tempdata *= na.exp(1j * self._phase_rate * self.sd.time)
        
        # Transpose
        tr = [1, 0, 2][:self.ndim]
        self.kdata[:] = na.transpose(self._tempdata, tr)
        
        # Do y and z ffts
        if self.ndim == 2:
            self.kdata[:] = fpack.fft(self.kdata, axis=1)
        else:
            self.kdata[:] = fpack.fftn(self.kdata, axes=(0,1))

    def rev_np(self):
    
        # Do y and z ffts
        if self.ndim == 2:
            self.kdata[:] = fpack.ifft(self.kdata, axis=1)
        else:
            self.kdata[:] = fpack.ifftn(self.kdata, axes=(0,1))
            
        # Transpose
        tr = [1, 0, 2][:self.ndim]
        self._tempdata[:] = na.transpose(self.kdata, tr)        

        # Phase shift
        self._tempdata *= na.exp(-1j * self._phase_rate * self.sd.time)
        
        # Do x fft
        self.xdata[:] = fpack.irfft(self._tempdata, axis=-1) * self.global_shape['xspace'].prod()

class SphericalHarmonicRepresentation(FourierRepresentation):
    """
    Dedalus should eventually support spherical and cylindrical geometries.

    """
    
    pass
