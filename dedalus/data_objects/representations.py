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
from dedalus.utils.parallelism import com_sys, swap_indices, get_plane
from dedalus.utils.logger import mylog
from dedalus.utils.fftw import fftw
from dedalus.utils.timer import timer
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

    timer = timer

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

        self.require_space(space)
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
            self.kdata, self._mdata, self.xdata, local_n0, local_n0_start, local_n1, local_n1_start = fftw.create_data(self.global_shape['xspace'], com_sys)
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

        self._static_k = True

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

    def find_mode(self, mode, exact=False):
        """
        Test if object has a given mode, return index for closest mode if so.

        Parameters
        ----------
        mode : tuple of ints or floats
            Tuple describing physical wavevector for which to search.  Recall
            kspace ordering (ky, kz, kx) for 3D, (kx, ky) for 2D.
        exact : bool
            Only return index if mode matches exactly (i.e. no binning).

        Returns
        -------
        index : tuple of ints, or bool
            Tuple of indexes to the closest mode, k, if input mode is present
            in (k - dk/2 <= mode < k + dk/2) in all dimensions. None otherwise.

        """

        # Get k-mode spacing
        dk = 2 * na.pi / swap_indices(self.length)

        # Intersect applicable mode sets by dimension
        test = na.ones(self.local_shape['kspace'], dtype=bool)
        for name, kn in self.k.iteritems():
            i = self.ktrans[name]
            if exact:
                itest = (kn == mode[i])
            else:
                itest = ((kn <= mode[i] + dk[i] / 2.) &
                         (kn >  mode[i] - dk[i] / 2.))
            test *= itest

        index = zip(*test.nonzero())

        if len(index) == 0:
            return None
        elif len(index) == 1:
            return index[0]
        else:
            raise ValueError("Multiple modes tested true. This shouldn't happen.")

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
        self.fplan = fftw.rPlan(self.xdata, self.kdata, com_sys, self.global_shape['xspace'],
                forward=True, flags=['FFTW_MEASURE'])
        self.rplan = fftw.rPlan(self.xdata, self.kdata, com_sys, self.global_shape['xspace'],
                forward=False, flags=['FFTW_MEASURE'])

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

        self.require_space('kspace')
        self.data[dmask] = 0.

    @timer
    def dealias_23_cython(self):
        """Orszag 2/3 dealiasing rule implemented in cython."""

        self.require_space('kspace')

        if self.ndim == 2:
            self._cython_dealias_function(self.data, self.k['x'], self.k['y'], self.kny)
        else:
            self._cython_dealias_function(self.data, self.k['x'], self.k['y'], self.k['z'], self.kny)

    def dealias_23_spherical(self):
        """Spherical 2/3 dealiasing rule."""

        # Zeroing mask
        dmask = (na.sqrt(self.k2()) >= 2 / 3. * na.min(self.kny))

        self.require_space('kspace')
        self.data[dmask] = 0.

    def deriv(self, dim):
        """Calculate derivative along specified dimension."""

        self.require_space('kspace')
        na.multiply(self.data, 1j * self.k[dim], self.deriv_data)

        return self.deriv_data

    def k2(self, no_zero=False, set_zero=1.):
        """
        Calculate wavenumber magnitudes squared.  If keyword 'no_zero' is True,
        set the mean mode amplitude to 1 (useful for division).

        """

        k2 = na.zeros(self.local_shape['kspace'])
        for k in self.k.values():
            k2 += k ** 2
        if no_zero:
            k2[k2 == 0] = set_zero

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

        self.require_space('kspace')
        self.data[dmask] = 0.

    def enforce_hermitian(self):
        """
        Enforce Hermitian symmetry, zeroing out Nyquist spaces.  Data with
        (kx == 0) & (ky < 0) or (kx, ky == 0) & (kz < 0) will be overwritten
        with the complex conjugate of its Hermitian pair.

        """

        # Zero Nyquist spaces
        self.require_space('kspace')
        self.zero_nyquist()

        if self.ndim == 2:
            # Enforce along kx=0 pencil, which is local to one process
            zindex = self.find_mode((0, 0), exact=True)
            if zindex:
                self.data[0, 0] = self.data[0, 0].real
                nyindex = self.local_shape['kspace'][1] / 2
                grab = self.data[0, 1:nyindex + 1]
                self.data[0, -nyindex:] = grab[::-1].conj()
        else:
            # Gather kx=0 plane from across processes
            proc_data = self.kdata[:, :, 0]
            gathered_data = com_sys.comm.gather(proc_data, root=0)

            if com_sys.myproc == 0:
                plane_data = na.concatenate(gathered_data)
                nyy, nyz = na.array(plane_data.shape) / 2

                # Enforce along ky=0 pencil
                grab = plane_data[0, 1:nyz + 1]
                plane_data[0, -nyz:] = grab[::-1].conj()

                # Enforce along kz=0 pencil
                grab = plane_data[1:nyy + 1, 0]
                plane_data[-nyy:, 0] = grab[::-1].conj()

                # Enforce on ky < 0 side of plane
                grab = plane_data[1:nyy + 1, 1:]
                plane_data[-nyy:, 1:] = grab[::-1, ::-1].conj()
            else:
                plane_data = None

            plane_data = com_sys.comm.bcast(plane_data, root=0)
            lstart = self.offset['kspace']
            lstop = self.local_shape['kspace'][0]
            self.kdata[:, :, 0] = plane_data[lstart:lstop, :]

    def zero_under_eps(self):
        """Zero out any modes with coefficients smaller than machine epsilon."""

        self.require_space('kspace')
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

    timer = timer

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
        self._static_k = False
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

        if method == 'numpy':
            mshape = self.local_shape['xspace'].copy()
            mshape[-1] = mshape[-1] / 2 + 1
            self._mdata = na.zeros(mshape, dtype=self.dtype['kspace'])

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

    def create_fftw_plans(self):
        gmshape = self.global_shape['xspace'].copy()
        gmshape[-1] = gmshape[-1] / 2 + 1

        self._fplan_x = fftw.rPencilPlan(self.xdata, self._mdata, forward=True,
                flags=['FFTW_MEASURE'])
        self._rplan_x = fftw.rPencilPlan(self.xdata, self._mdata, forward=False,
                flags=['FFTW_MEASURE'])
        if self.ndim == 2:
            self._fplan_tr = fftw.TransposePlan(self._mdata, self.kdata, com_sys,
                    gmshape, forward=True, flags=['FFTW_MEASURE'])
            self._rplan_tr = fftw.TransposePlan(self._mdata, self.kdata, com_sys,
                    gmshape, forward=False, flags=['FFTW_MEASURE'])
            self._fplan_y = fftw.PencilPlan(self.kdata, self.kdata, forward=True,
                    flags=['FFTW_MEASURE'])
            self._rplan_y = fftw.PencilPlan(self.kdata, self.kdata, forward=False,
                    flags=['FFTW_MEASURE'])
        else:
            self._fplan_yz = fftw.PlanePlan(self._mdata, self.kdata, com_sys,
                    gmshape, forward=True, flags=['FFTW_MEASURE'])
            self._rplan_yz = fftw.PlanePlan(self._mdata, self.kdata, com_sys,
                    gmshape, forward=False, flags=['FFTW_MEASURE'])

    def fwd_fftw(self):

        # Do x fft
        self._fplan_x()

        # Phase shift
        self._mdata *= na.exp(1j * self._phase_rate * self.sd.time)

        # Do y and z ffts (with MPI transpose)
        if self.ndim == 2:
            self._fplan_tr()
            self._fplan_y()
        else:
            self._fplan_yz()

        # Normalize
        self.kdata /= self.global_shape['xspace'].prod()

    def rev_fftw(self):

        # Do y and z iffts (with MPI transpose)
        if self.ndim == 2:
            self._rplan_y()
            self._rplan_tr()
        else:
            self._rplan_yz()

        # Phase shift
        self._mdata *= na.exp(-1j * self._phase_rate * self.sd.time)

        # Do x ifft
        self._rplan_x()

    def fwd_np(self):

        # Do x fft
        self._mdata[:] = fpack.rfft(self.xdata, axis=-1)

        # Phase shift
        self._mdata *= na.exp(1j * self._phase_rate * self.sd.time)

        # Transpose
        tr = [1, 0, 2][:self.ndim]
        self.kdata[:] = na.transpose(self._mdata, tr)

        # Do y and z ffts
        if self.ndim == 2:
            self.kdata[:] = fpack.fft(self.kdata, axis=1)
        else:
            self.kdata[:] = fpack.fftn(self.kdata, axes=(0,1))

        # Correct numpy normalization
        self.kdata /= self.global_shape['xspace'].prod()

    def rev_np(self):

        # Do y and z iffts
        if self.ndim == 2:
            self.kdata[:] = fpack.ifft(self.kdata, axis=1)
        else:
            self.kdata[:] = fpack.ifftn(self.kdata, axes=(0,1))

        # Transpose
        tr = [1, 0, 2][:self.ndim]
        self._mdata[:] = na.transpose(self.kdata, tr)

        # Phase shift
        self._mdata *= na.exp(-1j * self._phase_rate * self.sd.time)

        # Do x ifft
        self.xdata[:] = fpack.irfft(self._mdata, axis=-1)

        # Correct numpy normalization
        self.kdata *= self.global_shape['xspace'].prod()

class SphericalHarmonicRepresentation(FourierRepresentation):
    """
    Dedalus should eventually support spherical and cylindrical geometries.

    """

    pass
