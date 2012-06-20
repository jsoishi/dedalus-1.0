"""Custom FFTW wrappers for Dedalus

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
v
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
#from _fftw cimport *
from _fftw cimport fftw_iodim, FFTW_FORWARD, FFTW_BACKWARD,FFTW_MEASURE,FFTW_DESTROY_INPUT,FFTW_UNALIGNED,FFTW_CONSERVE_MEMORY,FFTW_EXHAUSTIVE,FFTW_PRESERVE_INPUT,FFTW_PATIENT,FFTW_ESTIMATE, FFTW_MPI_TRANSPOSED_IN, FFTW_MPI_TRANSPOSED_OUT, fftw_plan
cimport _fftw as fftw
cimport libc
cimport numpy as np
import numpy as np
from mpi4py cimport MPI
from mpi4py.mpi_c cimport *
fftw_flags = {'FFTW_FORWARD': FFTW_FORWARD,
              'FFTW_BACKWARD': FFTW_BACKWARD,        
              'FFTW_MEASURE': FFTW_MEASURE,
              'FFTW_DESTROY_INPUT': FFTW_DESTROY_INPUT,
              'FFTW_UNALIGNED': FFTW_UNALIGNED,
              'FFTW_CONSERVE_MEMORY': FFTW_CONSERVE_MEMORY,
              'FFTW_EXHAUSTIVE': FFTW_EXHAUSTIVE,
              'FFTW_PRESERVE_INPUT': FFTW_PRESERVE_INPUT,
              'FFTW_PATIENT': FFTW_PATIENT,
              'FFTW_ESTIMATE': FFTW_ESTIMATE,
              'FFTW_MPI_TRANSPOSED_IN': FFTW_MPI_TRANSPOSED_IN,
              'FFTW_MPI_TRANSPOSED_OUT': FFTW_MPI_TRANSPOSED_OUT
              }

from cpython cimport Py_INCREF

cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims, np.npy_intp* strides,
                                void* data, int flags, object obj)

cdef extern from "stdlib.h":
    void free(void *ptr)

DTYPEi = np.int64
ctypedef np.int64_t DTYPEi_t

cdef class fftwMemoryReleaser:
    cdef void* memory

    def __cinit__(self):
        self.memory = NULL
                  
    def __dealloc__(self):
        if self.memory:
            #release memory
            fftw.fftw_free(self.memory)
            #print "memory released", hex(<long>self.memory)

cdef fftwMemoryReleaser MemoryReleaserFactory(void* ptr):
    cdef fftwMemoryReleaser mr = fftwMemoryReleaser.__new__(fftwMemoryReleaser)
    mr.memory = ptr
    return mr

def fftw_mpi_init():
    fftw.fftw_mpi_init()

def create_data(np.ndarray[DTYPEi_t, ndim=1] shape not None, com_sys):
    """this allocates data using fftw's allocate routines. We allocate
    the complex part directly, shaping it with the local_n1 returned
    by FFTW's parallel interface.

    create_data allocates shape[-1]/2 + 1 complex arrays. 

    inputs
    ------

    shape -- numpy array of int64 giving the *global* logical shape in x-space grid points. 

    """
    np.import_array()
    cdef np.ndarray array
    cdef np.dtype dtype = np.dtype('complex128')
    Py_INCREF(dtype)

    cdef complex *data
    cdef size_t n, local_x0, local_x0_start, local_k1, local_k1_start
    cdef MPI.Comm comm = com_sys.comm
    cdef MPI_Comm c_comm = comm.ob_mpi
    cdef np.ndarray[DTYPEi_t, ndim=1] kshape = shape.copy()
    kshape[-1] = kshape[-1]/2 + 1 # create global k shape (not transposed)

    if shape.size == 2:
        n = fftw.fftw_mpi_local_size_2d_transposed(<size_t> kshape[0],
                                                   <size_t> kshape[1],
                                                   c_comm,
                                                   &local_x0, &local_x0_start,
                                                   &local_k1, &local_k1_start)
    elif shape.size == 3:
        n = fftw.fftw_mpi_local_size_3d_transposed(<size_t> kshape[0],
                                                   <size_t> kshape[1],
                                                   <size_t> kshape[2],
                                                   c_comm,
                                                   &local_x0, &local_x0_start,
                                                   &local_k1, &local_k1_start)
    else:
        raise ValueError("Data must be > 1 dimensional for MPI.")

    # now local k shape, properly transposed
    kshape[1] = kshape[0]
    kshape[0] = local_k1

    data = fftw.fftw_alloc_complex(n)
    cdef int i
    for i in range(n):
        data[i] = 0j
    rank = len(kshape)

    # this is necessary to pass the strides without them being garbage
    # collected, though why is not clear...
    cdef np.ndarray[DTYPEi_t, ndim=1] np_strides = np.array((1,)+tuple(kshape[1:][::-1])).cumprod()[::-1]
    np_strides *= 16
    cdef size_t *stridesp
    strides = <size_t *> libc.stdlib.malloc(sizeof(size_t) * rank)
    for i in range(rank):
        strides[i] = np_strides[i]

    array = PyArray_NewFromDescr(np.ndarray, dtype, rank, <np.npy_intp *> kshape.data, <np.npy_intp *> strides, <void *> data, np.NPY_DEFAULT, None)
    np.set_array_base(array, MemoryReleaserFactory(data))
    libc.stdlib.free(strides)

    return array, local_x0, local_x0_start, local_k1, local_k1_start

def fftw_mpi_allocate(comm):
    """Allocates memory for local data block. fftw provides the load
    balancing, and so this needs to be done before arrays of local k
    and local x are created.

    """
    pass
    
cdef class Plan:
    cdef fftw_plan _fftw_plan
    cdef np.ndarray _data
    cdef int direction
    cdef int flags
    def __init__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        """A custom wrapping of FFTW for use in Dedalus.

        """
        if direction == 'FFTW_FORWARD':
            self.direction = FFTW_FORWARD
        else:
            self.direction = FFTW_BACKWARD
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
        self._data = data
        if len(data.shape) == 2:
            self._fftw_plan = fftw.fftw_plan_dft_2d(data.shape[0],
                                               data.shape[1],
                                               <complex *> self._data.data,
                                               <complex *> self._data.data,
                                               self.direction,
                                               self.flags)
        elif len(data.shape) == 3:
            self._fftw_plan = fftw.fftw_plan_dft_3d(data.shape[0],
                                               data.shape[1],
                                               data.shape[2],
                                               <complex *> self._data.data,
                                               <complex *> self._data.data,
                                               self.direction,
                                               self.flags)
        else:
            raise RuntimeError("Only 2D and 3D arrays are supported.")

        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")

    def __call__(self):
        if self._fftw_plan != NULL:
            fftw.fftw_execute(self._fftw_plan)

    def __dealloc__(self):
        fftw.fftw_destroy_plan(self._fftw_plan)

    property data:
        def __get__(self):
            return self._data
    
        def __set__(self, inp):
            self._data[:]=inp

cdef class PlanPlane(Plan):
    def __init__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        """PlanPlane returns a special FFTW plan that will take a 2D
        FFT along the z-y planes of a 3D, row-major data array.
        """
        if direction == 'FFTW_FORWARD':
            self.direction = FFTW_FORWARD
        else:
            self.direction = FFTW_BACKWARD
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
        self._data = data

        # allocate 
        cdef int rank = 2
        cdef fftw_iodim *dims = <fftw_iodim *> libc.stdlib.malloc(sizeof(fftw_iodim) * rank)
        cdef int howmany_rank = 1
        cdef fftw_iodim *howmany_dims = <fftw_iodim *> libc.stdlib.malloc(sizeof(fftw_iodim) * howmany_rank)

        # setup transforms
        if len(data.shape) == 2:
            nz = 1
            ny = data.shape[0]
            nx = data.shape[1]
        else:
            nz = data.shape[0]
            ny = data.shape[1]
            nx = data.shape[2]
        dims[0].n = nz #data.shape[0]
        dims[0].ins = ny*nx # data.shape[1]*data.shape[2]
        dims[0].ous = ny*nx # data.shape[1]*data.shape[2]
        dims[1].n = ny # data.shape[1]
        dims[1].ins = nx # data.shape[2]
        dims[1].ous = nx # data.shape[2]

        howmany_dims[0].n = nx #data.shape[2]
        howmany_dims[0].ins = 1
        howmany_dims[0].ous = 1
        
        self._fftw_plan = fftw.fftw_plan_guru_dft(rank, dims,
                                             howmany_rank, howmany_dims,
                                             <complex *> self._data.data,
                                             <complex *> self._data.data,
                                             self.direction, self.flags)

        libc.stdlib.free(dims)
        libc.stdlib.free(howmany_dims)

cdef class PlanPencil(Plan):
    def __init__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        """PlanPencil returns a FFTW plan that will take a 1D
        FFT along the x pencils of a 3D, row-major data array.
        """
        if direction == 'FFTW_FORWARD':
            self.direction = FFTW_FORWARD
        else:
            self.direction = FFTW_BACKWARD
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
        self._data = data

        nx = data.shape[-1]
        ny = data.shape[-2]
        try:
            nz = data.shape[-3]
        except IndexError:
            nz = 1

        cdef np.ndarray n = np.array(nx, dtype='int32')
        cdef int rank = 1
        cdef int howmany = nz*ny
        cdef int istride = 1
        cdef int ostride = istride
        cdef int idist = nx
        cdef int odist = idist

        self._fftw_plan = fftw.fftw_plan_many_dft(rank, <int *> n.data,
                                             howmany,
                                             <complex *> self._data.data,
                                             <int *> n.data,
                                             istride, idist,
                                             <complex *> self._data.data,
                                             <int *> n.data,
                                             ostride, odist,
                                             self.direction,
                                             self.flags)
cdef class rPlan(Plan):
    cdef np.ndarray _xdata, _kdata
    def __init__(self, xdata, kdata, com_sys,shape=None,direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        """rPlan implements out-of-place, real-to-complex,
        complex-to-real transforms.

        """
        cdef MPI.Comm comm = com_sys.comm
        cdef MPI_Comm c_comm = comm.ob_mpi

        if direction == 'FFTW_FORWARD':
            self.direction = FFTW_FORWARD
        else:
            self.direction = FFTW_BACKWARD
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
        self._xdata = xdata
        self._kdata = kdata
        if shape == None:
            shape = xdata.shape
        if len(shape) == 2:
            if self.direction == FFTW_FORWARD:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_OUT']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_r2c_2d(shape[0],
                                                                shape[1],
                                                                <double *> self._xdata.data,
                                                                <complex *> self._kdata.data,
                                                                c_comm,
                                                                self.flags)
            else:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_IN']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_c2r_2d(shape[0],
                                                                shape[1],
                                                                <complex *> self._kdata.data,
                                                                <double *> self._xdata.data,
                                                                c_comm,
                                                                self.flags)

        elif len(shape) == 3:
            if self.direction == FFTW_FORWARD:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_OUT']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_r2c_3d(shape[0],
                                                                shape[1],
                                                                shape[2],
                                                                <double *> self._xdata.data,
                                                                <complex *> self._kdata.data,
                                                                c_comm,
                                                                self.flags)
            else:
                self.flags = self.flags | fftw_flags['FFTW_MPI_TRANSPOSED_IN']
                self._fftw_plan = fftw.fftw_mpi_plan_dft_c2r_3d(shape[0],
                                                                shape[1],
                                                                shape[2],
                                                                <complex *> self._kdata.data,
                                                                <double *> self._xdata.data,
                                                                c_comm,
                                                                self.flags)
        else:
            raise RuntimeError("Only 2D and 3D arrays are supported.")

        if self._fftw_plan == NULL:
            raise RuntimeError("FFTW could not create plan.")

# cdef class rPlanPencil(Plan):
#     cdef np.ndarray _xdata, _kdata
#     def __init__(self, xdata, kdata, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
#         """PlanPencil returns a FFTW plan that will take a 1D
#         FFT along the x pencils of a 3D, row-major data array.
#         """
#         if direction == 'FFTW_FORWARD':
#             self.direction = FFTW_FORWARD
#         else:
#             self.direction = FFTW_BACKWARD
#         for f in flags:
#             self.flags = self.flags | fftw_flags[f]
#         self._xdata = xdata
#         self._kdata = kdata

#         nx = data.shape[-1]
#         ny = data.shape[-2]
#         try:
#             nz = data.shape[-3]
#         except IndexError:
#             nz = 1

#         cdef np.ndarray n = np.array(nx, dtype='int32')
#         cdef int rank = 1
#         cdef int howmany = nz*ny
#         cdef int istride = 1
#         cdef int ostride = istride
#         cdef int idist = nx
#         cdef int odist = idist
        
#         if self.direction == FFTW_FORWARD:
#             self._fftw_plan = fftw_plan_many_r2c(rank, <int *> n.data,
#                                                  howmany,
#                                                  <double *> self._xdata.data,
#                                                  <int *> n.data,
#                                                  istride, idist,
#                                                  <complex *> self._kdata.data,
#                                                  <int *> n.data,
#                                                  ostride, odist,
#                                                  self.direction,
#                                                  self.flags)
#         else:
#             self._fftw_plan = fftw_plan_many_c2r(rank, <int *> n.data,
#                                                  howmany,
#                                                  <complex *> self._kdata.data,
#                                                  <int *> n.data,
#                                                  istride, idist,
#                                                  <double *> self._xdata.data,
#                                                  <int *> n.data,
#                                                  ostride, odist,
#                                                  self.direction,
#                                                  self.flags)
