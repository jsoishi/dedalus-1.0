from _fftw cimport *
cimport libc
cimport numpy as np
import numpy as np
fftw_flags = {'FFTW_FORWARD': FFTW_FORWARD,
              'FFTW_BACKWARD': FFTW_BACKWARD,        
              'FFTW_MEASURE': FFTW_MEASURE,
              'FFTW_DESTROY_INPUT': FFTW_DESTROY_INPUT,
              'FFTW_UNALIGNED': FFTW_UNALIGNED,
              'FFTW_CONSERVE_MEMORY': FFTW_CONSERVE_MEMORY,
              'FFTW_EXHAUSTIVE': FFTW_EXHAUSTIVE,
              'FFTW_PRESERVE_INPUT': FFTW_PRESERVE_INPUT,
              'FFTW_PATIENT': FFTW_PATIENT,
              'FFTW_ESTIMATE': FFTW_ESTIMATE
              }

cdef class Plan:
    cdef fftw_plan _fftw_plan
    cdef np.ndarray _data
    cdef int direction
    cdef int flags
    def __cinit__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        if direction == 'FFTW_FORWARD':
            self.direction = FFTW_FORWARD
        else:
            self.direction = FFTW_BACKWARD
        for f in flags:
            self.flags = self.flags | fftw_flags[f]
        self._data = data
        if len(data.shape) == 2:
            self._fftw_plan = fftw_plan_dft_2d(data.shape[0],
                                               data.shape[1],
                                               <complex *> self._data.data,
                                               <complex *> self._data.data,
                                               self.direction,
                                               self.flags)
        elif len(data.shape) == 3:
            self._fftw_plan = fftw_plan_dft_3d(data.shape[0],
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

            
    def __init__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        pass

    def __call__(self):
        if self._fftw_plan != NULL:
            fftw_execute(self._fftw_plan)

    def __dealloc__(self):
        fftw_destroy_plan(self._fftw_plan)

    property data:
        def __get__(self):
            return self._data
    
        def __set__(self, inp):
            self._data[:]=inp

cdef class PlanPlane(Plan):
    def __cinit__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
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
        dims[0].n = data.shape[0]
        dims[0].ins = data.shape[1]*data.shape[2]
        dims[0].ous = data.shape[1]*data.shape[2]
        dims[1].n = data.shape[1]
        dims[1].ins = data.shape[2]
        dims[1].ous = data.shape[2]

        howmany_dims[0].n = data.shape[2]
        howmany_dims[0].ins = 1
        howmany_dims[0].ous = 1
        
        self._fftw_plan = fftw_plan_guru_dft(rank, dims, howmany_rank, howmany_dims,
                                             <complex *> self._data.data, <complex *> self._data.data,
                                             self.direction, self.flags)

        libc.stdlib.free(dims)
        libc.stdlib.free(howmany_dims)

cdef class PlanPencil(Plan):
    def __cinit__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
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

        cdef np.ndarray n = np.array(data.shape, dtype='int32')
        cdef int rank = 1
        cdef int howmany = data.shape[0]*data.shape[1]
        cdef int istride = 1
        cdef int ostride = istride
        cdef int idist = data.shape[2]
        cdef int odist = idist

        self._fftw_plan = fftw_plan_many_dft(rank, <int *> n.data,
                                             howmany,
                                             <complex *> self._data.data,
                                             <int *> n. data,
                                             istride, idist,
                                             <complex *> self._data.data,
                                             <int *> n. data,
                                             ostride, odist,
                                             self.direction,
                                             self.flags)
