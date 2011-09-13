from _fftw cimport *

cimport numpy as np
import numpy as np

cdef class Plan:
    cdef fftw_plan _fftw_plan
    cdef np.ndarray _data
    
    def __cinit__(self, data, direction='FFTW_FORWARD', flags=['FFTW_MEASURE']):
        self._data = data
        if len(data.shape) == 2:
            self._fftw_plan = fftw_plan_dft_2d(data.shape[0],
                                               data.shape[1],
                                               <complex *> self._data.data,
                                               <complex *> self._data.data,
                                               FFTW_FORWARD,
                                               FFTW_MEASURE)
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
