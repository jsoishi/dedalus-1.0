import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp

DTYPE1 = np.complex128
ctypedef np.complex128_t DTYPE1_t

DTYPE2 = np.float64
ctypedef np.float64_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)
def linear_step(np.ndarray[DTYPE1_t, ndim=2] start not None,
                np.ndarray[DTYPE1_t, ndim=2] deriv not None,
                np.ndarray[DTYPE1_t, ndim=2] output not None,
                float dt):
    """Forward-Euler step."""

    assert start.dtype == DTYPE1
    assert deriv.dtype == DTYPE1
    assert output.dtype == DTYPE1
    
    cdef int xmax = output.shape[0]
    cdef int ymax = output.shape[1]
    cdef unsigned int x, y

    for x in range(xmax):
        for y in range(ymax):
            output[x, y] = start[x, y] + deriv[x, y] * dt
                
@cython.boundscheck(False)
@cython.wraparound(False)
def intfac_step(np.ndarray[DTYPE1_t, ndim=2] start not None,
                np.ndarray[DTYPE1_t, ndim=2] deriv not None,
                np.ndarray[DTYPE1_t, ndim=2] output not None,
                np.ndarray[DTYPE2_t, ndim=2] intfactor not None,
                float dt):
    """Integrating factor step."""

    assert start.dtype == DTYPE1
    assert deriv.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2
    
    cdef int xmax = output.shape[0]
    cdef int ymax = output.shape[1]
    cdef unsigned int x, y
    cdef DTYPE1_t eif

    for x in range(xmax):
        for y in range(ymax):
            if intfactor[x, y] == 0.:
                output[x, y] = start[x, y] + deriv[x, y] * dt
            else:
                eif = exp(intfactor[x, y] * dt)
                output[x, y] = (start[x, y] + deriv[x, y] * (eif - 1.) / intfactor[x, y]) / eif
