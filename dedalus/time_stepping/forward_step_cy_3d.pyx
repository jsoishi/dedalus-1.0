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
def linear_step(np.ndarray[DTYPE1_t, ndim=3] start not None,
                np.ndarray[DTYPE1_t, ndim=3] deriv not None,
                np.ndarray[DTYPE1_t, ndim=3] output not None,
                float dt):
    """Forward-Euler step."""

    assert start.dtype == DTYPE1
    assert deriv.dtype == DTYPE1
    assert output.dtype == DTYPE1

    cdef int xmax = output.shape[2]
    cdef int ymax = output.shape[0]
    cdef int zmax = output.shape[1]
    cdef unsigned int x, y, z

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                output[y, z, x] = start[y, z, x] + deriv[y, z, x] * dt

@cython.boundscheck(False)
@cython.wraparound(False)
def intfac_step(np.ndarray[DTYPE1_t, ndim=3] start not None,
                np.ndarray[DTYPE1_t, ndim=3] deriv not None,
                np.ndarray[DTYPE1_t, ndim=3] output not None,
                np.ndarray[DTYPE2_t, ndim=3] intfactor not None,
                float dt):
    """Integrating factor step."""

    assert start.dtype == DTYPE1
    assert deriv.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2

    cdef int xmax = output.shape[2]
    cdef int ymax = output.shape[0]
    cdef int zmax = output.shape[1]
    cdef unsigned int x, y, z
    cdef DTYPE1_t eif

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                if intfactor[y, z, x] == 0.:
                    output[y, z, x] = start[y, z, x] + deriv[y, z, x] * dt
                else:
                    eif = exp(intfactor[y, z, x] * dt)
                    output[y, z, x] = (start[y, z, x] + deriv[y, z, x] * (eif - 1.) / intfactor[y, z, x]) / eif
