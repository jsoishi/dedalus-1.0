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
def euler(np.ndarray[DTYPE1_t, ndim=3] start not None,
          np.ndarray[DTYPE1_t, ndim=3] output not None,
          np.ndarray[DTYPE1_t, ndim=3] deriv not None,
          DTYPE2_t dt):
    """First-order forward-Euler step."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv.dtype == DTYPE1

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                output[y, z, x] = start[y, z, x] + dt * deriv[y, z, x]


@cython.boundscheck(False)
@cython.wraparound(False)
def etd1(np.ndarray[DTYPE1_t, ndim=3] start not None,
         np.ndarray[DTYPE1_t, ndim=3] output not None,
         np.ndarray[DTYPE1_t, ndim=3] deriv not None,
         np.ndarray[DTYPE2_t, ndim=3] intfactor not None,
         DTYPE2_t dt):
    """First-order ETD step."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z
    cdef DTYPE2_t c, eif

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                if intfactor[y, z, x] == 0.:
                    output[y, z, x] = start[y, z, x] + dt * deriv[y, z, x]
                else:
                    c = intfactor[y, z, x]
                    eif = exp(c * dt)
                    output[y, z, x] = start[y, z, x] * eif + (eif - 1.) * deriv[y, z, x] / c


@cython.boundscheck(False)
@cython.wraparound(False)
def etd2rk1(np.ndarray[DTYPE1_t, ndim=3] start not None,
            np.ndarray[DTYPE1_t, ndim=3] output not None,
            np.ndarray[DTYPE1_t, ndim=3] deriv1 not None,
            np.ndarray[DTYPE1_t, ndim=3] deriv2 not None,
            np.ndarray[DTYPE2_t, ndim=3] intfactor not None,
            DTYPE2_t dt):
    """Second step of ETD2RK1."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv1.dtype == DTYPE1
    assert deriv2.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z
    cdef DTYPE2_t c, eif

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                if intfactor[y, z, x] == 0.:
                    output[y, z, x] = start[y, z, x] + dt / 2. * (deriv2[y, z, x] - deriv1[y, z, x])
                else:
                    c = intfactor[y, z, x]
                    eif = exp(c * dt)
                    output[y, z, x] = start[y, z, x] + (eif - c * dt - 1.) * (deriv2[y, z, x] - deriv1[y, z, x]) / (c * c * dt)


@cython.boundscheck(False)
@cython.wraparound(False)
def etd2rk2(np.ndarray[DTYPE1_t, ndim=3] start not None,
            np.ndarray[DTYPE1_t, ndim=3] output not None,
            np.ndarray[DTYPE1_t, ndim=3] deriv1 not None,
            np.ndarray[DTYPE1_t, ndim=3] deriv2 not None,
            np.ndarray[DTYPE2_t, ndim=3] intfactor not None,
            DTYPE2_t dt):
    """Second step of ETD2RK2."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv1.dtype == DTYPE1
    assert deriv2.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z
    cdef DTYPE2_t c, eif

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                if intfactor[y, z, x] == 0.:
                    output[y, z, x] = start[y, z, x] + dt * deriv2[y, z, x]
                else:
                    c = intfactor[y, z, x]
                    eif = exp(c * dt)
                    output[y, z, x] = start[y, z, x] * eif + (((c * dt - 2.) * eif + c * dt + 2.) * deriv1[y, z, x] + (2 * (eif - c * dt - 1.)) * deriv2[y, z, x]) / (c * c * dt)

