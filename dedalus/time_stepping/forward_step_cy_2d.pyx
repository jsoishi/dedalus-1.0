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
def euler(np.ndarray[DTYPE1_t, ndim=2] start not None,
          np.ndarray[DTYPE1_t, ndim=2] output not None,
          np.ndarray[DTYPE1_t, ndim=2] deriv not None,
          DTYPE2_t dt):
    """First-order forward-Euler step."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv.dtype == DTYPE1

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y

    for x in range(xmax):
        for y in range(ymax):
            output[x, y] = start[x, y] + dt * deriv[x, y]


@cython.boundscheck(False)
@cython.wraparound(False)
def etd1(np.ndarray[DTYPE1_t, ndim=2] start not None,
         np.ndarray[DTYPE1_t, ndim=2] output not None,
         np.ndarray[DTYPE1_t, ndim=2] deriv not None,
         np.ndarray[DTYPE2_t, ndim=2] intfactor not None,
         DTYPE2_t dt):
    """First-order ETD step."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y
    cdef DTYPE2_t c, eif

    for x in range(xmax):
        for y in range(ymax):
            if intfactor[x, y] == 0.:
                output[x, y] = start[x, y] + dt * deriv[x, y]
            else:
                c = intfactor[x, y]
                eif = exp(c * dt)
                output[x, y] = start[x, y] * eif + (eif - 1.) * deriv[x, y] / c


@cython.boundscheck(False)
@cython.wraparound(False)
def etd2rk1(np.ndarray[DTYPE1_t, ndim=2] start not None,
            np.ndarray[DTYPE1_t, ndim=2] output not None,
            np.ndarray[DTYPE1_t, ndim=2] deriv1 not None,
            np.ndarray[DTYPE1_t, ndim=2] deriv2 not None,
            np.ndarray[DTYPE2_t, ndim=2] intfactor not None,
            DTYPE2_t dt):
    """Second step of ETD2RK1."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv1.dtype == DTYPE1
    assert deriv2.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y
    cdef DTYPE2_t c, eif

    for x in range(xmax):
        for y in range(ymax):
            if intfactor[x, y] == 0.:
                output[x, y] = start[x, y] + dt / 2. * (deriv2[x, y] - deriv1[x, y])
            else:
                c = intfactor[x, y]
                eif = exp(c * dt)
                output[x, y] = start[x, y] + (eif - c * dt - 1.) * (deriv2[x, y] - deriv1[x, y]) / (c * c * dt)


@cython.boundscheck(False)
@cython.wraparound(False)
def etd2rk2(np.ndarray[DTYPE1_t, ndim=2] start not None,
            np.ndarray[DTYPE1_t, ndim=2] output not None,
            np.ndarray[DTYPE1_t, ndim=2] deriv1 not None,
            np.ndarray[DTYPE1_t, ndim=2] deriv2 not None,
            np.ndarray[DTYPE2_t, ndim=2] intfactor not None,
            DTYPE2_t dt):
    """Second step of ETD2RK2."""

    assert start.dtype == DTYPE1
    assert output.dtype == DTYPE1
    assert deriv1.dtype == DTYPE1
    assert deriv2.dtype == DTYPE1
    assert intfactor.dtype == DTYPE2

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y
    cdef DTYPE2_t c, eif

    for x in range(xmax):
        for y in range(ymax):
            if intfactor[x, y] == 0.:
                output[x, y] = start[x, y] + dt * deriv2[x, y]
            else:
                c = intfactor[x, y]
                eif = exp(c * dt)
                output[x, y] = start[x, y] * eif + (((c * dt - 2.) * eif + c * dt + 2.) * deriv1[x, y] + (2 * (eif - c * dt - 1.)) * deriv2[x, y]) / (c * c * dt)
