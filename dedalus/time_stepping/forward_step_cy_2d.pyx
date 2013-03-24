

#cython: boundscheck=False
#cython: wraparound=False
#cython: nonecheck=False
#cython: cdivision=True


import numpy as np
cimport numpy as np
from libc.math cimport exp


nc128 = np.complex128
ctypedef np.complex128_t nc128_t

nf64 = np.float64
ctypedef np.float64_t nf64_t


def euler(np.ndarray[nc128_t, ndim=2] start,
          np.ndarray[nc128_t, ndim=2] output,
          np.ndarray[nc128_t, ndim=2] deriv,
          nf64_t dt):
    """First-order forward-Euler step."""

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y

    for x in range(xmax):
        for y in range(ymax):
            output[x, y] = start[x, y] + dt * deriv[x, y]


def etd1(np.ndarray[nc128_t, ndim=2] start,
         np.ndarray[nc128_t, ndim=2] output,
         np.ndarray[nc128_t, ndim=2] deriv,
         np.ndarray[nf64_t, ndim=2] intfactor,
         nf64_t dt):
    """First-order ETD step."""

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y
    cdef nf64_t Z, f0, f1

    for x in range(xmax):
        for y in range(ymax):
            Z = intfactor[x, y] * dt
            if Z == 0.:
                output[x, y] = start[x, y] + dt * deriv[x, y]
            else:
                if ((Z < 0.5) and (Z > -0.5)):
                    f0 = f_taylor(0, Z)
                    f1 = f_taylor(1, Z)
                else:
                    f0 = exp(Z)
                    f1 = (f0 - 1.) / Z
                output[x, y] = start[x, y] * f0 + deriv[x, y] * f1 * dt


def etd2rk1(np.ndarray[nc128_t, ndim=2] start,
            np.ndarray[nc128_t, ndim=2] output,
            np.ndarray[nc128_t, ndim=2] deriv1,
            np.ndarray[nc128_t, ndim=2] deriv2,
            np.ndarray[nf64_t, ndim=2] intfactor,
            nf64_t dt):
    """Second step of ETD2RK1."""

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y
    cdef nf64_t Z, f0, f1, f2

    for x in range(xmax):
        for y in range(ymax):
            Z = intfactor[x, y] * dt
            if Z == 0.:
                output[x, y] = start[x, y] + dt / 2. * (deriv2[x, y] - deriv1[x, y])
            else:
                if ((Z < 0.5) and (Z > -0.5)):
                    f0 = f_taylor(0, Z)
                    f1 = f_taylor(1, Z)
                    f2 = f_taylor(2, Z)
                else:
                    f0 = exp(Z)
                    f1 = (f0 - 1.) / Z
                    f2 = (f1 - 1.) / Z
                output[x, y] = start[x, y] + (deriv2[x, y] - deriv1[x, y]) * f2 * dt


def etd2rk2(np.ndarray[nc128_t, ndim=2] start,
            np.ndarray[nc128_t, ndim=2] output,
            np.ndarray[nc128_t, ndim=2] deriv1,
            np.ndarray[nc128_t, ndim=2] deriv2,
            np.ndarray[nf64_t, ndim=2] intfactor,
            nf64_t dt):
    """Second step of ETD2RK2."""

    cdef int xmax = start.shape[0]
    cdef int ymax = start.shape[1]
    cdef unsigned int x, y
    cdef nf64_t Z, f0, f1, f2

    for x in range(xmax):
        for y in range(ymax):
            Z = intfactor[x, y] * dt
            if Z == 0.:
                output[x, y] = start[x, y] + dt * deriv2[x, y]
            else:
                if ((Z < 0.5) and (Z > -0.5)):
                    f0 = f_taylor(0, Z)
                    f1 = f_taylor(1, Z)
                    f2 = f_taylor(2, Z)
                else:
                    f0 = exp(Z)
                    f1 = (f0 - 1.) / Z
                    f2 = (f1 - 1.) / Z
                output[x, y] = start[x, y] * f0 + (deriv2[x, y] - deriv1[x, y]) * 2. * f2 * dt + deriv1[x, y] * f1 * dt


cdef nf64_t f_taylor(int k, nf64_t Z):
    """
    Truncated Taylor series approximations to f.

    Notes
    -----
         m
    f = sum Z ** (j - k) / j!
        j=k

    """

    cdef int j
    cdef nf64_t factj = 1.
    cdef nf64_t sum = 0.

    if k > 2:
        for j in range(1, k):
            factj *= j

    for j in range(k, 15):
        if j > 1:
            factj *= j
        sum += Z ** (j - k) / factj

    return sum

