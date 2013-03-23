

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


def euler(np.ndarray[nc128_t, ndim=3] start,
          np.ndarray[nc128_t, ndim=3] output,
          np.ndarray[nc128_t, ndim=3] deriv,
          nf64_t dt):
    """First-order forward-Euler step."""

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                output[y, z, x] = start[y, z, x] + dt * deriv[y, z, x]


def etd1(np.ndarray[nc128_t, ndim=3] start,
         np.ndarray[nc128_t, ndim=3] output,
         np.ndarray[nc128_t, ndim=3] deriv,
         np.ndarray[nf64_t, ndim=3] intfactor,
         nf64_t dt):
    """First-order ETD step."""

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z
    cdef nf64_t Z, f0, f1

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                Z = intfactor[y, z, x] * dt
                if Z == 0.:
                    output[y, z, x] = start[y, z, x] + dt * deriv[y, z, x]
                else:
                    if ((Z < 1.) and (Z > -1.)):
                        f0 = f_taylor(0, Z)
                        f1 = f_taylor(1, Z)
                    else:
                        f0 = exp(Z)
                        f1 = (f0 - 1.) / Z
                    output[y, z, x] = start[y, z, x] * f0 + deriv[y, z, x] * f1 * dt


def etd2rk1(np.ndarray[nc128_t, ndim=3] start,
            np.ndarray[nc128_t, ndim=3] output,
            np.ndarray[nc128_t, ndim=3] deriv1,
            np.ndarray[nc128_t, ndim=3] deriv2,
            np.ndarray[nf64_t, ndim=3] intfactor,
            nf64_t dt):
    """Second step of ETD2RK1."""

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z
    cdef nf64_t Z, f0, f1, f2

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                Z = intfactor[y, z, x] * dt
                if Z == 0.:
                    output[y, z, x] = start[y, z, x] + dt / 2. * (deriv2[y, z, x] - deriv1[y, z, x])
                else:
                    if ((Z < 1.) and (Z > -1.)):
                        f0 = f_taylor(0, Z)
                        f1 = f_taylor(1, Z)
                        f2 = f_taylor(2, Z)
                    else:
                        f0 = exp(Z)
                        f1 = (f0 - 1.) / Z
                        f2 = (f1 - 1.) / Z
                    output[y, z, x] = start[y, z, x] + (deriv2[y, z, x] - deriv1[y, z, x]) * f2 * dt


def etd2rk2(np.ndarray[nc128_t, ndim=3] start,
            np.ndarray[nc128_t, ndim=3] output,
            np.ndarray[nc128_t, ndim=3] deriv1,
            np.ndarray[nc128_t, ndim=3] deriv2,
            np.ndarray[nf64_t, ndim=3] intfactor,
            nf64_t dt):
    """Second step of ETD2RK2."""

    cdef int xmax = start.shape[2]
    cdef int ymax = start.shape[0]
    cdef int zmax = start.shape[1]
    cdef unsigned int x, y, z
    cdef nf64_t Z, f0, f1, f2

    for y in range(ymax):
        for z in range(zmax):
            for x in range(xmax):
                Z = intfactor[y, z, x] * dt
                if Z == 0.:
                    output[y, z, x] = start[y, z, x] + dt * deriv2[y, z, x]
                else:
                    if ((Z < 1.) and (Z > -1.)):
                        f0 = f_taylor(0, Z)
                        f1 = f_taylor(1, Z)
                        f2 = f_taylor(2, Z)
                    else:
                        f0 = exp(Z)
                        f1 = (f0 - 1.) / Z
                        f2 = (f1 - 1.) / Z
                    output[y, z, x] = start[y, z, x] * f0 + (deriv2[y, z, x] - deriv1[y, z, x]) * 2. * f2 * dt + deriv1[y, z, x] * f1 * dt


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
    cdef int m = 20
    cdef nf64_t factj = 1.
    cdef nf64_t sum = 0.

    if k > 2:
        for j in range(1, k):
            factj *= j

    for j in range(k, m):
        if j > 1:
            factj *= j
        sum += Z ** (j - k) / factj

    return sum

