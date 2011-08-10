import numpy as np
cimport numpy as np

DTYPE1 = np.complex128
ctypedef np.complex128_t DTYPE1_t

DTYPE2 = np.float64
ctypedef np.float64_t DTYPE2_t

def dealias_23(np.ndarray[DTYPE1_t, ndim=3] data not None,
               np.ndarray[DTYPE2_t, ndim=3] kx not None,
               np.ndarray[DTYPE2_t, ndim=3] ky not None,
               np.ndarray[DTYPE2_t, ndim=3] kz not None,
               np.ndarray[DTYPE2_t, ndim=1] knyquist not None):
    """Orszag 2/3 dealias rule"""

    assert data.dtype == DTYPE1
    assert kx.dtype == DTYPE2
    assert ky.dtype == DTYPE2
    assert ky.dtype == DTYPE2
    
    cdef int xmax = data.shape[2]
    cdef int ymax = data.shape[1]
    cdef int zmax = data.shape[0]
    cdef int x, y, z
    
    if kx.shape[1] == 1:
        for z in range(zmax):
            for y in range(ymax):
                for x in range(xmax):
                    if ((kx[0, 0, x] >= 2 / 3. * knyquist[2]) or (kx[0, 0, x] <= -2 / 3. * knyquist[2])) or \
                       ((ky[0, y, 0] >= 2 / 3. * knyquist[1]) or (ky[0, y, 0] <= -2 / 3. * knyquist[1])) or \
                       ((kz[z, 0, 0] >= 2 / 3. * knyquist[0]) or (kz[z, 0, 0] <= -2 / 3. * knyquist[0])):
                        data[z, y, x] = 0.

    else:
        for z in range(zmax):
            for y in range(ymax):
                for x in range(xmax):
                    if ((kx[0, y, x] >= 2 / 3. * knyquist[2]) or (kx[0, y, x] <= -2 / 3. * knyquist[2])) or \
                       ((ky[0, y, 0] >= 2 / 3. * knyquist[1]) or (ky[0, y, 0] <= -2 / 3. * knyquist[1])) or \
                       ((kz[z, 0, 0] >= 2 / 3. * knyquist[0]) or (kz[z, 0, 0] <= -2 / 3. * knyquist[0])):
                        data[z, y, x] = 0.