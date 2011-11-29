import numpy as np
cimport numpy as np
cimport cython
DTYPE1 = np.complex128
ctypedef np.complex128_t DTYPE1_t

DTYPE2 = np.float64
ctypedef np.float64_t DTYPE2_t
@cython.boundscheck(False)
def dealias_23(np.ndarray[DTYPE1_t, ndim=2] data not None,
               np.ndarray[DTYPE2_t, ndim=2] kx not None,
               np.ndarray[DTYPE2_t, ndim=2] ky not None,
               np.ndarray[DTYPE2_t, ndim=1] knyquist not None):
    """Orszag 2/3 dealias rule"""

    assert data.dtype == DTYPE1
    assert kx.dtype == DTYPE2
    assert ky.dtype == DTYPE2
    
    cdef int xmax = data.shape[1]
    cdef int ymax = data.shape[0]
    cdef int x, y
    
    if kx.shape[0] == 1:
        for y in range(ymax):
            for x in range(xmax):
                if ((kx[0, x] > 2 / 3. * knyquist[1]) or (kx[0, x] < -2 / 3. * knyquist[1])) or \
                   ((ky[y, 0] > 2 / 3. * knyquist[0]) or (ky[y, 0] < -2 / 3. * knyquist[0])):
                    data[y, x] = 0.

    else:
        for y in range(ymax):
            for x in range(xmax):
                if ((kx[y, x] > 2 / 3. * knyquist[1]) or (kx[y, x] < -2 / 3. * knyquist[1])) or \
                   ((ky[y, 0] > 2 / 3. * knyquist[0]) or (ky[y, 0] < -2 / 3. * knyquist[0])):
                    data[y, x] = 0.
