import numpy as np
cimport numpy as np
cimport cython

DTYPE1 = np.complex128
ctypedef np.complex128_t DTYPE1_t

DTYPE2 = np.float64
ctypedef np.float64_t DTYPE2_t

@cython.boundscheck(False)
@cython.wraparound(False)
def dealias_23(np.ndarray[DTYPE1_t, ndim=2] data not None,
               np.ndarray[DTYPE2_t, ndim=2] kx not None,
               np.ndarray[DTYPE2_t, ndim=2] ky not None,
               np.ndarray[DTYPE2_t, ndim=1] knyquist not None):
    """Orszag 2/3 dealias rule."""

    assert data.dtype == DTYPE1
    assert kx.dtype == DTYPE2
    assert ky.dtype == DTYPE2

    cdef int xmax = data.shape[0]
    cdef int ymax = data.shape[1]
    cdef unsigned int x, y

    if ky.shape[0] == 1:
        for x in range(xmax):
            for y in range(ymax):
                if ((kx[x, 0] >= 2. / 3. * knyquist[0]) or (kx[x, 0] <= -2. / 3. * knyquist[0])) or \
                   ((ky[0, y] >= 2. / 3. * knyquist[1]) or (ky[0, y] <= -2. / 3. * knyquist[1])):
                    data[x, y] = 0.

    else:
        for x in range(xmax):
            for y in range(ymax):
                if ((kx[x, 0] >= 2. / 3. * knyquist[0]) or (kx[x, 0] <= -2. / 3. * knyquist[0])) or \
                   ((ky[x, y] >= 2. / 3. * knyquist[1]) or (ky[x, y] <= -2. / 3. * knyquist[1])):
                    data[x, y] = 0.
