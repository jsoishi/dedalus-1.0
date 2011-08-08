import numpy as np
cimport numpy as np

DTYPE = np.complex128
ctypedef np.complex128_t DTYPE_t

def dealias_23(np.ndarray[DTYPE_t, ndim=2] data not None):
    """Orszag 2/3 dealias rule"""

    assert data.dtype == DTYPE
    cdef int xmax = data.shape[1]
    cdef int ymax = data.shape[0]
    cdef int x, y
    for y in range(ymax):
        for x in range(xmax):
            if (x > 1/3. * xmax and x < 2/3. * xmax) or (y > 1/3. * ymax and y < 2/3. * ymax):
                data[y,x] = 0.
