import time
import numpy as np
import pylab as P
from fftw import PlanPlanes

size = 128
data = np.zeros((size,size,size), dtype='complex')

p_start = time.time()
p = PlanPlanes(data)
p_stop = time.time()

data[:] = np.random.rand(size, size, size) + 0j
data2 = data.copy()

fw_start = time.time()
p()
fw_stop = time.time()

np_start = time.time()
npdata = np.fft.fftn(data2,axes=(0,1))
np_stop = time.time()


np.testing.assert_almost_equal(data, npdata)

print "numpy time = %10.5f sec" % (np_stop-np_start)
print "fftw time = %10.5f sec" % (fw_stop-fw_start)
print "plan time = %10.5f sec" % (p_stop-p_start)
