import time
import numpy as np
import pylab as P
from fftw import Plan

size = 128
data = np.zeros((size,size,size), dtype='complex')

fp = Plan(data)
rp = Plan(data, direction='FFTW_BACKWARD', flags=['FFTW_MEASURE'])
x = np.linspace(0,2*np.pi,size, endpoint=False)
y = np.linspace(0,2*np.pi,size, endpoint=False)
xx, yy = np.meshgrid(x,y)
data[:] = (np.sin(xx) + np.sin(yy) + 0j) * np.ones((1,1,size))
data_blah = data.copy()

start = time.time()
fp()
rp()
end = time.time()
print "ratio = %10.5f" % (data[10,10,10].real/data_blah[10,10,10].real)
print "128**3 = %10.5f" % 128**3

# P.subplot(121)
# P.imshow(data_blah[:,:,0].real)
# P.subplot(122)
# P.imshow(data[:,:,0].real)
# P.show()

print "time = %10.5f" % (end-start)
