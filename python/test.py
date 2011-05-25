import pylab as P
import numpy.fft as fpack
from init_cond import taylor_green, sin_x
from fourier_data import FourierData

ux = FourierData((32,32))
uy = FourierData((32,32))
duxdy = FourierData((32,32))
#ux, uy = taylor_green(ux,uy)
ux = sin_x(ux)
duxdy.data[:,:] = ux.deriv('x')

blah = fpack.fftn(duxdy.data)

print duxdy._curr_space
P.subplot(121)
P.imshow(duxdy['xspace'].real)
print duxdy._curr_space
#P.imshow(blah.real)
print "kspace min, max = (%10.5e,%10.5e)" % (ux['kspace'].real.min(), ux['kspace'].real.max())
P.subplot(122)
#P.imshow(duxdy['xspace'].real)
P.imshow(ux['xspace'].real)
P.show()
