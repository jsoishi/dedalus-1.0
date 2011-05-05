import pylab as P
from init_cond import taylor_green
from fourier_data import FourierData

ux = FourierData((32,32))
uy = FourierData((32,32))

ux, uy = taylor_green(ux,uy)

P.subplot(121)
P.imshow(ux['kspace'].imag)
print "kspace min, max = (%10.5e,%10.5e)" % (ux['kspace'].real.min(), ux['kspace'].real.max())
P.subplot(122)
P.imshow(ux['xspace'].real)
P.show()
