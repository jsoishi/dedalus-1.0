import pylab as P
from init_cond import taylor_green
from fourier_data import FourierData

ux = FourierData((32,32))
uy = FourierData((32,32))
oo = P.ones((32,32))
ux, uy = taylor_green(ux,uy)

P.subplot(121)
P.imshow(oo*ux.k['x'])
P.colorbar()
P.subplot(122)
P.imshow(oo*ux.k['y'])
P.colorbar()
P.show()
