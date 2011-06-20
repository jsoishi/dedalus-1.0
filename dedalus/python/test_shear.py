import pylab as P
from init_cond import sin_y
from fourier_data import FourierShearData

f = FourierShearData((128,128),-1)

f = sin_y(f)

P.subplot(121)
P.imshow(f['xspace',5.].real)
P.colorbar()
P.subplot(122)
P.imshow(f['xspace',0.5].real)
P.colorbar()
P.show()

