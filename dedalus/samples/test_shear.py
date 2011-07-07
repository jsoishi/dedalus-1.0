import pylab as P
from dedalus.mods import *
from dedalus.data_objects.api import FourierShearData

f = FourierShearData((128,128),-1)

sin_y(f)

P.subplot(121)
P.imshow(f['xspace',5.].real)
P.colorbar()
P.subplot(122)
P.imshow(f['xspace',0.5].real)
P.colorbar()
P.show()

