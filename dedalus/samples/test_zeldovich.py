from dedalus.mods import *
import numpy as na

import pylab as pl

shape = (16,16,16)
RHS = CollisionlessCosmology(shape, FourierRepresentation)
data = RHS.create_fields(0.)
RHS.parameters['Omega_r'] = 0#8.4e-5
RHS.parameters['Omega_m'] = 1#0.276
RHS.parameters['Omega_l'] = 0#0.724
RHS.parameters['H0'] = 2.27826587e-18 # 70.3 km/s/Mpc in seconds^-1
zeldovich(data)

ti = RK2simple(RHS)
ti.stop_time(3.15e7*1e9)
ti.set_nsnap(1000)
ti.set_dtsnap(1e17)
dt=3.15e7*1e6

i = 0
while ti.ok:
    print i
    ti.advance(data, dt)

    if i % 200 == 0: 
        pass
        #print data['u'][2]['kspace']
        pl.figure()
        pl.plot(data['u'][2]['xspace'][0,0,:].real)
        pl.show()
    i += 1
