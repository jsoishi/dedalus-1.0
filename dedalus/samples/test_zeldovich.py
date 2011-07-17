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
zeldovich(data, 1e-18)

ti = RK2simple(RHS)
ti.stop_time(3.15e7*1e9)
ti.set_nsnap(1000)
ti.set_dtsnap(1e17)
dt=3.15e7*1e6/16
i = 0

ddelta = []
uu = []
uk= []
deltak = []

while ti.ok:
    print i
    ti.advance(data, dt)
    
    if i % 100 == 0:
        tmp = na.zeros_like(data['u'][2]['xspace'][:,0,0].real)
        tmp[:] = data['u'][2]['xspace'][:,0,0].real

        tmp2 = na.zeros_like(data['delta']['xspace'][:,0,0].real)
        tmp2[:] = data['delta']['xspace'][:,0,0].real

        tmp3 = na.zeros_like(data['u'][2]['kspace'][:,0,0])
        tmp3[:] = na.abs(data['u'][2]['kspace'][:,0,0])

        tmp4 = na.zeros_like(data['delta']['kspace'][:,0,0])
        tmp4[:] = na.abs(data['delta']['kspace'][:,0,0])

        uu.append(tmp)
        ddelta.append(tmp2)
        uk.append(tmp3)
        deltak.append(tmp4)
    i += 1

def reorder(arr):
    di = len(arr)/2
    tmp = [arr[i-di] for i in xrange(len(arr))]
    for i in xrange(len(arr)):
        arr[i] = tmp[i]
    return arr

for i in xrange(len(ddelta)):
    pl.plot(reorder(ddelta[i]),hold=True)

pl.figure()
for i in xrange(len(uu)):
    pl.plot(reorder(uu[i]),hold=True)

pl.figure()
for i in xrange(len(deltak)):
    pl.plot(reorder(deltak[i]),hold=True)

pl.figure()
for i in xrange(len(uk)):
    pl.plot(reorder(uk[i]),hold=True)

pl.show()


