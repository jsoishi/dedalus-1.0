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
zeldovich(data, 1e-19)

Myr = 3.15e13 # 10^6 years in seconds
tstop = Myr*4e3

ti = RK2simple(RHS)
ti.stop_time(tstop)
ti.set_nsnap(1000)
ti.set_dtsnap(1e19)
dt=10*Myr

ddelta = []
uu = []
uk = []

while ti.ok:
    ti.advance(data, dt)
    
    if data.time % (tstop/20) == 0:
        tmp = na.zeros_like(data['u'][0]['xspace'][0,0,:].real)
        tmp[:] = data['u'][0]['xspace'][0,0,:].real

        tmp2 = na.zeros_like(data['delta']['xspace'][0,0,:].real)
        tmp2[:] = data['delta']['xspace'][0,0,:].real

        tmp3 = na.zeros_like(data['u'][0]['kspace'][0,0,:].real)
        tmp3[:] = data['u'][0]['kspace'][0,0,:].real

        uu.append(tmp)
        ddelta.append(tmp2)
        uk.append(tmp3)

def reorder(arr):
    di = len(arr)/2
    tmp = [arr[i-di] for i in xrange(len(arr))]
    return tmp 

pl.figure()
for i in xrange(len(ddelta)):
    pl.plot(reorder(ddelta[i]),hold=True)

pl.figure()
for i in xrange(len(uu)):
    pl.plot(reorder(uu[i]),hold=True)

pl.figure()
for i in xrange(len(uk)):
    pl.plot(reorder(uk[i][(len(uk[i])/2):]),hold=True)

pl.show()
