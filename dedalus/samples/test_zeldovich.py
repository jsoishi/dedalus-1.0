from dedalus.mods import *
import numpy as na
import pylab as pl

shape = (16,16,16)
RHS = CollisionlessCosmology(shape, FourierRepresentation)
data = RHS.create_fields(0.)
H_0 = 2.27826587e-18 # 70.3 km/s/Mpc in seconds^-1
ampl = 4e-19 # amplitude of initial velocity wave
RHS.parameters['Omega_r'] = 0#8.4e-5
RHS.parameters['Omega_m'] = 1#0.276
RHS.parameters['Omega_l'] = 0#0.724
RHS.parameters['H0'] = H_0
zeldovich(data, ampl)

Myr = 3.15e13 # 10^6 years in seconds
tstop = Myr*4e3

ti = RK2simple(RHS)
ti.stop_time(tstop)
ti.set_nsnap(1000)
ti.set_dtsnap(1e19)
dt = Myr*1e1

t_snapshots = []

ddelta = []
uu = []
uk = []

an = AnalysisSet(data, ti)
an.add("field_snap", 20)
an.add("en_spec", 20)

i = 0
#an.run()
while ti.ok:
    ti.advance(data, dt)
    print "step: ", i
    if data.time % (tstop/20) == 0:
        tmp = na.zeros_like(data['u'][0]['xspace'][0,0,:].real)
        tmp[:] = data['u'][0]['xspace'][0,0,:].real

        tmp2 = na.zeros_like(data['delta']['xspace'][0,0,:].real)
        tmp2[:] = data['delta']['xspace'][0,0,:].real

        tmp3 = na.zeros_like(data['u'][0]['kspace'][0,0,:].real)
        tmp3[:] = na.abs(data['u'][0]['kspace'][0,0,:])

        uu.append(tmp)
        ddelta.append(tmp2)
        uk.append(tmp3)
        t_snapshots.append(data.time)
    #an.run()
    i = i + 1

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

L = 64 # size of box
N_p1 = 64 # number of parallel sheets of particles
q = na.array([i for i in xrange(0,64)]) * L / N_p1
k = na.pi/32
a_i = 0.002 # initial scale factor
H_0 = 2.27826587e-18 # 70.3 km/s/Mpc in seconds^-1
t0 = (2./3.)/H_0 # present age of E-dS universe
ti = (a_i**(3./2.)) * t0 # time at which a = a_i in this universe

Ddot_i = (2./3) * ((1./t0)**(2./3)) * (ti**(-1./3))
A = ampl / a_i / Ddot_i
a_cross = 1 / (A * k)
tcross = (a_cross**(3./2.))*t0

pl.figure()
for t in t_snapshots:
    a = (t/t0)**(2./3)
    D = a
    x = q - D*A*na.sin(k*q)
    Ddot = (2./3) * ((1./t0)**(2./3)) * (t**(-1./3))
    delta = 1./(1.+D*A*k*na.cos(k*x))-1.
    v = a * Ddot * A * na.sin(k*x)
    pl.plot(v, hold=True)
pl.show()
