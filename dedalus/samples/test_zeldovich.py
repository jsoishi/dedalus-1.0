from dedalus.mods import *
import numpy as na
import pylab as pl

shape = (16,16,16)
RHS = CollisionlessCosmology(shape, FourierRepresentation)
data = RHS.create_fields(0.)
H_0 = 7.185e-5 # 70.3 km/s/Mpc in Myr^-1 (2.27826587e-18 seconds^-1) 
ampl = 5e-5 # amplitude of initial velocity wave

L = 2*na.pi # size of box
N_p = 128 # resolution of analytic solution
q = na.array([i for i in xrange(0,N_p)]) * L / N_p
k = 2*na.pi/L # wavenumber of initial velocity perturbation
a_i = 0.002 # initial scale factor
t0 = (2./3.)/H_0 # present age of E-dS universe
t_init = (a_i**(3./2.)) * t0 # time at which a = a_i in this universe

Ddot_i = (2./3.) * ((1./t0)**(2./3.)) * (t_init**(-1./3.)) / a_i
A = ampl / a_i / Ddot_i
a_cross = a_i / (A * k)
print "a_cross = ", a_cross
tcross = (a_cross**(3./2.))*t0

RHS.parameters['Omega_r'] = 0#8.4e-5
RHS.parameters['Omega_m'] = 1#0.276
RHS.parameters['Omega_l'] = 0#0.724
RHS.parameters['H0'] = H_0
zeldovich(data, ampl, a_i, a_cross)

Myr = 1 # 3.15e13 seconds
tstop = tcross - t_init
dt = Myr

ti = RK2simple(RHS)
ti.stop_time(tstop)
ti.set_nsnap(1000)
ti.set_dtsnap(1e19)

ddelta = []
uu = []
uk = []

t_snapshots = []
a_snapshots = []
  
an = AnalysisSet(data, ti)
an.add("field_snap", 20)
an.add("en_spec", 20)

i = 0
#an.run()
while ti.ok:
    print "step: ", i, " a = ", RHS.aux_eqns['a'].value
    if i % 80 == 0:
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
        a_snapshots.append(RHS.aux_eqns['a'].value)
    ti.advance(data, dt)
    #an.run()
    i = i + 1

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
a_snapshots.append(RHS.aux_eqns['a'].value)
print "a_stop = ", RHS.aux_eqns['a'].value

def reorder(arr):
    di = len(arr)/2
    tmp = [arr[i-di] for i in xrange(len(arr))]
    return tmp 

xx = [L*i/len(uu[0]) for i in xrange(len(uu[0]))]
pl.figure()
for delta in ddelta:
    pl.plot(xx, reorder(delta),hold=True)

pl.figure()
for u in uu:
    pl.plot(xx, reorder(u),hold=True)

pl.figure()
for u in uk:
    pl.plot(reorder(u)[(len(u)/2):],hold=True)

for i,a in enumerate(a_snapshots):
    t = a**(3./2.) * t0
    D = a / a_i
    x = q + D*A*na.sin(k*q)
    Ddot = (2./3.) * ((1./t0)**(2./3.)) * (t**(-1./3.)) / a_i
    #delta = 1./(1.+D*A*k*na.cos(k*q))-1.
    v = a * Ddot * A * na.sin(k*q)
    #phi = 3/2/a * ( (q**2 - x**2)/2 + 
    #                D * A * k * (k*q*na.sin(k*q) + na.cos(k*q) - 1) )
    pl.figure()
    pl.plot(x, v)
    pl.plot(xx, reorder(uu[i]), '.', hold=True)
    pl.title(a)

pl.show()
