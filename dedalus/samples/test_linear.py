from dedalus.mods import *
import numpy as na
from dedalus.data_objects import hermitianize
shape = (10,10,10)
RHS = LinearCollisionlessCosmology(shape, FourierData)
data = RHS.create_fields(0.)
RHS.parameters['Omega_r'] = 0#8.4e-5
RHS.parameters['Omega_m'] = 1#0.276
RHS.parameters['Omega_l'] = 0#0.724
RHS.parameters['H0'] = 3.24077649e-18 # 100 km/s/Mpc in seconds
for f in data.fields:
    data[f] = na.random.normal(0,1,data[f]['kspace'].shape) + 1j*na.random.normal(0,1,data[f]['kspace'].shape) # garbage IC
    hermitianize.enforce_hermitian(data[f]['kspace'])
    hermitianize.zero_nyquist(data[f]['kspace'])
ti = RK2simple(RHS)
ti.stop_time(3.15e7*1e9)
ti.set_nsnap(1000)
ti.set_dtsnap(1e17)
dt=3.15e7*1e6

outfile = open('growth.dat','w')

delta_init = na.zeros_like(data['delta']['kspace'])
ti.advance(data, dt)
delta_init[:] = data['delta']['kspace']

while ti.ok:
    ti.advance(data, dt)
    delta = data['delta']['kspace'][1,1,1].real
    outfile.write("%10.5e\t%10.5e\t%10.5e\n" %(ti.time, ti.RHS.aux_eqns['a'].value, delta))
delta_final = data['delta']['kspace']

print delta_final/delta_init

outfile.close()


