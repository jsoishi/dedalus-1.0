from dedalus.mods import *

shape = (10,10,10)
RHS = LinearCollisionlessCosmology(shape, FourierData)
data = RHS.create_fields(0.)
RHS.parameters['Omega_r'] = 8.4e-5
RHS.parameters['Omega_m'] = 0.276
RHS.parameters['Omega_l'] = 0.724
RHS.parameters['H0'] = 3.24077649e-18 # 100 km/s/Mpc in seconds
ti = RK2simple(RHS)

ti.stop_time(3.15e7*1e9)
ti.set_nsnap(1000)
ti.set_dtsnap(1e17)
dt=3.15e7*1e6
outfile = open('scale_factor.dat','w')
while ti.ok:
    ti.advance(data, dt)
    outfile.write("%10.5e\t%10.5e\n" %(ti.time, ti.RHS.aux_eqns['H'].value))

outfile.close()


