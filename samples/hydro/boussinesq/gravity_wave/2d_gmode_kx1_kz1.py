from dedalus.mods import *
from dedalus.funcs import insert_ipython
import numpy as np

decfg.set('analysis','slice_axis', '1')

shape = (48, 2, 48)
RHS = BoussinesqHydro(shape, FourierRepresentation)
data = RHS.create_fields(0.)

# set up parameters
RHS.parameters['g'] = 1.
RHS.parameters['alpha_t'] = 1.
RHS.parameters['beta'] = 1. # N = alpha_t * g * beta = 1

RHS.parameters['nu']    = 0.
RHS.parameters['kappa'] = 0.

# ICs
uz = 1.
kx = 1.
kz = 1.

mode = data['u']['z'].find_mode(np.array([0,kz,kx]))
if mode:
    data['u']['z']['kspace'][tuple(mode)] = uz
    data['u']['x']['kspace'][tuple(mode)] = -kz/kx * uz
    data['T']['kspace'][tuple(mode)] = -1j*RHS.parameters['beta']/np.sqrt(kx**2/(kx**2 + kz**2)) * uz
    
# Integration parameters
ti = RK2simplevisc(RHS)
ti.stop_time(2.) # 2 wave periods
#ti.stop_iter(1)
ti.stop_walltime(86400.) # stop after 24 hours
ti.set_nsnap(1e7)
#ti.set_nsnap(100)
ti.set_dtsnap(2.)

vs = VolumeAverageSet(data)
vs.add('ekin', '%10.5e')
vs.add('energy_dissipation', '%10.5e')
vs.add('thermal_energy_dissipation', '%10.5e')
vs.add('ux2', '%10.5e')
vs.add('uy2', '%10.5e')
vs.add('uz2', '%10.5e')
vs.add('temp2', '%10.5e')
vs.add('divergence', '%10.5e')
vs.add('divergence_sum', '%10.5e')

an = AnalysisSet(data, ti)
#an.add(VolumeAverage(20,va_obj=vs))
an.add('volume_average', 20, {'va_obj': vs})
an.add("field_snap", 50)
an.add("mode_track", 10, {'flist': ['u','T'],
                        'klist': [(0, 1, 1)]})
# Main loop
#dt = 0.1
dt = 0.001
an.run()
my_proc = com_sys.myproc
while ti.ok:
    ti.advance(data, dt)
    an.run()

ti.finalize(data)
