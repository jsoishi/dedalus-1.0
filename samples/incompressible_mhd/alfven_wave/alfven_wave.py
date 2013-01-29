"""
3D Alfven wave test.

"""

from dedalus.mods import *
import numpy as np

# Physics
shape = (32, 32, 32)
RHS = IncompressibleMHD(shape, FourierRepresentation)

# Initial conditions
data = RHS.create_fields(0.)
rho0 = RHS.parameters['rho0']

#   Background magnetic field
B0mag = 1.
B0 = B0mag * np.array([1., 0., 0.])

index = data['B']['z'].find_mode([0, 0, 0])
if index:
    data['B']['z']['kspace'][index] += B0[0]
    data['B']['y']['kspace'][index] += B0[1]
    data['B']['x']['kspace'][index] += B0[2]

#   Perturbation
k = np.array([3., 2., 1.])
u1mag = 1e-3
u1 = u1mag * np.array([0., -1., 2.]) / np.sqrt(5)
vA = np.sqrt(B0mag ** 2 / (4 * np.pi * rho0))
omega = vA * np.dot(k, B0) / B0mag
B1 = -np.dot(k, B0) * u1 / omega

ksim = swap_indices(k)
index = data['B']['y'].find_mode(ksim)
if index:
    data['B']['z']['kspace'][index] += B1[0] / 2.
    data['B']['y']['kspace'][index] += B1[1] / 2.
    data['B']['x']['kspace'][index] += B1[2] / 2.

    data['u']['z']['kspace'][index] += u1[0] / 2.
    data['u']['y']['kspace'][index] += u1[1] / 2.
    data['u']['x']['kspace'][index] += u1[2] / 2.

index = data['B']['y'].find_mode(-ksim)
if index:
    data['B']['z']['kspace'][index] += B1[0] / 2.
    data['B']['y']['kspace'][index] += B1[1] / 2.
    data['B']['x']['kspace'][index] += B1[2] / 2.

    data['u']['z']['kspace'][index] += u1[0] / 2.
    data['u']['y']['kspace'][index] += u1[1] / 2.
    data['u']['x']['kspace'][index] += u1[2] / 2.

# Integration
ti = RK2mid(RHS,CFL=0.4)
ti.stop_iteration = 1e6
ti.sim_stop_time = 2 * np.pi / omega
ti.save_cadence = 10000
ti.max_save_period = 100.

# Analysis
an = AnalysisSet(data, ti)
an.add(Snapshot(10, space='xspace', axis='y', index='middle'))
an.add(Snapshot(10, space='kspace', axis='y', index=2))
an.add(TrackMode(10, modelist=[ksim]))

# Timestep
vA_cfl_time = np.min(2 * np.pi / np.array(shape)) / vA
cfl_times = np.array([vA_cfl_time])
dt = cfl_times.min() / 20
print 'CFL times: ', cfl_times
print 'Chosen dt: ', dt
print '-' * 10

# Main loop
an.run()
while ti.ok:
    ti.advance(data, dt)
    an.run()

an.cleanup()
ti.final_stats()

