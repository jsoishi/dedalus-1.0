"""
This is a 2D gravity mode test. It sets up an exact eigenmode of the
gravity wave, and evolves it for two wave periods.

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2012 J. S. Oishi.  All Rights Reserved.

  This file is part of dedalus.

  dedalus is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from dedalus.mods import *
import numpy as np

decfg.set('analysis','slice_axis', 'y')

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
uz = 0.5 # the total amplitude is 1...keep in mind the hermitian symmetric pair
kx = 1.
kz = 1.
omega = np.sqrt(kx**2/(kx**2 + kz**2)) * RHS.parameters['beta']
tstop = 2*(2*np.pi/omega) # 2 mode periods

mode = data['u']['z'].find_mode(np.array([0,kz,kx]))
if mode:
    data['u']['z']['kspace'][tuple(mode)] = uz
    data['u']['x']['kspace'][tuple(mode)] = -kz/kx * uz
    data['T']['kspace'][tuple(mode)] = -1j*RHS.parameters['beta']/np.sqrt(kx**2/(kx**2 + kz**2)) * uz

# Integration parameters
ti = RK2mid(RHS)
ti.sim_stop_time = tstop # 2 wave periods
#ti.stop_iteration = 1
ti.wall_stop_time = 86400. # stop after 24 hours
ti.save_cadence = 1e7
ti.max_save_period = 2.

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
an.add(VolumeAverage(20, vs))
an.add(Snapshot(50))
an.add(TrackMode(10, modelist=[(0., 1., 1.)]))

# Main loop
#dt = 0.1
dt = 0.001
an.run()
my_proc = com_sys.myproc
while ti.ok:
    ti.advance(data, dt)
    an.run()

ti.finalize(data)
