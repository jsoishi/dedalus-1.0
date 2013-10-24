"""MIT vortex test ran with MHD

Authors: J. S. Oishi <jsoishi@gmail.com>
         K. J. Burns <kburns@berkeley.edu>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi, K. J. Burns.  All Rights Reserved.

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
from dedalus.funcs import insert_ipython
import numpy as np

vangle = np.pi / 3.
shape = (128, 128)
length = (1/np.cos(vangle/2.), 2.)
RHS = IncompressibleMHD(shape, FourierRepresentation, length=length)
data = RHS.create_fields(0.)

# Create grid parameters
y,x = data['B']['x'].xspace_grid()
x -= length[-1] / 2.
y -= length[-2] / 2.
r = np.sqrt(x ** 2 + y ** 2)
theta = np.arctan2(y, x)

# Setup B field
aux = data.clone()
aux.add_field('Az', 'ScalarField')

env = 1e-3 * np.cos(r * 4 * np.pi / length[-1]) ** 2
rc = length[-1] * 0.25 / 2
width = length[0]/shape[0] # dx...
aux['Az']['xspace'] = env * (1. + np.tanh((rc - r)/width))/2
data['B']['x']['kspace'] = aux['Az'].deriv('y')
data['B']['y']['kspace'] = -aux['Az'].deriv('x')
data['B'].div_free()

# Setup flow
V = 1.0
data['u']['x']['xspace'] += V * np.sin(vangle)
data['u']['y']['xspace'] += V * np.cos(vangle)

# Integration parameters
ti = RK2mid(RHS, CFL=0.4)
ti.stop_iteration = 1e6
ti.sim_stop_time = 4.
ti.save_cadence = 10000
ti.max_save_period = 100.

vs = VolumeAverageSet(data)
vs.add('ekin', '%17.10e')
vs.add('emag', '%17.10e')
vs.add('ux2', '%17.10e')
vs.add('uy2', '%17.10e')
vs.add('bx2', '%17.10e')
vs.add('by2', '%17.10e')
vs.add('divergence', '%17.10e')
vs.add('divergence_sum', '%17.10e')


an = AnalysisSet(data, ti)
an.add(Snapshot(50, space='xspace', axis='z', index='middle'))
an.add(Snapshot(50, space='kspace', axis='z', index=2))
an.add(VolumeAverage(10,vs))
#an.add("en_spec", 1)

# Main loop
CFLtime = np.min(np.array(length) / np.array(shape)) / V
dt = CFLtime / 10
print 'CFL time: ', CFLtime
print 'Chosen dt: ', dt

an.run()
while ti.ok:
    ti.advance(data, dt)
    an.run()

ti.final_stats()
