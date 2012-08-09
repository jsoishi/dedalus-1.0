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
import numpy as na

shape = (128, 128)
length = (2., 2.)
RHS = MHD(shape, FourierRepresentation, length=length)
data = RHS.create_fields(0.)

# Create grid parameters
x,y = na.meshgrid(na.linspace(0, length[-1], shape[-1], endpoint=False),
                  na.linspace(0, length[-2], shape[-2], endpoint=False))
x -= length[-1] / 2.
y -= length[-2] / 2.
r = na.sqrt(x ** 2 + y ** 2)
theta = na.arctan2(y, x)

# Setup B field
aux = data.clone()
aux.add_field('Az', 'ScalarField')

env = 1e-3 * na.cos(r * 4 * na.pi / length[-1]) ** 2
rc = length[-1] * 0.25 / 2
width = 4*length[0]/shape[0] # 4 dx...
aux['Az']['xspace'] = env * (1. + na.tanh((rc - r)/width))
data['B']['x']['kspace'] = aux['Az'].deriv('y')
data['B']['y']['kspace'] = -aux['Az'].deriv('x')
data['B'].div_free()

# Setup flow
V = 1.0
vangle = na.pi / 3.
data['u']['x']['xspace'] += V * na.sin(vangle)
data['u']['y']['xspace'] += V * na.cos(vangle)

# Integration parameters
ti = RK2simple(RHS, CFL=0.4)
ti.stop_time(5.) # set stoptime
ti.stop_walltime(5*3600.) # stop after 1 hour

an = AnalysisSet(data, ti)
an.add("field_snap", 10)
#an.add("en_spec", 1)

# Main loop
CFLtime = na.min(na.array(length) / na.array(shape)) / V
dt = CFLtime / 10
print 'CFL time: ', CFLtime
print 'Chosen dt: ', dt

an.run()
while ti.ok:
    print "step: %i" %ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
