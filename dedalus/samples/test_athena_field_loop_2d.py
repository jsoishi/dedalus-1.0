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

shape = (64, 64)
length = (2. , 2.)
RHS = MHD(shape, FourierRepresentation, length=length)
data = RHS.create_fields(0.)

# Create grid parameters
x,y = na.meshgrid(na.linspace(0, length[-1], shape[-1], endpoint=False),
                  na.linspace(0, length[-2], shape[-2], endpoint=False))
x -= length[-1] / 2.
y -= length[-2] / 2.
r = na.sqrt(x ** 2 + y ** 2)
theta = na.arctan2(y, x)

# Create vector potential
# A = data._field_classes['scalar'](data)
# Amag = 1.0e-3
# R0 = 0.3
# A['xspace'] = Amag * (R0 - r)
# A['xspace'][A['xspace'] < 0.] = 0.
#
#RHS.curlX(A, data['B'])

env = na.sin(r * 8 * na.pi / length[-1]) ** 2
data['B']['x']['xspace'] = -env * na.sin(theta)
data['B']['y']['xspace'] = env * na.cos(theta)

data['B']['x']['xspace'][r < length[-1] * 0.25 / 2] = 0
data['B']['y']['xspace'][r < length[-2] * 0.25 / 2] = 0
data['B']['x']['xspace'][r > length[-1] * 0.5 / 2] = 0
data['B']['y']['xspace'][r > length[-2] * 0.5 / 2] = 0

data['B'].div_free()

# Setup flow
V = length[-1] / 2.
vangle = na.pi / 2.
data['u']['x']['xspace'] += V * na.sin(vangle)
data['u']['y']['xspace'] += V * na.cos(vangle)

# Integration parameters
ti = RK2simple(RHS, CFL=0.4)
ti.stop_time(5.) # set stoptime
ti.stop_walltime(3600.) # stop after 1 hour

an = AnalysisSet(data, ti)
an.add("field_snap", 1)
an.add("en_spec", 1)

# Main loop
CFLtime = na.min(na.array(length) / na.array(shape)) / V
dt = 0.005
print 'CFL time: ', CFLtime
print 'Chosen dt: ', dt

an.run()
while ti.ok:
    print "step: %i" %ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
