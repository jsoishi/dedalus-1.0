"""3d field advection test

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

shape = (32, 32, 32)
length = (2., 2., 2.)
RHS = MHD(shape, FourierRepresentation, length=length)
data = RHS.create_fields(0.)

# Create grid parameters
Bk = (na.random.rand(*shape) + 1j * na.random.rand(*shape)) - (0.5 + 0.5j)
Bk[5:-4, :, :] = 0.
Bk[:, 5:-4, :] = 0.
Bk[:, :, 5:-4] = 0.
Bx = na.fft.ifftn(Bk)
Bx.imag = 0.

# Setup B field
Bmag = 1e-3
Bx *= Bmag / na.max(na.abs(Bx))

data['B']['y']['xspace'] = Bx
data['B'].div_free()

# Setup flow
V = 1.0
theta = 0.
phi = 2 * na.pi / 3.
data['u']['x']['xspace'] += V * na.cos(theta) * na.cos(phi)
data['u']['y']['xspace'] += V * na.cos(theta) * na.sin(phi)
data['u']['z']['xspace'] += V * na.sin(theta)

# Integration parameters
ti = RK2simple(RHS, CFL=0.4)
ti.stop_time(10.) # set stoptime
ti.stop_walltime(3600.) # stop after 1 hour
ti.set_nsnap(1e6)
ti.set_dtsnap(1e6)

an = AnalysisSet(data, ti)
an.add("field_snap", 20)
an.add("field_snap", 20, {'space': 'kspace'})
an.add("en_spec", 20, {'flist': ['u', 'B']})

# Main loop
CFLtime = na.min(na.array(length) / na.array(shape)) / V
dt = CFLtime / 20
print 'CFL time: ', CFLtime
print 'Chosen dt: ', dt

an.run()
while ti.ok:
    print "step: %i" %ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
