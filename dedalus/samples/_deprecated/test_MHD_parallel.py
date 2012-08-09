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

gshape = (32, 32, 32)
glength = (2 * na.pi, 2 * na.pi, 2 * na.pi)

shape, length = setup_parallel_objs(gshape, glength)
myproc = com_sys.myproc
RHS = MHD(shape, ParallelFourierRepresentation)
data = RHS.create_fields(0.)

k = (2, 0, 0)
p_vec = (0, 1, 0)
B0mag = 5.0
u1mag = 5e-6
alfven(data, k=k, B0mag=B0mag, u1mag=u1mag, p_vec=p_vec)

ti = RK2simple(RHS, CFL=0.4)
ti.stop_iter(51)
ti.stop_time(2.) # set stoptime
ti.stop_walltime(3600.) # stop after 1 hour
ti.set_nsnap(1)
ti.set_dtsnap(1e6)

an = AnalysisSet(data, ti)
an.add("field_snap", 10)
#an.add("phase_amp", 10, {'fclist': [('u', 'z'), ('B', 'z')], 'klist': [k]})
an.add("en_spec", 10)

# Main loop
cA = B0mag / na.sqrt(4 * na.pi * data.parameters['rho0'])
CFLtime = na.min(na.array(length) / na.array(shape)) / cA
dt = CFLtime / 10
print dt
#an.run()
while ti.ok:
    if myproc == 0:
        print "step: %i" %ti.iter
    ti.advance(data, dt)
    #an.run()

ti.final_stats()
