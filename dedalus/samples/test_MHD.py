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

shape = (32,32,32)
RHS = MHD(shape, FourierRepresentation)
data = RHS.create_fields(0.)

k = (1, -1, 0)
alfven(data, k=k)

ti = RK2simple(RHS, CFL=0.4)
ti.stop_time(50.) # set stoptime
ti.stop_walltime(3600.) # stop after 1 hour

an = AnalysisSet(data, ti)
an.add("field_snap", 10)
an.add("phase_amp", 10, {'fclist': [('u', 'z'), ('B', 'z')], 'klist': [k]})
an.add("en_spec", 10)

# Main loop
dt = 0.05
an.run()
while ti.ok:
    print "step: %i" %ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
