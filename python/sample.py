"""Simple 2D Taylor-Green vortex. Useful for timing tests (the
C-vs-python shootout).

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi.  All Rights Reserved.

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

from physics import Hydro
from fourier_data import FourierData
from time_step import RK2simple,RK2simplevisc
from init_cond import taylor_green

shape = (100,100) 
RHS = Hydro(shape, FourierData)
RHS.parameters['nu'] = 0.1
data = RHS.create_fields(0.)

taylor_green(data['ux'],data['uy'])
ti = RK2simplevisc(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
ti.stop_iter(100) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours

#main loop
dt = 1e-3
while ti.ok:
    #print "step: %i" % ti.iter
    ti.advance(data, dt)
ti.final_stats()
