"""Lithwick 2007 shearing wave test

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
import numpy as na
from dedalus.mods import *
from dedalus.funcs import insert_ipython

shape = (10,30)
RHS = ShearHydro(shape, FourierShearRepresentation, length=[2*na.pi*100,2*na.pi])

RHS.parameters['Omega'] = 1.
RHS.parameters['S'] = -1.5
data = RHS.create_fields(0.)

shearing_wave(data,0.01,[-1,10])

ti = RK2simple(RHS,CFL=0.4)
ti.stop_iter(1000000)
ti.stop_time(1000.)
ti.set_nsnap(10000)
ti.set_dtsnap(100.)
an = AnalysisSet(data, ti)
vs = VolumeAverageSet(data)
vs.add("ekin", "%10.5e")
an.add("volume_average", 100, {'va_obj':vs})
an.add("field_snap",1000)

dt=1/150.
an.run()

while ti.ok:
    ti.advance(data,dt)
    an.run()

ti.final_stats()

