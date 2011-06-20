"""2D Turbulence test, following McWilliams JFM 1990, 219:361-385

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
from time_step import RK2simple,RK2simplevisc, RK2simplehypervisc4
from init_cond import turb, mcwilliams_spec
from analysis import AnalysisSet

shape = (450,450)
RHS = Hydro(shape, FourierData)
RHS.parameters['nu'] = 3.5e-7 # 100x mcwilliams
data = RHS.create_fields(0.)

turb(data['ux'],data['uy'],mcwilliams_spec,k0=23.)
ti = RK2simplehypervisc4(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
ti.stop_iter(100) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours

an = AnalysisSet(data, ti)
an.add("print_energy", 1)
an.add("field_snap", 10)
an.add("en_spec",5)
#main loop
dt = 2.5e-3
#snapshot(data,0)
an.run()
while ti.ok:
    print "step: %i" % ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
