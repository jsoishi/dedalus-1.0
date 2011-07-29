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
from dedalus.mods import *

shape = (450,450)
RHS = Hydro(shape, FourierRepresentation)
RHS.parameters['nu'] = 3.5e-7 # 100x mcwilliams
data = RHS.create_fields(0.)

turb(data['u']['x'],data['u']['y'],mcwilliams_spec,k0=23.)
ti = RK2simplehypervisc4(RHS,CFL=0.4)
ti.stop_time(1.) # set stoptime
ti.stop_iter(100) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours

ti.set_nsnap(50)
ti.set_dtsnap(1.)
vs = VolumeAverageSet(data)
vs.add("ekin","%10.5e")
vs.add("enstrophy","%10.5e")
vs.add("vort_cenk","%10.5e")
vs.add("ux_imag", "%10.5e")
vs.add("uy_imag", "%10.5e")
vs.add("ux_imag_max", "%10.5e")
vs.add("uy_imag_max", "%10.5e")

an = AnalysisSet(data, ti)
an.add("field_snap", 20)
an.add("en_spec",10)
an.add("volume_average",1,{'va_obj': vs})


#main loop
dt = 2.5e-3
an.run()
while ti.ok:
    print "step: %i" % ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
