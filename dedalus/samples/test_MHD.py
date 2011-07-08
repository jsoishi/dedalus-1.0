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
shape = (128,128) #(86, 86) 
RHS = MHD(shape, FourierData)
RHS.parameters['nu'] = 1e-3
data = RHS.create_fields(0.)            # B SHOULD BE ZERO BY DEFAULT

MIT_vortices(data)
#ti = RK2simplevisc(RHS,CFL=0.4)
ti = CrankNicholsonVisc(RHS)
ti.stop_time(100.) # set stoptime
#ti.stop_iter(10) # max iterations
ti.stop_walltime(3600.) # stop after 10 hours
ti.set_nsnap(100) # save data every 100 timesteps
ti.set_dtsnap(100.)

vs = VolumeAverageSet(data)
vs.add("ekin","%10.5e")
vs.add("enstrophy","%10.5e")
vs.add("vort_cenk","%10.5e")

an = AnalysisSet(data, ti)
an.add("field_snap", 25)
an.add("en_spec",100)
an.add("volume_average",10,{'va_obj': vs})


#main loop
dt = 1e-1
an.run()
while ti.ok:
    print "step: %i" % ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
