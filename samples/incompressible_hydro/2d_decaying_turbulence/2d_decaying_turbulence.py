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

shape = (450, 450)
RHS = IncompressibleHydro(shape, FourierRepresentation)
RHS.parameters['viscous_order'] = 2
RHS.parameters['nu'] = 3.5e-9
data = RHS.create_fields(0.)

turb_new(data, mcwilliams_spec, k0=30., E0=1.)
ti = RK2mid(RHS, CFL=0.4)
ti.sim_stop_time = 1. # set stoptime
ti.wall_stop_time = 36000. # stop after 10 hours
ti.save_cadence = 50
ti.max_save_period = 1.

vs = VolumeAverageSet(data)
vs.add("ekin","%20.10e", options={'space': 'kspace'})
vs.add("ekin","%20.10e", options={'space': 'xspace'})
vs.add("enstrophy","%10.5e", options={'space': 'kspace'})
vs.add("enstrophy","%10.5e", options={'space': 'xspace'})
vs.add("vort_cenk","%10.5e")

an = AnalysisSet(data, ti)
an.add(Snapshot(10, space='kspace'))
an.add(Snapshot(10, space='xspace'))
an.add(PowerSpectrum(10))
an.add(VolumeAverage(10, vs))

# Main loop
an.run()

dt = 1e-3
while ti.ok:
    ti.advance(data, dt)
    an.run()

ti.final_stats()
