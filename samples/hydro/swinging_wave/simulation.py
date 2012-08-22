"""
Swinging vorticity wave test from

    Lithwick, Y. Nonlinear Evolution of Hydrodynamical Shear Flows in Two
        Dimensions. The Astrophysical Journal 670, 789-804 (2007).
        http://labs.adsabs.harvard.edu/ui/abs/2007ApJ...670..789L

Author: K. J. Burns <keaton.burns@gmail.com>
Affiliation: UC Berkeley
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

import numpy as np
from dedalus.mods import *

shape = (30, 10)
length = (2 * np.pi, 100 * 2 * np.pi)
RHS = ShearIncompressibleHydro(shape, FourierShearRepresentation, length=length)
RHS.parameters['Omega'] = 1.
RHS.parameters['S'] = -1.5
data = RHS.create_fields(0.)

# Setup vorticity wave
k = (0.01, 4)
w = 0.01
vorticity_wave(data, k, w)

ti = RK2mid(RHS,CFL=0.4)
ti.stop_iter(1e6)
ti.stop_time(534)
ti.set_nsnap(10000)
ti.set_dtsnap(100.)

an = AnalysisSet(data, ti)
#vs = VolumeAverageSet(data)
#vs.add("ekin", "%10.5e")
#an.add("volume_average", 100, {'va_obj':vs})

an.add(Snapshot(1500, space='kspace'))
index = data['u']['x'].find_mode(k)
an.add(TrackMode(150, indexlist=[index]))

dt = 1. / 150.
an.run()

while ti.ok:
    ti.advance(data, dt)
    an.run()

an.cleanup()
ti.final_stats()

