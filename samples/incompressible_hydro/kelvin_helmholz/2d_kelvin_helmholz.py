"""2D Kelvin-Helmholz test, following McNally, Lyra, & Passy ApJ 201:18 (2012)

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
import numpy as np
from dedalus.config import decfg
decfg.set('physics','use_tracer', 'True')
decfg.set('analysis', 'snapshot_cmap','bone')

from dedalus.mods import *

shape = (128, 128) # low resolution run
length = (1., 1.)
RHS = IncompressibleHydro(shape, FourierRepresentation, length=length)
RHS.parameters['viscous_order'] = 2
RHS.parameters['nu'] = 3.5e-9
data = RHS.create_fields(0.)

# initial conditions
def kelvin_helmholz(data, L=0.025):
    """construct well-posed Kelvin-Helmholz test.
    inputs
    ------
    data -- the data object
    L (optional) -- smoothing length
    """
    # parameters
    u1 = 0.5
    u2 = -0.5
    um = (u1 - u2)/2.
    rho1 = 1.0
    rho2 = 2.0
    rhom = (rho1 - rho2)/2.

    grid = data['u']['x'].xspace_grid()
    y = grid[0]

    # base flow
    y1 = (y >= 0) & (y < 0.25)
    data['u']['x']['xspace'][y1] = u1 - um * np.exp((y[y1] - 0.25)/L)
    data['c']['xspace'][y1] = rho1 - rhom * np.exp((y[y1] - 0.25)/L)

    y2 = (y >= 0.25) & (y < 0.5)
    data['u']['x']['xspace'][y2] = u2 + um * np.exp((-y[y2] + 0.25)/L)
    data['c']['xspace'][y2] = rho2 + rhom * np.exp((-y[y2] + 0.25)/L)

    y3 = (y >= 0.5) & (y < 0.75)
    data['u']['x']['xspace'][y3] = u2 + um * np.exp(-(0.75 - y[y3])/L)
    data['c']['xspace'][y3] = rho2 + rhom * np.exp(-(0.75 - y[y3])/L)

    y4 = (y >= 0.75) & (y < 1.)
    data['u']['x']['xspace'][y4] = u1 - um * np.exp(-(y[y4] - 0.75)/L)
    data['c']['xspace'][y4] = rho1 - rhom * np.exp(-(y[y4] - 0.75)/L)

    # perturbation
    data['u']['y']['xspace'] = 0.01 * np.sin(4*np.pi* grid[1])

kelvin_helmholz(data)

ti = RK2mid(RHS, CFL=0.4)
ti.sim_stop_time = 2. # set stoptime
ti.wall_stop_time = 36000. # stop after 10 hours
#ti.stop_iterartion = 1
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
an.add(Snapshot(10, space='xspace', even_scale=False))
an.add(PowerSpectrum(10))
an.add(VolumeAverage(10, vs))

# Main loop
an.run()

dt = 1e-3
while ti.ok:
    ti.advance(data, dt)
    an.run()

ti.final_stats()
