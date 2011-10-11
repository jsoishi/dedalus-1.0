"""Random seeded MRI test

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
import numpy as np

shape = (32, 32, 32)
length = (2 * np.pi,) * 3
RHS = ShearMHD(shape, FourierShearRepresentation, length=length)
RHS.parameters['Omega'] = 2 * np.pi
RHS.parameters['S'] = -1.5
data = RHS.create_fields(0.)

# Setup magnetic field
lambda_crit = np.pi / 2. 
vA = lambda_crit * data.parameters['Omega'] / (2 * np.pi * np.sqrt(16. / 15.))
Bmag = vA * np.sqrt(4 * np.pi * data.parameters['rho0'])
print 'lambda_crit = ', lambda_crit
print 'vA = ', vA
print 'Bmag = ', Bmag

# Setup B field
data['B']['z']['xspace'] += Bmag

# Setup u field
for i in xrange(3):
    ux = np.random.rand(*shape) - 0.5
    data['u'][i]['xspace'] = ux
    
data['u'].div_free()

umag = 1e-5
umax = np.max([data['u'][i]['xspace'].max() for i in xrange(3)])

for i in xrange(3):
    data['u'][i]['xspace'] *= umag / umax

# Integration parameters
ti = RK2simple(RHS)
ti.stop_iter(10)
ti.stop_time(10.) # set stoptime
ti.stop_walltime(3600.) # stop after 1 hour
ti.set_nsnap(1e6)
ti.set_dtsnap(1e6)

vs = VolumeAverageSet(data)
vs.add('ekin', '%10.5e')
vs.add('emag', '%10.5e')

xzslice = [slice(None), shape[1] / 2, slice(None)]
yzslice = [slice(None), slice(None), shape[2] / 2]
plot_cadence = 20

an = AnalysisSet(data, ti)
an.add("volume_average",1,{'va_obj': vs})
an.add("field_snap", plot_cadence, {'plot_slice': xzslice, 'saveas': 'xz_snap'})
an.add("field_snap", plot_cadence, {'plot_slice': yzslice, 'saveas': 'yz_snap'})
an.add("k_plot", plot_cadence)
an.add("en_spec", plot_cadence, {'flist': ['u', 'B']})
an.add("phase_track", plot_cadence, {'flist': ['u', 'B'],
                                   'klist': [(2, 0, 0), (4, 0, 0), (6, 0, 0)]})

# Main loop
vA_cfl_time = np.min(np.array(length) / np.array(shape)) / vA
cfl_times = np.array([vA_cfl_time])
dt = cfl_times.min() / 20
print 'CFL times: ', cfl_times
print 'Chosen dt: ', dt
print '-' * 10

an.run()
while ti.ok:
    print "step: %i" %ti.iter
    ti.advance(data, dt)
    an.run()

ti.final_stats()
