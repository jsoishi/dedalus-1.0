"""Analysis.py defines analysis sets, which are groups of tasks, each
with a defined cadence in terms of timesteps. When run, the set will
see what tasks should be run, and runs only those.

Author: Matthew Turk <matthewturk@gmail.com
Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 Matthew Turk, J. S. Oishi.  All Rights Reserved.

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

import pylab as P
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as na
import os
from functools import wraps

class AnalysisSet(object):
    known_analysis = {}
    def __init__(self, data, ti):
        self.data = data
        self.ti = ti
        self.tasks = []

    def add(self, name, cadence, options={}):
        self.tasks.append((self.known_analysis[name], cadence, options))

    def run(self):
        for f, c, kwargs in self.tasks:
            if self.ti.iter % c != 0: continue
            if len(kwargs) == 0:
                f(self.data, self.ti.iter)
            else:
                f(self.data, self.ti.iter, **kwargs)

    @classmethod
    def register_task(cls, func):
        cls.known_analysis[func.func_name] = func

@AnalysisSet.register_task
def volume_average(data, it, va_obj=None):
    va_obj.run()

@AnalysisSet.register_task
def field_snap(data, it):
    """take a snapshot of all fields defined. currently only works in
    2D; it will need a slice index for 3D.

    """
    fields = data.fields.keys()
    fields.sort()
    nvars = len(fields)
    nrow = nvars / 3
    if nrow == 0:
        ncol = nvars % 3
        nrow = 1
    else:
        ncol = 3
    fig = P.figure(1,figsize=(24.*ncol/3.,24.*nrow/3.))
    grid = AxesGrid(fig,111,
                    nrows_ncols = (nrow, ncol),
                    axes_pad=0.1,
                    cbar_pad=0.,
                    share_all=True,
                    label_mode="1",
                    cbar_location="top",
                    cbar_mode="each")
    for i,f in enumerate(fields):
        im = grid[i].imshow(data[f]['xspace'].real)
        grid[i].text(0.05,0.95,f, transform=grid[i].transAxes, size=24,color='white')
        grid.cbar_axes[i].colorbar(im)
    tstr = 't = %5.2f' % data.time
    grid[0].text(-0.3,1.,tstr, transform=grid[0].transAxes,size=24,color='black')
    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/snap_%04i.png" % it
    fig.savefig(outfile)
    fig.clf()

@AnalysisSet.register_task
def print_energy(data, it):
    """compute energy in real space

    """

    energy = na.zeros(data['ux']['xspace'].shape)
    e2 = na.zeros_like(energy)
    for f in data.fields:
        energy += (data[f]['xspace']*data[f]['xspace'].conj()).real
        e2 += (data[f]['kspace']*data[f]['kspace'].conj()).real
    print "k energy: %10.5e" % (0.5* e2.sum())
    print "x energy: %10.5e" % (0.5*energy.sum()/energy.size)

@AnalysisSet.register_task
def en_spec(data, it):
    kk = na.zeros(data['ux'].data.shape)
    for k in data['ux'].k.values():
        kk += k**2
    kk = na.sqrt(kk)
    power = na.zeros(data['ux'].data.shape)
    for f in data.fields:
        power += (data[f]['kspace']*data[f]['kspace'].conj()).real

    power *= 0.5
    nbins = (data['ux'].k['x'].size)/2 
    k = na.arange(nbins)
    spec = na.zeros(nbins)
    for i in range(nbins):
        #spec[i] = (4*na.pi*i**2*power[(kk >= (i-1/2.)) & (kk <= (i+1/2.))]).sum()
        spec[i] = (power[(kk >= (i-1/2.)) & (kk <= (i+1/2.))]).sum()

    P.loglog(k[1:],spec[1:])
    from dedalus.init_cond.api import mcwilliams_spec
    mspec = mcwilliams_spec(k,30.)
    mspec *= 0.5/mspec.sum()
    print "E tot spec 1D = %10.5e" % mspec.sum()
    print "E tot spec 2D = %10.5e" % spec.sum()
    print "E0 2D = %10.5e" % spec[0]
    P.loglog(k[1:], mspec[1:])
    P.xlabel(r"$k$")
    P.ylabel(r"$E(k)$")

    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/enspec_%04i.png" % it
    P.savefig(outfile)
    P.clf()

