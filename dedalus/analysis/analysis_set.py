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

import matplotlib.pyplot as P
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as na
import os
from functools import wraps

class AnalysisSet(object):

    # Dictionary of registered analysis tasks
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

    # Task registration
    @classmethod
    def register_task(cls, func):
        cls.known_analysis[func.func_name] = func

@AnalysisSet.register_task
def volume_average(data, it, va_obj=None):
    va_obj.run()

@AnalysisSet.register_task
def field_snap(data, it, use_extent=False, **kwargs):
    """
    Take a snapshot of all fields defined. Currently takes z[0] slice for 3D.
    
    """
    
    # Determine image grid size
    nvars = 0
    for f in data.fields.values():
        nvars += f.ncomp
    if nvars == 4:
        nrow = ncol = 2
    elif nvars == 9:
        nrow = ncol = 3
    else:
        nrow = na.ceil(nvars / 3.)
        ncol = na.min([nvars, 3])
    nrow = na.int(nrow)
    ncol = na.int(ncol)

    if use_extent:
        extent = [0.,data.length[1], 0., data.length[0]]
    else:
        extent = None
    
    # Figure setup
    fig = P.figure(1, figsize=(24. * ncol / 3., 24. * nrow / 3.))
    grid = AxesGrid(fig, 111,
                    nrows_ncols = (nrow, ncol),
                    axes_pad=0.3,
                    cbar_pad=0.,
                    share_all=True,
                    label_mode="1",
                    cbar_location="top",
                    cbar_mode="each")
                    
    # Plot field components
    I = 0
    for k,f in data.fields.iteritems():
        for i in xrange(f.ncomp):
            if f[i].ndim == 3:
                plot_array = f[i]['xspace'][0,:,:].real
            else:
                plot_array = f[i]['xspace'].real
            im = grid[I].imshow(plot_array, extent=extent, origin='lower', 
                                interpolation=None, **kwargs)
            grid[I].text(0.05, 0.95, k + str(i), transform=grid[I].transAxes, size=24,color='white')
            grid.cbar_axes[I].colorbar(im)
            I += 1
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
    kk = na.sqrt(data['u']['x'].k2())
    power = na.zeros(data['u']['x'].data.shape)
    for i in range(data['u'].ndim):
        power += (data['u'][i]['kspace']*data['u'][i]['kspace'].conj()).real

    power *= 0.5
    nbins = (data['u']['x'].k['x'].size)/2 
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
    
@AnalysisSet.register_task
def phase_amp(data, it, fclist=[], klist=[]):
    """
    Plot phase velocity and amplification of specified modes.
    
    Inputs:
        data        Data object
        it          Iteration number
        fclist      List of field/component tuples to track: [('u', 'x'), ('phi', 0), ...]
        klist       List of wavevectors given as tuples: [(1,0,0), (1,1,1), ...]
    
    """
    
    if it == 0:
        # Construct container on first pass
        data._save_modes = {}
        data._init_power = {}
        
        for f,c in fclist:
            data._init_power[f] = 0
        
        for f,c in fclist:
            for k in klist:
                data._save_modes[(f,c,k)] = [data[f][c]['kspace'][k[::-1]]]
                data._init_power[f] += na.abs(data._save_modes[(f,c,k)]) ** 2.
        data._save_modes['time'] = [data.time]
        return
        
    # Save components and time
    for f,c in fclist:
        for k in klist:
            data._save_modes[(f,c,k)].append(data[f][c]['kspace'][k[::-1]])
    data._save_modes['time'].append(data.time)
    
    # Plotting setup
    nvars = len(fclist)
    fig, axs = P.subplots(2, nvars, num=2, figsize=(8 * nvars, 6 * 2)) 

    # Plot field components
    time = na.array(data._save_modes['time'])
    
    I = 0
    for f,c in fclist:
        for k in klist:
            plot_array = na.array(data._save_modes[(f,c,k)])
            
            # Calculate amplitude growth, normalized to initial power
            amp_growth = na.abs(plot_array) / na.sqrt(data._init_power[f]) - 1
            
            # Phase evolution at fixed point is propto exp(-omega * t)
            dtheta = -na.diff(na.angle(plot_array))
            
            # Correct for pi boundary crossing
            dtheta[dtheta > na.pi] -= 2 * na.pi
            dtheta[dtheta < -na.pi] += 2 * na.pi
            
            # Calculate phase velocity
            omega = dtheta / na.diff(time)
            phase_velocity = omega / na.linalg.norm(k)

            axs[0, I].plot(time, amp_growth, '.-', label=str(k))
            axs[1, I].plot(time[1:], phase_velocity, '.-', label=str(k))

        # Pad and label axes
        axs[0, I].axis(padrange(axs[0, I].axis(), 0.05))
        axs[1, I].axis(padrange(axs[1, I].axis(), 0.05))
                        
        if I == 0:
            axs[0, I].set_ylabel('normalized amplitude growth')
            axs[1, I].set_ylabel('phase velocity')
            axs[1, I].set_xlabel('time')
        
        axs[0, I].legend()
        axs[1, I].legend()
        axs[0, I].set_title(f + c)

        I += 1

    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/phase_amp.png"
    fig.savefig(outfile)
    fig.clf()
    
def padrange(range, pad=0.05):
    xmin, xmax, ymin, ymax = range
    outrange = [xmin - pad * (xmax - xmin),
                xmax + pad * (xmax - xmin),
                ymin - pad * (ymax - ymin),
                ymax + pad * (ymax - ymin)]
                
    return outrange
    
     

