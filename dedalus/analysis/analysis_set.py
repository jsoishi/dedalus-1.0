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

import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as P
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as na
import os
from functools import wraps

class AnalysisSet(object):

    # Dictionary of registered tasks
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
def field_snap(data, it, plot_slice=None, use_extent=False, space='xspace', **kwargs):
    """
    Take a snapshot of all fields defined. Currently takes z[0] slice for 3D.
    
    """
    
    # Determine image grid size
    nrow = len(data.fields.keys())
    ncol = na.max([f.ncomp for f in data.fields.values()])
    
    # Figure setup
    fig = P.figure(4, figsize=(8 * ncol, 8 * nrow))

    if use_extent:
        extent = [0., data.length[-1], 0., data.length[-2]]
    else:
        extent = None
    
    # Figure setup
    fig = P.figure(1, figsize=(8 * ncol, 8 * nrow))
    
    grid = AxesGrid(fig, 111,
                    nrows_ncols = (nrow, ncol),
                    axes_pad=0.3,
                    cbar_pad=0.,
                    share_all=True,
                    label_mode="1",
                    cbar_location="top",
                    cbar_mode="each")
                    
    # Plot field components
    i = -1
    for k,f in data.fields.iteritems():
        i += 1
        for j in xrange(f.ncomp):
            I = i * ncol + j
        
            if f[j].ndim == 3:
                if plot_slice == None:
                    # Default to bottom xy plane
                    plot_slice = [0, slice(None), slice(None)]
                plot_array = f[j][space][plot_slice]
            else:
                plot_array = f[j][space]
                
            if space == 'kspace':
                plot_array = na.abs(plot_array)
                plot_array[plot_array == 0] = na.nan
                plot_array = na.log10(plot_array)

            else:
                plot_array = plot_array.real

            # Plot
            im = grid[I].imshow(plot_array, extent=extent, origin='lower', 
                                interpolation='nearest', **kwargs)
            grid[I].text(0.05, 0.95, k + str(j), transform=grid[I].transAxes, size=24, color='white')
            grid.cbar_axes[I].colorbar(im)
            
    # Time label     
    tstr = 't = %5.2f' % data.time
    grid[0].text(-0.3,1.,tstr, transform=grid[0].transAxes,size=24,color='black')
    
    if not os.path.exists('frames'):
        os.mkdir('frames')
    if space == 'kspace':
        outfile = "frames/k_snap_%07i.png" % it
    else:
        outfile = "frames/snap_%07i.png" % it
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
def en_spec(data, it, flist=['u']):
    """Record power spectrum of specified fields."""
    
    N = len(flist)
    fig = P.figure(2, figsize=(8 * N, 8))
    
    for i,f in enumerate(flist):
        fx = data[f]['x']
        
        # Calculate power in each mode
        power = na.zeros(fx.data.shape)
        for j in xrange(data[f].ncomp):
            power += 0.5 * na.abs(data[f][j]['kspace']) ** 2

        # Construct bins by wavevector magnitude
        kmag = na.sqrt(fx.k2())
        k = na.linspace(0, na.max(kmag), na.max(data.shape) / 2.)
        kbottom = k - k[1] / 2.
        ktop = k + k[1] / 2.
        spectrum = na.zeros_like(k)
        
        for j in xrange(k.size):
            spectrum[j] = (power[(kmag >= kbottom[j]) & (kmag < ktop[j])]).sum()
    
        # Plotting, skip if all modes are zero
        if spectrum[1:].nonzero()[0].size == 0:
            return
        
        ax = fig.add_subplot(1, N, i+1)
        ax.semilogy(k[1:], spectrum[1:], 'o-')
        
        print "%s E total power = %10.5e" %(f, spectrum.sum())
        print "%s E0 power = %10.5e" %(f, spectrum[0])
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"$E(k)$")
        ax.set_title('%s Power, time = %5.2f' %(f, data.time))

        
    # Add timestamp
    #tstr = 't = %5.2f' % data.time
    #P.text(-0.3,1.,tstr, transform=P.gca().transAxes, size=24, color='black')

    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/enspec_%04i.png" %it
    P.savefig(outfile)
    P.clf()
    
@AnalysisSet.register_task
def phase_amp(data, it, fclist=[], klist=[], log=True):
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

        for f,c in fclist:
            for k in klist:
                data._save_modes[(f,c,k)] = [data[f][c]['kspace'][k[::-1]]]
        data._save_modes['time'] = [data.time]
        return
        
    # Save components and time
    for f,c in fclist:
        for k in klist:
            data._save_modes[(f,c,k)].append(data[f][c]['kspace'][k[::-1]])
    data._save_modes['time'].append(data.time)
    
    # Plotting setup
    nvars = len(fclist)
    fig, axs = P.subplots(2, nvars, num=3, figsize=(8 * nvars, 6 * 2)) 

    # Plot field components
    time = na.array(data._save_modes['time'])
    
    I = 0
    for f,c in fclist:
        for k in klist:
            plot_array = na.array(data._save_modes[(f,c,k)])
            power = 0.5 * na.abs(plot_array) ** 2
            
            # Phase evolution at fixed point is propto exp(-omega * t)
            dtheta = -na.diff(na.angle(plot_array))
            
            # Correct for pi boundary crossing
            dtheta[dtheta > na.pi] -= 2 * na.pi
            dtheta[dtheta < -na.pi] += 2 * na.pi
            
            # Calculate phase velocity
            omega = dtheta / na.diff(time)
            phase_velocity = omega / na.linalg.norm(k)

            if log:
                axs[0, I].semilogy(time, power, '.-', label=str(k))
            else:
                axs[0, I].plot(time, power, '.-', label=str(k))   
                
            axs[1, I].plot(time[1:], phase_velocity, '.-', label=str(k))

        # Pad and label axes
        axs[0, I].axis(padrange(axs[0, I].axis(), 0.05))
        axs[1, I].axis(padrange(axs[1, I].axis(), 0.05))
                        
        if I == 0:
            axs[0, I].set_ylabel('power')
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
    
@AnalysisSet.register_task
def k_plot(data, it, zcut=0):
    """
    Plot k-power for moving k modes (i.e. ShearReps)
    
    Inputs:
        data        Data object
        it          Iteration number

    Keywords:
        zcut        kz index for the kx,ky plane being plotted

    """
    
    # Determine image grid size
    nrow = len(data.fields.keys())
    ncol = na.max([f.ncomp for f in data.fields.values()])
    
    # Figure setup
    fig = P.figure(4, figsize=(8 * ncol, 8 * nrow))
    
    grid = AxesGrid(fig, 111,
                    nrows_ncols = (nrow, ncol),
                    aspect=False,
                    share_all=True,
                    axes_pad=0.3,
                    cbar_pad=0.,
                    label_mode="1",
                    cbar_location="top",
                    cbar_mode="each")
                    
    # Plot field components
    i = -1
    z_ = na.zeros(data.shape[-2:])
    ny = data['u']['x'].kny
    
    for k,f in data.fields.iteritems():
        i += 1
        for j in xrange(f.ncomp):
            I = i * ncol + j
        
            x = f[j].k['x'][0] + z_
            y = f[j].k['y'][0] + z_

            if f[j].ndim == 3:
                plot_array = f[j]['kspace'][zcut]
            else:
                plot_array = f[j]['kspace']
                
            plot_array = na.abs(plot_array)
            plot_array[plot_array == 0] = 1e-40
            plot_array = na.log10(plot_array)
            
            # Plot
            im = grid[I].scatter(x, y, c=plot_array, linewidth=0, 
                                 vmax=na.max([plot_array.max(), -39]))
            
            # Nyquist boundary
            nysquarex = na.array([-ny[-1], -ny[-1], ny[-1], ny[-1], -ny[-1]])
            nysquarey = na.array([-ny[-2], ny[-2], ny[-2], -ny[-2], -ny[-2]])
            grid[I].plot(nysquarex, nysquarey, 'k--')
            
            # Dealiasing boundary
            grid[I].plot(2/3. * nysquarex, 2/3. * nysquarey, 'k:')
            
            # Plot range and labels
            grid[I].axis(padrange([-ny[-1], ny[1], -ny[-2], ny[-2]], 0.2))
            grid[I].text(0.05, 0.95, k + str(j), transform=grid[I].transAxes, size=24, color='black')
            grid.cbar_axes[I].colorbar(im)
            
    # Time label
    tstr = 't = %6.3f' % data.time
    grid[0].text(-0.3,1.,tstr, transform=grid[0].transAxes, size=24, color='black')
    
    grid[0].set_xlabel('kx')
    grid[0].set_ylabel('ky')
    
    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/k_plot_%07i.png" % it
    fig.savefig(outfile)
    fig.clf()

    
def padrange(range, pad=0.05):
    xmin, xmax, ymin, ymax = range
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0: dx = 1.
    if dy == 0: dy = 1.
    outrange = [xmin - pad * dx,
                xmax + pad * dx,
                ymin - pad * dy,
                ymax + pad * dy]
                
    return outrange
    
     

