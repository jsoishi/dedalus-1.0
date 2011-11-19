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
from dedalus.utils.parallelism import com_sys

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
def field_snap(data, it, plot_slice=None, use_extent=False, space='xspace', saveas='snap', **kwargs):
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
    
    if not os.path.exists('frames') and com_sys.myproc == 0:
        os.mkdir('frames')
    if space == 'kspace':
        outfile = "frames/k_" + saveas + "_%07i.png" % it
    else:
        outfile = "frames/" + saveas + "_%07i.png" % it
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

def compute_en_spec(data, field, normalization=1.0, averaging=None):
    """Compute power spectrum (helper function for analysis tasks).

    Inputs:
        fc               (field name, component) tuple 
        normalization    Power in each mode is multiplied by normalization
        averaging        None     : no averaging (default)
                         'all'    : divide power in each bin by number of modes 
                                    included in that bin
                         'nonzero': like 'all', but count only nonzero modes

    Returns:
        k                centers of k-bins
        spec             Power spectrum of f
    """
    f = data[field]
    power = na.zeros(f[0]['kspace'].shape)
    for i in xrange(f.ncomp):
        power += 0.5 * na.abs(f[i]['kspace']) ** 2
    power *= normalization

    # Construct bins by wavevector magnitude (evenly spaced)
    kmag = na.sqrt(f[0].k2())
    k = na.linspace(0, na.max(kmag), na.max(data.shape) / 2.)
    if com_sys.comm:
        # note: not the same k-values as serial version
        k = na.linspace(0, na.max(f[0].kny), na.max(data.shape) / 2.)
    
    kbottom = k - k[1] / 2.
    ktop = k + k[1] / 2.
    spec = na.zeros_like(k)
        
    nonzero = (power > 0)

    comm = com_sys.comm
    MPI = com_sys.MPI
    if comm:
        myspec = na.zeros_like(k)
        myproc = com_sys.myproc
        nk = na.zeros_like(spec)
        mynk = na.zeros_like(spec)
        for i in xrange(k.size):
            kshell = (kmag >= kbottom[i]) & (kmag < ktop[i])
            myspec[i] = (power[kshell]).sum()
            if averaging == 'all':
                mynk[i] = kshell.sum()
            elif averaging == 'nonzero':
                mynk[i] = (kshell & nonzero).sum()
        spec = comm.reduce(myspec, op=MPI.SUM, root=0)
        nk = comm.reduce(mynk, op=MPI.SUM, root=0)
        if myproc != 0: return None, None
        if averaging is None:
            return k, spec
        else:
            nk[(nk==0)] = 1.
            return k, spec/nk
    else:
        for i in xrange(k.size):
            kshell = (kmag >= kbottom[i]) & (kmag < ktop[i])
            spec[i] = (power[kshell]).sum()
            if averaging == 'nonzero':
                spec[i] /= (kshell & nonzero).sum()
            elif averaging == 'all':
                spec[i] /= kshell.sum()
        return k, spec

@AnalysisSet.register_task
def en_spec(data, it, flist=['u']):
    """Record power spectrum of specified fields."""
    N = len(flist)
    fig = P.figure(2, figsize=(8 * N, 8))
    
    for i,f in enumerate(flist):
        k, spectrum = compute_en_spec(data, f)

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
    #P.text(-0.3,1.,tstr, transform=P.gca().transAxes,size=24,color='black')
       
    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/enspec_%04i.png" %it
    P.savefig(outfile)
    P.clf()
    txtout = open("power.dat",'a')
    txtout.write(' '.join([str(i) for i in spectrum.tolist()])+'\n')
    txtout.close()

@AnalysisSet.register_task
def compare_power(data, it, f1='delta_b', f2='delta_c', comparison='ratio', output_columns=True):
    """Compare power spectrum of two fields. Defaults for baryon

    Inputs:
        data            Data object
        it              Iteration number
        f1, f2          Fields to compare
        comparison      'ratio'      : use P(f1)/P(f2) (default)
                        'difference' : use P(f1) - P(f2) 
        output_columns  if True, output data as columns in a file

    """
    k, spec_f1 = compute_en_spec(data, f1, averaging='nonzero')
    k, spec_f2 = compute_en_spec(data, f2, averaging='nonzero')

    if com_sys.myproc != 0:
        return

    if not os.path.exists('frames'):
        os.mkdir('frames')

    if output_columns:
        outfile = open('frames/spec_data_%s_%s_%04i.txt'%(f1,f2,it), 'w')
        for ak, s1, s2 in zip(k, spec_f1, spec_f2):
            outfile.write('%08f\t%08e\t%08e\n'%(ak, s1, s2))
        outfile.close()
        return

    fig = P.figure(figsize=(8,6))

    if comparison == 'ratio':
        spec_f2[spec_f2==0] = 1.
        spec_compare = spec_f1/spec_f2
        P.title('Comparison of %s and %s power, t = %5.2f' %(f1, f2, data.time))
        P.ylabel(r"P(%s)/P(%s)" %(f1, f2))
    elif comparison == 'difference':
        spec_compare = spec_f1 - spec_f2
        P.title('Comparison of %s and %s power, t = %5.2f' %(f1, f2, data.time))
        P.ylabel(r"$P(%s) - P(%s)$" %(f1, f2))
        
    P.xlabel(r"$k$")
    P.loglog(k[1:], spec_compare[1:], 'o-')
    
    outfile = "frames/cmpspec_%s_%s_%04i.png" %(f1, f2, it)
    P.savefig(outfile)
    P.clf()
    
@AnalysisSet.register_task
def mode_track(data, it, flist=[], klist=[], log=True):
    """
    Plot amplification of specified modes.
    
    Inputs:
        data        Data object
        it          Iteration number
        flist       List of fields to track: ['u', ...]
        klist       List of wavevectors given as tuples: [(1,0,0), (1,1,1), ...]
    
    """
    
    if it == 0:
        # Construct container on first pass
        data._save_modes = {}

        for f in flist:
            for i in xrange(data[f].ncomp):
                for k in klist:
                    data._save_modes[(f,i,k)] = [data[f][i]['kspace'][k[::-1]]]
        data._save_modes['time'] = [data.time]
        return
        
    # Save components and time
    for f in flist:
        for i in xrange(data[f].ncomp):
            for k in klist:
                data._save_modes[(f,i,k)].append(data[f][i]['kspace'][k[::-1]])
    data._save_modes['time'].append(data.time)
    
    # Determine image grid size
    nrow = len(flist)
    ncol = na.max([data[f].ncomp for f in flist])
    fig, axs = P.subplots(nrow, ncol, num=3, figsize=(8 * ncol, 6 * nrow)) 

    # Plot field components
    time = na.array(data._save_modes['time'])
        
    for j,f in enumerate(flist):
        for i in xrange(data[f].ncomp):
            for k in klist:
                plot_array = na.array(data._save_modes[(f,i,k)])
                power = 0.5 * na.abs(plot_array) ** 2
                
                if log:
                    axs[j, i].semilogy(time, power, '.-', label=str(k))
                else:
                    axs[j, i].plot(time, power, '.-', label=str(k))   
    
            # Pad and label axes
            axs[j, i].axis(padrange(axs[j, i].axis(), 0.05))
            axs[j, i].legend()
            axs[j, i].set_title(f + str(i))
                            

    axs[-1, 0].set_ylabel('power')
    axs[-1, 0].set_xlabel('time')

    if not os.path.exists('frames'):
        os.mkdir('frames')
    outfile = "frames/mode_track.png"
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
    
     

