"""
Analysis.py defines analysis sets, which are groups of tasks, each
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
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


import numpy as na
import numpy.lib.stride_tricks as st
import os
from functools import wraps
from dedalus.config import decfg
from dedalus.utils.parallelism import com_sys, get_plane, strided_copy
from dedalus.utils.logger import mylog

class AnalysisSet(object):
    def __init__(self, data, ti):
        self.data = data
        self.ti = ti
        self.tasks = []

    def add(self, task):
        self.tasks.append(task)
        task._set = self
        task._n = len(self.tasks)
        task.setup(self.data, self.ti.iter, **task.options)

    def run(self):
        for task in self.tasks:
            if self.ti.iter % task.cadence == 0:
                task.run(self.data, self.ti.iter, **task.options)
                
    def cleanup(self):
        for task in self.tasks:
            task.cleanup(self.data, self.ti.iter, **task.options)
              
class AnalysisTask(object):
    def __init__(self, cadence, **kwargs):
        self._set = None
        self.cadence = cadence
        self.options = kwargs
        
    def setup(self, data, iter, **kwargs):
        pass
        
    def run(self, data, iter, **kwargs):
        pass
        
    def cleanup(self, data, iter, **kwargs):
        pass
        
class Snapshot(AnalysisTask):
    """
    Take image snapshot of data in array configuration.
    
    Keywords
    --------
    space: str
        'xspace' or 'kspace'
    axis : str, optional
        Axis normal to desired slice, defaults to 'z'. Ignored for 2D data.
        i.e. 'x' for a y-z plane
             'y' for a x-z plane
             'z' for a x-y plane
    index : int or string, optional
        Index for slicing as an integer, or 'top', 'middle' (default), or 'bottom'.
        Ignored for 2D data.
    aspect : str
        'auto' stretches image to match axis
        'equal' stretches axis to match image
    units : boolean
        True for physical-space layout and units.
        False for array-based layout and indexing.

    """
    
    def setup(self, data, it, **kwargs):
    
        # Determine grid size
        self.nrows = len(data.fields.keys())
        self.ncols = na.max([f.ncomp for f in data.fields.values()])
        
        # Figure setup for proc 0
        if com_sys.myproc == 0:

            # Create figure and axes grid
            self.firstplot = True
            self.patch_lists = {}
            self.patch_collections = {}
            self.fig = P.figure(self._n, figsize=(8 * self.ncols, 8 * self.nrows))
            self.fig.clear()
            self.grid = AxesGrid(self.fig, 111,
                                 nrows_ncols = (self.nrows, self.ncols),
                                 aspect=False,
                                 share_all=True,
                                 axes_pad=0.3,
                                 cbar_pad=0.,
                                 label_mode="L",
                                 cbar_location="top",
                                 cbar_mode="each")
                                     
            # Directory setup
            if not os.path.exists('frames'):
                os.mkdir('frames')
    
    def run(self, data, it, space='xspace', axis='z', index='middle', 
            aspect='auto', units=True, **kwargs):

        # Plot field components
        for row, (fname, field) in enumerate(data):
            for cindex, comp in field:
                axnum = row * self.ncols + cindex
                
                # Retrieve correct slice
                if axnum == 0:
                    plane_data, outindex, namex, x, namey, y = get_plane(data, 
                        fname, cindex, space=space, axis=axis, index=index)
                else:
                    plane_data, outindex = get_plane(data, fname, cindex,
                            space=space, axis=axis, index=index,
                            return_position_arrays=False)

                if com_sys.myproc == 0:
                    # kspace: take logarithm of magnitude, flooring at eps
                    if space == 'kspace':
                        kmag = na.abs(plane_data)
                        zero_mask = (kmag == 0)
                        kmag[kmag < comp._eps['kspace']] = comp._eps['kspace']
                        kmag = na.ma.array(kmag, mask=zero_mask)
                        plane_data = na.log10(kmag)
    
                    # Plot
                    if self.firstplot:
                        self.add_patches(axnum, x, y, plane_data, units, space)
                        self.add_lines(axnum, x, y, units, space, comp, namex, namey)
                    else:
                        self.update_patches(axnum, x, y, plane_data, units, space)
                        
                    # Labels
                    if self.firstplot:
                        title = self.grid[axnum].set_title(fname + field.ctrans[cindex],
                                size=20, color='k')
                        title.set_y(1.08)
                        if row == self.nrows - 1:
                            self.grid[axnum].set_xlabel(namex)
                        if cindex == 0:
                            self.grid[axnum].set_ylabel(namey)
                              
        if com_sys.myproc == 0:
            # Add time to figure title
            tstr = 't = %6.3f' % data.time
            if self.firstplot:
                self.timestr = self.fig.suptitle(tstr, size=24, color='k')
                self.firstplot = False
            else:
                self.timestr.set_text(tstr)
            
            # Save in frames folder
            if data.ndim == 2:
                slicestr = ''
            elif data.ndim == 3:
                slicestr = '%s_%i_' %(axis, outindex)
            outfile = "frames/%ssnap_%sn%07i.png" %(space[0], slicestr, it)
            self.fig.savefig(outfile)
                                  
    def add_patches(self, axnum, x, y, plane_data, units, space):

        # Construct patches
        shape = plane_data.shape
        patches = []
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                if units:
                    dx = x[0, 1] - x[0, 0]
                    dy = y[1, 0] - y[0, 0]
                    if space == 'kspace':
                        xy = (x[i, j] - dx / 2., y[i, j] - dy / 2.)
                    else:
                        xy = (x[i, j], y[i, j])
                else:
                    dx = dy = 1
                    xy = (j - dx / 2., i - dy / 2.)
                rect = Rectangle(xy, dx, dy)
                patches.append(rect) 
                  
        cmap = matplotlib.cm.Spectral_r
        cmap.set_bad('0.7')
        pc = PatchCollection(patches, cmap=cmap, alpha=1., zorder=1, edgecolors='white') 
                 
        # Store for updating        
        self.patch_lists[axnum] = patches
        self.patch_collections[axnum] = pc
        
        # Set values
        pc.set_array(na.ma.ravel(plane_data))
        self.grid[axnum].add_collection(pc)
        self.grid.cbar_axes[axnum].colorbar(pc)
        
    def update_patches(self, axnum, x, y, plane_data, units, space):
    
        # Retrieve patches
        shape = plane_data.shape
        patches = self.patch_lists[axnum]
        pc = self.patch_collections[axnum]
        
        # Update positions
        if units and space == 'kspace':
            for i in xrange(shape[0]):
                for j in xrange(shape[1]):
                    dx = x[0, 1] - x[0, 0]
                    dy = y[1, 0] - y[0, 0]
                    xy = (x[i, j] - dx / 2., y[i, j] - dy / 2.)
                    patches[i * shape[1] + j].set_xy(xy)
            pc.set_paths(patches)
           
        # Update values and colorbar     
        pc.set_array(na.ma.ravel(plane_data))
        pc.set_clim(plane_data.min(), plane_data.max())
        
    def add_lines(self, axnum, x, y, units, space, comp, namex, namey):
        
        if units:
            dx = x[0, 1] - x[0, 0]
            dy = y[1, 0] - y[0, 0]
            
            if space == 'kspace':
            
                # Zero lines
                self.grid[axnum].axhline(0, c='k', zorder=2, lw=2)
                self.grid[axnum].axvline(0, c='k', zorder=2, lw=2)
                
                # Dealiasing boundary
                nyx = comp.kny[comp.ktrans[namex[1]]]
                nyy = comp.kny[comp.ktrans[namey[1]]]
                if namex == 'kx':
                    xsq = na.array([0, nyx, nyx, 0, 0])
                    ysq = na.array([nyy, nyy, -nyy, -nyy, nyy])
                else:
                    xsq = na.array([-nyx, nyx, nyx, -nyx, -nyx])
                    ysq = na.array([nyy, nyy, -nyy, -nyy, nyy])
                xsh = na.array([-dx / 2., dx / 2., dx / 2., -dx / 2., -dx / 2.])
                ysh = na.array([dy / 2., dy / 2., -dy / 2., -dy / 2., dy / 2.])
                self.grid[axnum].plot(2./3. * xsq + xsh, 2./3. * ysq + ysh, 
                        'k--', zorder=2, lw=2)
                
                plot_extent = [xsq[0] - dx / 2., xsq[1] + dx / 2., 
                               ysq[2] - dy / 2., ysq[0] + dy / 2.]
            
            elif space == 'xspace':
                xlen = comp.length[comp.xtrans[namex]]
                ylen = comp.length[comp.xtrans[namey]]
                plot_extent = [0, xlen, 0, ylen]
        
            # Plot range
            self.grid[axnum].axis(plot_extent)

                    





 
# 
# @AnalysisSet.register_task
# def volume_average(data, it, va_obj=None):
#     va_obj.run()
#        
# @AnalysisSet.register_task
# def array_snapshot(data, it, space='xspace', axis='z', index='middle', aspect='auto', figproc=0):
#     """
#     Take image snapshot of data in array configuration.
#     
#     Parameters
#     ----------
#     data : StateData object
#         Data input
#     it : int
#         Iteration number
#     space: str
#         'xspace' or 'kspace'
#     axis : str, optional
#         Axis normal to desired slice, defaults to 'z'. Ignored for 2D data.
#         i.e. 'x' for a y-z plane
#              'y' for a x-z plane
#              'z' for a x-y plane
#     index : int or string, optional
#         Index for slicing as an integer, or 'top', 'middle' (default), or 'bottom'.
#         Ignored for 2D data.
#     aspect : str
#         'auto' stretches image to match axis
#         'equal' stretches axis to match image
#     figproc : int
#         Processor to handle figure creation and drawing
# 
#     """
# 
#     # Figure setup
#     if com_sys.myproc == figproc:
#     
#         # Determine grid size
#         nrows = len(data.fields.keys())
#         ncols = na.max([f.ncomp for f in data.fields.values()])
#         
#         # Create figure and axes grid
#         fig = P.figure(1, figsize=(8 * ncols, 8 * nrows))
#         grid = AxesGrid(fig, 111,
#                         nrows_ncols = (nrows, ncols),
#                         aspect=False,
#                         share_all=True,
#                         axes_pad=0.3,
#                         cbar_pad=0.,
#                         label_mode="L",
#                         cbar_location="top",
#                         cbar_mode="each")
#                     
#     # Plot field components
#     row = -1
#     for fname, field in data:
#         row += 1
#         for cindex, comp in field:
# 
#             # Retrieve correct slice
#             plane_data, outindex, name0, x0, name1, x1 = get_plane(data, 
#                     fname, cindex, space=space, axis=axis, index=index)
#             if com_sys.myproc != figproc:
#                 continue
#               
#             # Plot
#             axnum = row * ncols + cindex
#             if space == 'kspace':
#                 # Take logarithm of magnitude, flooring at eps
#                 kmag = na.abs(plane_data)
#                 zero_mask = (kmag == 0)    
#                 kmag[kmag < comp._eps['kspace']] = comp._eps['kspace']
#                 plane_data = na.log10(kmag)
#                 plane_data[zero_mask] = na.nan
# 
#             im = grid[axnum].imshow(plane_data, origin='lower', aspect='auto',
#                     interpolation='nearest', zorder=2)     
#             grid.cbar_axes[axnum].colorbar(im)
#             
#             # Labels
#             grid[axnum].text(0.05, 0.95, fname + field.ctrans[cindex], 
#                     transform=grid[axnum].transAxes, size=24, color='w')
#             if row == nrows - 1:
#                 grid[axnum].set_xlabel(name0)
#             if cindex == 0:
#                 grid[axnum].set_ylabel(name1)
#     
#     # Bail of not figproc
#     if com_sys.myproc != figproc:
#         return
#             
#     # Add time to figure title
#     tstr = 't = %6.3f' % data.time
#     fig.suptitle(tstr, size=24, color='k')
#     
#     # Save in frames folder
#     if not os.path.exists('frames'):
#         os.mkdir('frames')
#     if data.ndim == 2:
#         slicestr = ''
#     elif data.ndim == 3:
#         slicestr = '%s_%i_' %(axis, outindex)
#     outfile = "frames/%s_array_%sn%07i.png" %(space[0], slicestr, it)
#     fig.savefig(outfile)
#     fig.clf()
# 
# @AnalysisSet.register_task
# def print_energy(data, it):
#     """compute energy in real space
# 
#     """
# 
#     energy = na.zeros(data['ux']['xspace'].shape)
#     e2 = na.zeros_like(energy)
#     for f in data.fields:
#         energy += (data[f]['xspace']*data[f]['xspace'].conj()).real
#         e2 += (data[f]['kspace']*data[f]['kspace'].conj()).real
#     print "k energy: %10.5e" % (0.5* e2.sum())
#     print "x energy: %10.5e" % (0.5*energy.sum()/energy.size)
# 
# def compute_en_spec(data, field, normalization=1.0, averaging=None):
#     """Compute power spectrum (helper function for analysis tasks).
# 
#     Inputs:
#         fc               (field name, component) tuple 
#         normalization    Power in each mode is multiplied by normalization
#         averaging        None     : no averaging (default)
#                          'all'    : divide power in each bin by number of modes 
#                                     included in that bin
#                          'nonzero': like 'all', but count only nonzero modes
# 
#     Returns:
#         k                centers of k-bins
#         spec             Power spectrum of f
#     """
#     f = data[field]
#     power = na.zeros(f[0]['kspace'].shape)
#     for i in xrange(f.ncomp):
#         power += na.abs(f[i]['kspace']) ** 2
#     power *= normalization
# 
#     # Construct bins by wavevector magnitude (evenly spaced)
#     kmag = na.sqrt(f[0].k2())
# 
#     if com_sys.comm:
#         # note: not the same k-values as serial version
#         k = na.linspace(0, int(na.max(f[0].kny)), na.max(data.shape)/2 + 1, endpoint=False)
#     else:
#         k = na.linspace(0, na.max(kmag), na.max(data.shape) / 2.)
#     dk = k[1] - k[0]
#     print "dk = %10.5e" % dk
#     kbottom = k #- k[1] / 2.
#     ktop = k + k[1]#/ 2.
#     print "sizeof(kbottom) = %i" % kbottom.size
#     print "sizeof(ktop) = %i" % ktop.size
#     spec = na.zeros_like(k)
#     nonzero = (power > 0)
# 
#     comm = com_sys.comm
#     MPI = com_sys.MPI
#     if comm:
#         myspec = na.zeros_like(k)
#         myproc = com_sys.myproc
#         nk = na.zeros_like(spec)
#         mynk = na.zeros_like(spec)
#         for i in xrange(k.size):
#             kshell = (kmag >= kbottom[i]) & (kmag < ktop[i])
#             myspec[i] = (power[kshell]).sum()
#             if averaging == 'all':
#                 mynk[i] = kshell.sum()
#             elif averaging == 'nonzero':
#                 mynk[i] = (kshell & nonzero).sum()
#         spec = comm.reduce(myspec, op=MPI.SUM, root=0)
#         nk = comm.reduce(mynk, op=MPI.SUM, root=0)
#         if myproc != 0: return None, None
#         if averaging is None:
#             return k, spec
#         else:
#             nk[(nk==0)] = 1.
#             return k, spec/nk
#     else:
#         for i in xrange(k.size):
#             kshell = (kmag >= kbottom[i]) & (kmag < ktop[i])
#             spec[i] = (power[kshell]).sum()
#             if averaging == 'nonzero':
#                 spec[i] /= (kshell & nonzero).sum()
#             elif averaging == 'all':
#                 spec[i] /= kshell.sum()
#         return k, spec
# 
# @AnalysisSet.register_task
# def en_spec(data, it, flist=['u'], loglog=False):
#     """Record power spectrum of specified fields."""
#     N = len(flist)
#     if com_sys.myproc == 0:
#         fig = P.figure(2, figsize=(8 * N, 8))
#     
#     for i,f in enumerate(flist):
#         k, spectrum = compute_en_spec(data, f)
# 
#         if com_sys.myproc == 0:
#             # Plotting, skip if all modes are zero
#             if spectrum[1:].nonzero()[0].size == 0:
#                 return
# 
#             ax = fig.add_subplot(1, N, i+1)
#             if loglog:
#                 ax.loglog(k[1:], spectrum[1:], 'o-')
#             else:
#                 ax.semilogy(k[1:], spectrum[1:], 'o-')
# 
#             print "%s E total power = %10.5e" %(f, spectrum.sum())
#             print "%s E0 power = %10.5e" %(f, spectrum[0])
#             ax.set_xlabel(r"$k$")
#             ax.set_ylabel(r"$E(k)$")
#             ax.set_title('%s Power, time = %5.2f' %(f, data.time))
# 
#     if com_sys.myproc == 0:
#         # Add timestamp
#         #tstr = 't = %5.2f' % data.time
#         #P.text(-0.3,1.,tstr, transform=P.gca().transAxes,size=24,color='black')
#        
#         if not os.path.exists('frames'):
#             os.mkdir('frames')
#         outfile = "frames/enspec_%04i.png" %it
#         P.savefig(outfile)
#         P.clf()
#         txtout = open("power.dat",'a')
#         txtout.write(' '.join([str(i) for i in spectrum.tolist()])+'\n')
#         txtout.close()
# 
# @AnalysisSet.register_task
# def compare_power(data, it, f1='delta_b', f2='delta_c', comparison='ratio', output_columns=True):
#     """Compare power spectrum of two fields. Defaults for baryon
# 
#     Inputs:
#         data            Data object
#         it              Iteration number
#         f1, f2          Fields to compare
#         comparison      'ratio'      : use P(f1)/P(f2) (default)
#                         'difference' : use P(f1) - P(f2) 
#         output_columns  if True, output data as columns in a file
# 
#     """
#     k, spec_f1 = compute_en_spec(data, f1, averaging='nonzero')
#     k, spec_f2 = compute_en_spec(data, f2, averaging='nonzero')
# 
#     if com_sys.myproc != 0:
#         return
# 
#     if not os.path.exists('frames'):
#         os.mkdir('frames')
# 
#     if output_columns:
#         outfile = open('frames/spec_data_%s_%s_%04i.txt'%(f1,f2,it), 'w')
#         for ak, s1, s2 in zip(k, spec_f1, spec_f2):
#             outfile.write('%08f\t%08e\t%08e\n'%(ak, s1, s2))
#         outfile.close()
#         return
# 
#     fig = P.figure(figsize=(8,6))
# 
#     if comparison == 'ratio':
#         spec_f2[spec_f2==0] = 1.
#         spec_compare = spec_f1/spec_f2
#         P.title('Comparison of %s and %s power, t = %5.2f' %(f1, f2, data.time))
#         P.ylabel(r"P(%s)/P(%s)" %(f1, f2))
#     elif comparison == 'difference':
#         spec_compare = spec_f1 - spec_f2
#         P.title('Comparison of %s and %s power, t = %5.2f' %(f1, f2, data.time))
#         P.ylabel(r"$P(%s) - P(%s)$" %(f1, f2))
#         
#     P.xlabel(r"$k$")
#     P.loglog(k[1:], spec_compare[1:], 'o-')
#     
#     outfile = "frames/cmpspec_%s_%s_%04i.png" %(f1, f2, it)
#     P.savefig(outfile)
#     P.clf()
#     
# @AnalysisSet.register_task
# def mode_track(data, it, flist=[], klist=[], log=True, write=True):
#     """
#     Plot amplification of specified modes.
#     
#     Inputs:
#         data        Data object
#         it          Iteration number
#         flist       List of fields to track: ['u', ...]
#         klist       List of wavevectors given as tuples: [(1,0,0), (1,1,1), ...]
#         log         (default True) plot a log-log plot
#         write       (default False) do not plot; instead write to mode tracking files.
#     """
#     if write:
#         for f in flist:
#             for i in xrange(data[f].ncomp):
#                 if com_sys.myproc == 0:
#                     outfile = open('mode_amplitudes_%s_%i.dat' % (f, i), 'a')
#                     amplitudes = []
#                 for k in klist:
#                     if data[f][i].find_mode(k):
#                         kampl = data[f][i]['kspace'][k]
#                     else:
#                         kampl = 0.
#                     try:
#                         tot_kampl = com_sys.comm.reduce(kampl,root=0)
#                     except AttributeError:
#                         pass
# 
#                     if com_sys.myproc == 0:
#                         amplitudes.append(abs(tot_kampl))
#                     
#                 if com_sys.myproc == 0:
#                     if it == 0:
#                         outfile.write("# Dedalus Mode Amplitudes\n")
#                         outfile.write("# Column 0: time\n")
#                         for nk,k in enumerate(klist):
#                             outfile.write("# Column %i: %s\n" % (nk, na.array(k)))
#                     outstring = '\t'.join([str(i) for i in amplitudes])
#                     outfile.write("%s\t%s\n" % (data.time,outstring))
#                     outfile.close()
#                 
#         return
#     if it == 0:
#         # Construct container on first pass
#         data._save_modes = {}
# 
#         for f in flist:
#             for i in xrange(data[f].ncomp):
#                 for k in klist:
#                     data._save_modes[(f,i,k)] = [data[f][i]['kspace'][k[::-1]]]
#         data._save_modes['time'] = [data.time]
#         return
#         
#     # Save components and time
#     for f in flist:
#         for i in xrange(data[f].ncomp):
#             for k in klist:
#                 data._save_modes[(f,i,k)].append(data[f][i]['kspace'][k[::-1]])
#     data._save_modes['time'].append(data.time)
#     
#     # Determine image grid size
#     nrow = len(flist)
#     ncol = na.max([data[f].ncomp for f in flist])
#     fig, axs = P.subplots(nrow, ncol, num=3, figsize=(8 * ncol, 6 * nrow)) 
# 
#     # Plot field components
#     time = na.array(data._save_modes['time'])
#         
#     for j,f in enumerate(flist):
#         for i in xrange(data[f].ncomp):
#             for k in klist:
#                 plot_array = na.array(data._save_modes[(f,i,k)])
#                 power = 0.5 * na.abs(plot_array) ** 2
#                 
#                 if log:
#                     axs[j, i].semilogy(time, power, '.-', label=str(k))
#                 else:
#                     axs[j, i].plot(time, power, '.-', label=str(k))   
#     
#             # Pad and label axes
#             axs[j, i].axis(padrange(axs[j, i].axis(), 0.05))
#             axs[j, i].legend()
#             axs[j, i].set_title(f + str(i))
#                             
# 
#     axs[-1, 0].set_ylabel('power')
#     axs[-1, 0].set_xlabel('time')
# 
#     if not os.path.exists('frames'):
#         os.mkdir('frames')
#     outfile = "frames/mode_track.png"
#     fig.savefig(outfile)
#     fig.clf()
#     
# @AnalysisSet.register_task
# def scatter_snapshot(data, it, space='kspace', axis='z', index='middle', figproc=0):
#     """
#     Take scatterplot snapshot of data in physical configuration.
#     
#     Parameters
#     ----------
#     data : StateData object
#         Data input
#     it : int
#         Iteration number
#     space: str
#         'xspace' or 'kspace'
#     axis : str, optional
#         Axis normal to desired slice, defaults to 'z'. Ignored for 2D data.
#         i.e. 'x' for a y-z plane
#              'y' for a x-z plane
#              'z' for a x-y plane
#     index : int or string, optional
#         Index for slicing as an integer, or 'top', 'middle' (default), or 'bottom'.
#         Ignored for 2D data.
#     figproc : int
#         Processor to handle figure creation and drawing
# 
#     """
# 
#     # Figure setup
#     if com_sys.myproc == figproc:
#     
#         # Determine grid size
#         nrows = len(data.fields.keys())
#         ncols = na.max([f.ncomp for f in data.fields.values()])
#         
#         # Create figure and axes grid
#         fig = P.figure(4, figsize=(8 * ncols, 8 * nrows))
#         grid = AxesGrid(fig, 111,
#                         nrows_ncols = (nrows, ncols),
#                         aspect=False,
#                         share_all=True,
#                         axes_pad=0.3,
#                         cbar_pad=0.,
#                         label_mode="L",
#                         cbar_location="top",
#                         cbar_mode="each")
#                     
#     # Plot field components
#     row = -1
#     firstplot = True
#     for fname, field in data:
#         row += 1
#         for cindex, comp in field:
#             
#             # Retrieve correct slice
#             if firstplot:
#                 plane_data, outindex, name0, x0, name1, x1 = get_plane(data, 
#                         fname, cindex, space=space, axis=axis, index=index)
#             else:
#                 plane_data, outindex = get_plane(data, fname, cindex, space=space,
#                         axis=axis, index=index, return_position_arrays=False)
#             if com_sys.myproc != figproc:
#                 firstplot = False
#                 continue
#             
#             # Plot
#             axnum = row * ncols + cindex
#             if space == 'kspace':
#                 # Take logarithm of magnitude, flooring at eps
#                 kmag = na.abs(plane_data)
#                 zero_mask = (kmag == 0)    
#                 kmag[kmag < comp._eps['kspace']] = comp._eps['kspace']
#                 logkmag = na.log10(kmag)
#                     
#                 # Plot nonzero
#                 if na.sum(~zero_mask):
#                     im = grid[axnum].scatter(x0[~zero_mask], x1[~zero_mask], 
#                             c=logkmag[~zero_mask], lw=0, s=40, zorder=2)
#                     grid.cbar_axes[axnum].colorbar(im)
#                 # Plot zeros
#                 grid[axnum].scatter(x0[zero_mask], x1[zero_mask], c='w', s=40, zorder=2)
#                     
#             elif space == 'xspace':
#                 im = grid[axnum].scatter(x0, x1, c=plane_data, lw=0, s=40, zorder=2)
#                 grid.cbar_axes[axnum].colorbar(im)    
#             
#             # Lines
#             if space == 'kspace':
#                 # Zero lines
#                 grid[axnum].axhline(0, c='k', zorder=1)
#                 grid[axnum].axvline(0, c='k', zorder=1)
#                 
#                 # Nyquist boundary
#                 if firstplot:
#                     ny0 = comp.kny[comp.ktrans[name0[1]]]
#                     ny1 = comp.kny[comp.ktrans[name1[1]]]
#                     if name0 == 'kx':
#                         nysquare0 = na.array([0, ny0, ny0, 0])
#                         nysquare1 = na.array([ny1, ny1, -ny1, -ny1])
#                     else:
#                         nysquare0 = na.array([-ny0, ny0, ny0, -ny0, -ny0])
#                         nysquare1 = na.array([ny1, ny1, -ny1, -ny1, ny1])
#                 grid[axnum].plot(nysquare0, nysquare1, 'k--', zorder=1)
#                 
#                 # Dealiasing boundary
#                 grid[axnum].plot(2./3. * nysquare0, 2./3. * nysquare1, 'k:', zorder=1)
#                 
#                 plot_extent = [nysquare0[0], nysquare0[1], nysquare1[2], nysquare1[0]]
#             
#             elif space == 'xspace':
#                 # Real space boundary
#                 if firstplot:
#                     x0len = comp.length[comp.xtrans[name0]]
#                     x1len = comp.length[comp.xtrans[name1]]
#                     dx0 = x0[0, 1] - x0[0, 0]
#                     dx1 = x1[1, 0] - x1[0, 0]
#                     xsquare0 = na.array([0, x0len, x0len, 0, 0]) - dx0 / 2.
#                     xsquare1 = na.array([x1len, x1len, 0, 0, x1len]) - dx1 / 2.
#                 grid[axnum].plot(xsquare0, xsquare1, 'k', zorder=1)
#                 
#                 plot_extent = [xsquare0[0], xsquare0[1], xsquare1[2], xsquare1[0]]
#                 
#             # Plot range and labels
#             grid[axnum].axis(padrange(plot_extent, 0.1))
#             grid[axnum].text(0.05, 0.95, fname + field.ctrans[cindex], 
#                     transform=grid[axnum].transAxes, size=24, color='k')
#             if row == nrows - 1:
#                 grid[axnum].set_xlabel(name0)
#             if cindex == 0:
#                 grid[axnum].set_ylabel(name1)
#             
#             if firstplot:
#                 firstplot = False
#                 
#     # Bail if not figproc
#     if com_sys.myproc != figproc:
#         return
#             
#     # Add time to figure title
#     tstr = 't = %6.3f' % data.time
#     fig.suptitle(tstr, size=24, color='k')
#     
#     # Save in frames folder
#     if not os.path.exists('frames'):
#         os.mkdir('frames')
#     if data.ndim == 2:
#         slicestr = ''
#     elif data.ndim == 3:
#         slicestr = '%s_%i_' %(axis, outindex)
#     outfile = "frames/%s_scatter_%sn%07i.png" %(space[0], slicestr, it)
#     fig.savefig(outfile)
#     fig.clf()
    
def padrange(range, pad=0.1):
    """Pad a list of the form [x0, x1, y0, y1] by specified fraction."""
    
    if pad == 0.:
        return range
    
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
    
     

