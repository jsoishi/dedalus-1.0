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
import os
from dedalus.utils.parallelism import com_sys, get_plane
from dedalus.utils.logger import mylog

class AnalysisSet(object):
    def __init__(self, data, ti):
        self.data = data
        self.ti = ti
        self.tasks = []

    def add(self, task):
        self.tasks.append(task)
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
    units : boolean
        True for physical-space layout and units.
        False for array-based layout and indexing.

    """
    
    def setup(self, data, it, space='xspace', units=True, **kwargs):
    
        # Determine if moving patches are required
        if units and space == 'kspace' and hasattr(data['u']['x'], '_ky'):
            self._moves = True
            if com_sys.myproc == 0:
                self.patch_lists = {}
                self.patch_collections = {}
        else:
            self._moves = False
            if com_sys.myproc == 0:
                self.images = {}
    
        # Determine grid size
        self.nrows = len(data.fields.keys())
        self.ncols = na.max([f.ncomp for f in data.fields.values()])
        
        # Figure setup for proc 0
        if com_sys.myproc == 0:

            # Create figure and axes grid
            self.firstplot = True
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
    
    def run(self, data, it, space='xspace', axis='z', index='middle', units=True):

        # Plot field components
        for row, (fname, field) in enumerate(data):
            for cindex, comp in field:
                axnum = row * self.ncols + cindex
                
                # Retrieve correct slice
                if axnum == 0:
                    plane_data, outindex, namex, x, namey, y = get_plane(comp, 
                            space=space, axis=axis, index=index)
                else:
                    plane_data, outindex = get_plane(comp, space=space, axis=axis, 
                            index=index, return_position_arrays=False)

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
                        if self._moves:
                            self.add_patches(axnum, x, y, plane_data, units, space)
                        else:
                            self.add_image(axnum, x, y, plane_data, units, space, 
                                    comp, namex, namey)
                        self.add_lines(axnum, x, y, plane_data, units, space, 
                                comp, namex, namey)
                        self.add_labels(axnum, fname, field.ctrans[cindex],
                                namex, namey, row, cindex, units)
                    else:
                        if self._moves:
                            self.update_patches(axnum, x, y, plane_data, units, space)
                        else:
                            self.update_image(axnum, plane_data)
                              
        if com_sys.myproc == 0:
            # Add time to figure title
            tstr = 't = %6.3f' % data.time
            if self.firstplot:
                self.timestr = self.fig.suptitle(tstr, size=24, color='k')
                self.firstplot = False
            else:
                self.timestr.set_text(tstr)
            
            # Save in frames folder
            spacestr = space[0]
            if not units:
                spacestr += '_ind'
            if data.ndim == 2:
                slicestr = ''
            else:
                slicestr = '%s_%i_' %(axis, outindex)
            outfile = "frames/%s_snap_%sn%07i.png" %(spacestr, slicestr, it)
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
        if space == 'kspace':
            pc = PatchCollection(patches, cmap=cmap, zorder=1, edgecolors='white') 
        else:
            pc = PatchCollection(patches, cmap=cmap, zorder=1, lw=0)

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
        
    def add_image(self, axnum, x, y, plane_data, units, space, comp, namex, namey):

        # Construct image
        if units:
            if space == 'kspace':
                dx = x[0, 1] - x[0, 0]
                dy = y[1, 0] - y[0, 0]
                          
                # Extend to Nyquist mode even if not present
                nyx = comp.kny[comp.ktrans[namex[1]]]
                nyy = comp.kny[comp.ktrans[namey[1]]]
                if namex == 'kx':
                    xsq = na.array([0, nyx, nyx, 0, 0])
                    ysq = na.array([nyy, nyy, -nyy, -nyy, nyy])
                else:
                    xsq = na.array([-nyx, nyx, nyx, -nyx, -nyx])
                    ysq = na.array([nyy, nyy, -nyy, -nyy, nyy])
                extent = [xsq[0] - dx / 2., xsq[1] + dx / 2., 
                          ysq[2] - dy / 2., ysq[0] + dy / 2.]
            else:
                xlen = comp.length[comp.xtrans[namex]]
                ylen = comp.length[comp.xtrans[namey]]
                extent = [0, xlen, 0, ylen]
        else:
            extent = None
        
        cmap = matplotlib.cm.Spectral_r
        cmap.set_bad('0.7')
        im = self.grid[axnum].imshow(plane_data, cmap=cmap, zorder=1, aspect='auto', 
                interpolation='nearest', origin='lower', extent=extent)
        self.grid.cbar_axes[axnum].colorbar(im)

        # Store for updating        
        self.images[axnum] = im
        
    def update_image(self, axnum, plane_data):
    
        # Retrieve image
        im = self.images[axnum]
           
        # Update values and colorbar     
        im.set_array(plane_data)
        im.set_clim(plane_data.min(), plane_data.max())
        
    def add_lines(self, axnum, x, y, plane_data, units, space, comp, namex, namey):
        
        if units:            
            if space == 'kspace':
                dx = x[0, 1] - x[0, 0]
                dy = y[1, 0] - y[0, 0]
            
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
        
        else:
            shape = plane_data.shape
            plot_extent = [-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5]
        
        # Plot range
        self.grid[axnum].axis(plot_extent)

    def add_labels(self, axnum, fname, cname, namex, namey, row, cindex, units):

        # Title
        title = self.grid[axnum].set_title(fname + cname, size=20, color='k')
        title.set_y(1.08)
        
        # Axis labels
        if not units:
            namex += ' index'
            namey += ' index'
        if row == self.nrows - 1:
            self.grid[axnum].set_xlabel(namex)
        if cindex == 0:
            self.grid[axnum].set_ylabel(namey)                    

class TrackMode(AnalysisTask):
    """
    Record complex amplitude of specified modes.
    
    Keywords
    --------
    fieldlist : None or list of strings
        List containing names of fields to track: ['u', ...]. If None, all fields
        in data will be tracked.
    modelist : list of tuples of floats
        List containing physical wavevectors to track: [(0., 0., -3.), ...]
    indexlist : None or list of tuples of ints
        List containing local kspace indices to track : [(1, 0, 0), ...]
        None should be passed to all processors without the desired mode.
    
    Notes
    -----        
    Keep in mind that for parallelism the data layouts in k-space when specifying
    modes and indices:

    In 3D: y, z, x
    In 2D: x, y
    
    """
    
    def setup(self, data, it, fieldlist=None, modelist=[], indexlist=[]):
    
        if fieldlist is None:
            fieldlist = data.fields.keys()
                              
        # Construct string defining columns
        if com_sys.myproc == 0:
            columnnames = 'time'
            for mode in modelist:
                columnnames += '\t' + str(list(mode))
        for index in indexlist:
            if index is None:
                column = ''
            else:
                gindex = (index[0] + data['u']['x'].offset['kspace'],) + index[1:]
                column = str(gindex)
            column = com_sys.comm.reduce(column, root=0)
            if com_sys.myproc == 0:
                columnnames += '\t' + column

        if com_sys.myproc == 0:
            # Create file for each field component
            for fname in fieldlist:
                field = data[fname]
                for cindex, comp in field:
                    name = fname + field.ctrans[cindex]
                    outfile = open('%s_mode_amplitudes.dat' %name, 'w')
                    outfile.write("# Dedalus Mode Amplitudes\n")
                    outfile.write(columnnames)
                    outfile.write('\n')
                    outfile.close()

    def run(self, data, it, fieldlist=None, modelist=[], indexlist=[]):
            
        if fieldlist is None:
            fieldlist = data.fields.keys()
            
        for fname in fieldlist:
            field = data[fname]
            for cindex, comp in field:

                # Retrieve file
                if com_sys.myproc == 0:
                    amplitudes = []
                    name = fname + field.ctrans[cindex]
                    outfile = open('%s_mode_amplitudes.dat' %name, 'a')
                
                # Gather mode amplitudes
                for mode in modelist:
                    index = comp.find_mode(mode)
                    if index:
                        amp = comp['kspace'][index]
                    else:
                        amp = 0.
                    amp = com_sys.comm.reduce(amp, root=0)
                    if com_sys.myproc == 0:
                        amplitudes.append(amp)

                # Gather index amplitudes
                for index in indexlist:
                    if index is None:
                        amp = 0.
                    else:
                        amp = comp['kspace'][index]
                    amp = com_sys.comm.reduce(amp, root=0)
                    if com_sys.myproc == 0:
                        amplitudes.append(amp)
                
                # Write
                if com_sys.myproc == 0:
                    tstring = '%s\t' %data.time
                    ampstring = '\t'.join([repr(amp) for amp in amplitudes])
                    outfile.write(tstring + ampstring)
                    outfile.write('\n')
                    outfile.close()
                    
                    
                    
                    
                    
                    
#                     
#                     
#         if it == 0:
#             # Construct container on first pass
#             data._save_modes = {}
#     
#             for f in flist:
#                 for i in xrange(data[f].ncomp):
#                     for k in klist:
#                         data._save_modes[(f,i,k)] = [data[f][i]['kspace'][k[::-1]]]
#             data._save_modes['time'] = [data.time]
#             return
#             
#         # Save components and time
#         for f in flist:
#             for i in xrange(data[f].ncomp):
#                 for k in klist:
#                     data._save_modes[(f,i,k)].append(data[f][i]['kspace'][k[::-1]])
#         data._save_modes['time'].append(data.time)
#     
#         # Plot field components
#         time = na.array(data._save_modes['time'])
#             
#         for j,f in enumerate(flist):
#             for i in xrange(data[f].ncomp):
#                 for k in klist:
#                     plot_array = na.array(data._save_modes[(f,i,k)])
#                     power = 0.5 * na.abs(plot_array) ** 2
#                     
#                     if log:
#                         axs[j, i].semilogy(time, power, '.-', label=str(k))
#                     else:
#                         axs[j, i].plot(time, power, '.-', label=str(k))   
#         
#                 # Pad and label axes
#                 axs[j, i].axis(padrange(axs[j, i].axis(), 0.05))
#                 axs[j, i].legend()
#                 axs[j, i].set_title(f + str(i))
#                                 
#     
#         axs[-1, 0].set_ylabel('power')
#         axs[-1, 0].set_xlabel('time')
#     
#         outfile = "frames/mode_track.png"
#         fig.savefig(outfile)
#         fig.clf()
#         





 
# 
# @AnalysisSet.register_task
# def volume_average(data, it, va_obj=None):
#     va_obj.run()
#        
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
