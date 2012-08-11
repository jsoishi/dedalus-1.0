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

import os
import numpy as na
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as P
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import AxesGrid
from dedalus.utils.parallelism import com_sys, get_plane

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
    Save image of specified plane in data.
    
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
                            self.update_image(axnum, plane_data, namex, namey, space, units)
                              
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
        if space == 'kspace':
            pc.set_clim(plane_data.min(), plane_data.max())
        else:
            lim = na.max(na.abs([plane_data.min(), plane_data.max()]))
            pc.set_clim(-lim, lim)
        
    def add_image(self, axnum, x, y, plane_data, units, space, comp, namex, namey):

        # Construct image
        if units:
            if space == 'kspace':
                dx = x[0, 1] - x[0, 0]
                dy = y[1, 0] - y[0, 0]
                extent = [x.min() - dx / 2., x.max() + dx / 2., 
                          y.min() - dy / 2.,y.max() + dy / 2.]
                          
                # Roll array
                if namey != 'kx':
                    plane_data = na.roll(plane_data, -(plane_data.shape[0] / 2 + 1), axis=0)
                if namex != 'kx':
                    plane_data = na.roll(plane_data, -(plane_data.shape[1] / 2 + 1), axis=1)
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
        
    def update_image(self, axnum, plane_data, namex, namey, space, units):
    
        # Retrieve image
        im = self.images[axnum]
        
        if units and space == 'kspace':
            # Roll array
            if namey != 'kx':
                plane_data = na.roll(plane_data, -(plane_data.shape[0] / 2 + 1), axis=0)
            if namex != 'kx':
                plane_data = na.roll(plane_data, -(plane_data.shape[1] / 2 + 1), axis=1)
       
        # Update values and colorbar     
        im.set_array(plane_data)
        if space == 'kspace':
            im.set_clim(plane_data.min(), plane_data.max())
        else:
            lim = na.max(na.abs([plane_data.min(), plane_data.max()]))
            im.set_clim(-lim, lim)
        
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
    Record complex amplitude of specified modes to text file.
    
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
                    file = open('%s_mode_amplitudes.dat' %name, 'w')
                    file.write("# Dedalus Mode Amplitudes\n")
                    file.write(columnnames)
                    file.write('\n')
                    file.close()

    def run(self, data, it, fieldlist=None, modelist=[], indexlist=[]):
            
        if fieldlist is None:
            fieldlist = data.fields.keys()
            
        for fname in fieldlist:
            field = data[fname]
            for cindex, comp in field:
                if com_sys.myproc == 0:
                    amplitudes = []
                
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
                    
                    name = fname + field.ctrans[cindex]
                    file = open('%s_mode_amplitudes.dat' %name, 'a')
                    file.write(tstring + ampstring)
                    file.write('\n')
                    file.close()
                    
class PowerSpectrum(AnalysisTask):
    """
    Save and plot power spectrum of specified field.

    Keywords
    --------
    fieldlist : None or list of strings
        List containing names of fields to track: ['u', ...]. If None, all fields
        in data will be tracked.
    norm : int    
        Power in each mode is multiplied by normalization
    loglog : bool
        Plot log-log plot if True, semilogy otherwise.
    plot : bool
        Save plot of power spectra.
    write : bool
        Write power spectra to text file.
    nyquistlines : bool
        Plot lines at individual and composite Nyquist wavenumbers
    dealiasinglines : bool
        Plot lines at 2/3 of the individual and composite Nyquist wavenumbers
        
    """
    
    def setup(self, data, it, fieldlist=None, plot=True, write=True, **kwargs):
    
        if fieldlist is None:
            fieldlist = data.fields.keys()

        if com_sys.myproc == 0:
            self.first = True
            self.lines = {}
            if plot:
                # Create figure and axes grid
                ncols = len(fieldlist)
                self.fig, self.axes = P.subplots(1, ncols, num=self._n, 
                        figsize=(6 * ncols, 6))
                                         
                # Directory setup
                if not os.path.exists('frames'):
                    os.mkdir('frames')
                    
            if write:
                # Create file for each field
                for fname in fieldlist:
                    file = open('%s_power_spectra.dat' %fname, 'w')
                    file.write("# Dedalus Power Spectrum\n")
                    file.close()
    
    def run(self, data, it, fieldlist=None, norm=1.0, loglog=True, plot=True, 
                write=True, nyquistlines=False, dealiasinglines=True):

        if fieldlist is None:
            fieldlist = data.fields.keys()

        for i, fname in enumerate(fieldlist):
            field = data[fname]
            
            # Compute spectrum
            k, spectrum = self.compute_spectrum(field, norm=norm)
    
            if com_sys.myproc == 0:
                if plot:
                    # Skip if all modes are zero
                    if spectrum[1:].nonzero()[0].size == 0:
                        continue                 
                                
                    # Plot
                    if len(fieldlist) == 1:
                        ax = self.axes
                    else:
                        ax = self.axes[i]
                        
                    if self.first:
                        if loglog:
                            linelist = ax.loglog(k[1:], spectrum[1:], 'b.-', mew=0, ms=5)
                        else:
                            linelist = ax.semilogy(k[1:], spectrum[1:], 'b.-', mew=0, ms=5)
                        self.lines[i] = linelist[0]
                        ax.set_title(fname, size=16, color='k')
                        ax.set_xlabel('k')
                        ax.set_ylabel('E(k)')
                        for kny in field[0].kny:
                            if nyquistlines:
                                ax.axvline(kny, ls='dashed', color='k')
                            if dealiasinglines:
                                ax.axvline(kny * 2. / 3., ls='dotted', color='k')
                        if nyquistlines:
                            ax.axvline(na.sqrt(na.sum(field[0].kny ** 2)), ls='dashed', color='r') 
                        if dealiasinglines:
                            ax.axvline(na.sqrt(na.sum((2./3. * field[0].kny) ** 2)), ls='dotted', color='r')
                    
                    else:
                        line = self.lines[i]
                        line.set_ydata(spectrum[1:])
                        ax.relim()
                        ax.autoscale_view()
                                     
                if write:
                    file = open('%s_power_spectra.dat' %fname, 'a')
                    if self.first:
                        columnnames = 'time\t'
                        columnnames += '\t'.join([repr(ki) for ki in k])
                        file.write(columnnames)
                        file.write('\n')
                    tstring = '%s\t' %data.time
                    ampstring = '\t'.join([repr(amp) for amp in spectrum])
                    file.write(tstring + ampstring)
                    file.write('\n')
                    file.close()
               
        if com_sys.myproc == 0:
            if plot:
                # Add time to figure title
                tstr = 't = %6.3f' % data.time
                if self.first:
                    self.timestr = self.fig.suptitle(tstr, size=20, color='k')
                    self.first = False
                else:
                    self.timestr.set_text(tstr)
                
                # Save in frames folder
                outfile = "frames/power_spectra_n%07i.png" %it
                self.fig.savefig(outfile)
                
            if self.first:
                self.first = False

    def compute_spectrum(self, field, norm=1.0):

        # Compute 1D power samples
        kmag = na.sqrt(field[0].k2())
        power = na.zeros(field[0].local_shape['kspace'])
        for cindex, comp in field:
            power += na.abs(comp['kspace']) ** 2
        power *= norm
        if field.ndim == 2:
            power1d = power * 2. * na.pi * kmag
        else:
            power1d = power * 4. * na.pi * kmag ** 2
    
        # Construct bins by wavevector magnitude (evenly spaced)
        kmax = na.sqrt(na.sum(field[0].kny ** 2))
        n = int(na.product(field[0].global_shape['xspace'] / 2.) ** (1. / field.ndim))
        k = na.linspace(0, kmax, n, endpoint=False)
        kbottom = k
        ktop = k + k[1]
        
        # Bin power samples
        spectrum = na.zeros_like(k)
        n = na.zeros_like(k)
        for i in xrange(k.size):
            mask = (kmag >= kbottom[i]) & (kmag < ktop[i])
            nonzero = (power1d != 0)
            spectrum[i] = na.sum(power1d[mask])
            n[i] = na.sum(mask * nonzero)
        
        # Collect from all processes
        if com_sys.nproc != 1:
            spectrum = com_sys.comm.reduce(spectrum, op=com_sys.MPI.SUM, root=0)
            n = com_sys.comm.reduce(n, op=com_sys.MPI.SUM, root=0)
            
            if com_sys.myproc != 0:
                return (None, None)
        
        # Return bin centers and 1D power sample averages
        n[n == 0] = 1
        return k + k[1] / 2. , spectrum / n

class VolumeAverage(AnalysisTask):
    """Run volume average tasks."""
    
    def run(self, data, it, va_obj=None):
        va_obj.run()
       


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
