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
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from dedalus.utils.parallelism import com_sys, get_plane
from dedalus.config import decfg
import time

class AnalysisSet(object):

    def __init__(self, data, ti):
        self.data = data
        self.ti = ti
        self.tasks = []

    def add(self, task):
        self.tasks.append(task)
        task._n = len(self.tasks)
        task.setup(self.data, self.ti.iter)

    def run(self):
        for task in self.tasks:
            if self.ti.iter % task.cadence == 0:
                task.run(self.data, self.ti.iter)
                
    def cleanup(self):
        for task in self.tasks:
            task.cleanup(self.data, self.ti.iter)
              
class AnalysisTask(object):

    def __init__(self, cadence, **kwargs):
        self.cadence = cadence
        self.options = kwargs
        
    def setup(self, data, iter):
        pass
        
    def run(self, data, iter):
        pass
        
    def cleanup(self, data, iter):
        pass
        
class Snapshot(AnalysisTask):

    def __init__(self, cadence, space=None, axis=None, index=None, units=None):
        """
        Save image of specified plane in data.
        
        Parameters
        ----------
        cadence : int
            Iteration cadence for running task.
        space: str, optional
            'xspace' or 'kspace'
        axis : str, optional
            Axis normal to desired slice. Ignored for 2D data.
            i.e. 'x' for a y-z plane
                 'y' for a x-z plane
                 'z' for a x-y plane
        index : int or string, optional
            Index for slicing as an integer, or 'top', 'middle', or 'bottom'.
            Ignored for 2D data.
        units : boolean, optional
            True for physical-space layout and units.
            False for array-based layout and indexing.
            
        Notes
        -----
        Any unspecified keywords will have their values taken from the config.
    
        """
    
        # Store inputs
        self.cadence = cadence
        
        if space is None:
            self.space = decfg.get('analysis', 'snapshot_space')
        else:
            self.space = space
            
        if axis is None:
            self.axis = decfg.get('analysis', 'snapshot_axis')
        else:
            self.axis = axis

        if index is None:
            self.index = decfg.get('analysis','snapshot_index')
        else:
            self.index = index
        
        if units is None:
            self.units = decfg.getboolean('analysis', 'snapshot_units')
        else:
            self.units = units
    
    def setup(self, data, it):

        # Determine if moving patches are required
        self.firstrun = True
        if self.units and (self.space == 'kspace') and hasattr(data['u']['x'], '_ky'):
            self._moves = True
            if com_sys.myproc == 0:
                self.patch_lists = {}
                self.patch_collections = {}
        else:
            self._moves = False
            if com_sys.myproc == 0:
                self.images = {}
        
        # Figure setup for proc 0
        if com_sys.myproc == 0:
        
            self.image_axes = {}
            self.cbar_axes = {}
            
            # Determine grid size
            nrows = len(data.fields.keys())
            ncols = na.max([field.ncomp for field in data.fields.values()])
            
            # Setup spacing [top, bottom, left, right] and [height, width]
            t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.2, 0.2)
            t_pad, b_pad, l_pad, r_pad = (0.15, 0.03, 0.03, 0.03)
            h_cbar, w_cbar = (0.05, 1.)
            h_data, w_data = (1., 1.)
            
            h_im = t_pad + h_cbar + h_data + b_pad
            w_im = l_pad + w_data + r_pad
            h_total = t_mar + nrows * h_im + b_mar
            w_total = l_mar + ncols * w_im + r_mar
            scale = 4.0
            
            # Create figure and axes
            self.fig = plt.figure(self._n, figsize=(scale * w_total, scale * h_total))
            for row, (fname, field) in enumerate(data):
                for cindex, comp in field:
                    left = (l_mar + w_im * cindex + l_pad) / w_total
                    bottom = 1 - (t_mar + h_im * (row + 1) - b_pad) / h_total
                    width = w_data / w_total
                    height = h_data / h_total
                    self.image_axes[(row, cindex)] = self.fig.add_axes([left, bottom, width, height])
                    self.image_axes[(row, cindex)].lastrow = (row == nrows - 1)
                    self.image_axes[(row, cindex)].firstcol = (cindex == 0)
                                        
                    left = (l_mar + w_im * cindex + l_pad) / w_total
                    bottom = 1 - (t_mar + h_im * row + t_pad + h_cbar) / h_total
                    width = w_cbar / w_total
                    height = h_cbar / h_total
                    self.cbar_axes[(row, cindex)] = self.fig.add_axes([left, bottom, width, height])
        
            # Title
            height = 1 - (0.6 * t_mar) / h_total
            self.timestring = self.fig.suptitle(r'', y=height, fontsize=16)
        
            # Directory setup
            if not os.path.exists('frames'):
                os.mkdir('frames')
    
    def run(self, data, it):

        # Plot field components
        for row, (fname, field) in enumerate(data):
            for cindex, comp in field:
                
                # Retrieve correct slice
                if (row == 0) and (cindex == 0) and (self.firstrun or self._moves):
                    packed_data = get_plane(comp, space=self.space, axis=self.axis, index=self.index)
                    plane_data, outindex, self.namex, x, self.namey, y = packed_data
                else:
                    plane_data, outindex = get_plane(comp, space=self.space, axis=self.axis, 
                            index=self.index, return_position_arrays=False)

                if com_sys.myproc == 0:
                    # kspace: take logarithm of magnitude, flooring at eps
                    if self.space == 'kspace':
                        kmag = na.abs(plane_data)
                        zero_mask = (kmag == 0)
                        kmag[kmag < comp._eps['kspace']] = comp._eps['kspace']
                        kmag = na.ma.array(kmag, mask=zero_mask)
                        plane_data = na.log10(kmag)
    
                    # Plot
                    axtup = (row, cindex)
                    start = time.time()
                    if self.firstrun:
                        if self._moves:
                            self.add_patches(axtup, x, y, plane_data)
                        else:
                            self.add_image(axtup, x, y, plane_data)
                        self.add_lines(axtup, x, y, plane_data, comp)
                        self.add_labels(axtup, fname, field.ctrans[cindex], row, cindex)
                    else:
                        if self._moves:
                            self.update_patches(axtup, x, y, plane_data)
                        else:
                            self.update_image(axtup, plane_data)

                              
        if com_sys.myproc == 0:
            # Change figure title
            tstr = r'$t = %6.3f$' % data.time
            self.timestring.set_text(tstr)            
            
            # Save in frames folder
            spacestr = self.space[0]
            if not self.units:
                spacestr += '_ind'
            if data.ndim == 2:
                slicestr = ''
            else:
                slicestr = '%s_%i_' %(self.axis, outindex)
            outfile = "frames/%s_snap_%sn%07i.png" %(spacestr, slicestr, it)
            self.fig.savefig(outfile, dpi=100)
            
        if self.firstrun:
            self.firstrun = False
                                  
    def add_patches(self, axtup, x, y, plane_data):
    
        imax = self.image_axes[axtup]
        cbax = self.cbar_axes[axtup]
        
        # Construct patches
        shape = plane_data.shape
        patches = []
        
        dx = x[0, 1] - x[0, 0]
        dy = y[1, 0] - y[0, 0]
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                xy = (x[i, j] - dx / 2., y[i, j] - dy / 2.)
                rect = Rectangle(xy, dx, dy)
                patches.append(rect) 
             
        # Set values and colorbar     
        cmap = matplotlib.cm.Spectral_r
        cmap.set_bad('0.7')
        pc = PatchCollection(patches, cmap=cmap, zorder=1, edgecolors='white') 
        pc.set_array(na.ma.ravel(plane_data))
        pc.set_clim(plane_data.min(), plane_data.max())
        imax.add_collection(pc)
        self.fig.colorbar(pc, cax=cbax, orientation='horizontal',
                ticks=MaxNLocator(nbins=5, prune='both'))
                
        # Store for updating        
        self.patch_lists[axtup] = patches
        self.patch_collections[axtup] = pc
        
    def update_patches(self, axtup, x, y, plane_data):
    
        # Retrieve patches
        shape = plane_data.shape
        patches = self.patch_lists[axtup]
        pc = self.patch_collections[axtup]
        
        # Update positions
        dx = x[0, 1] - x[0, 0]
        dy = y[1, 0] - y[0, 0]
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                xy = (x[i, j] - dx / 2., y[i, j] - dy / 2.)
                patches[i * shape[1] + j].set_xy(xy)
        pc.set_paths(patches)
           
        # Update values and colorbar     
        pc.set_array(na.ma.ravel(plane_data))
        pc.set_clim(plane_data.min(), plane_data.max())
        
    def add_image(self, axtup, x, y, plane_data):
    
        imax = self.image_axes[axtup]
        cbax = self.cbar_axes[axtup]

        # Construct image
        if self.units:
            dx = x[0, 1] - x[0, 0]
            dy = y[1, 0] - y[0, 0] 
            if self.space == 'kspace':
                # Roll array
                if self.namey != 'kx':
                    plane_data = na.roll(plane_data, -(plane_data.shape[0] / 2 + 1), axis=0)
                if self.namex != 'kx':
                    plane_data = na.roll(plane_data, -(plane_data.shape[1] / 2 + 1), axis=1)
                    
                extent = [x.min() - dx / 2., x.max() + dx / 2., 
                          y.min() - dy / 2., y.max() + dy / 2.]
            else:
                extent = [x.min(), x.max() + dx,
                          y.min(), y.max() + dy]
        else:
            extent = None
        
        cmap = matplotlib.cm.Spectral_r
        cmap.set_bad('0.7')
        im = imax.imshow(plane_data, cmap=cmap, zorder=1, aspect='auto', 
                interpolation='nearest', origin='lower', extent=extent)   
        self.fig.colorbar(im, cax=cbax, orientation='horizontal',
                ticks=MaxNLocator(nbins=5, prune='both'))

        # Store for updating        
        self.images[axtup] = im
        
    def update_image(self, axtup, plane_data):
    
        # Retrieve image
        im = self.images[axtup]
        
        if self.units and (self.space == 'kspace'):
            # Roll array
            if self.namey != 'kx':
                plane_data = na.roll(plane_data, -(plane_data.shape[0] / 2 + 1), axis=0)
            if self.namex != 'kx':
                plane_data = na.roll(plane_data, -(plane_data.shape[1] / 2 + 1), axis=1)
       
        # Update values and colorbar     
        im.set_array(plane_data)
        if self.space == 'kspace':
            im.set_clim(plane_data.min(), plane_data.max())
        else:
            lim = na.max(na.abs([plane_data.min(), plane_data.max()]))
            im.set_clim(-lim, lim)
        
    def add_lines(self, axtup, x, y, plane_data, comp):
    
        imax = self.image_axes[axtup]
        
        if self.units:            
            if self.space == 'kspace':
                dx = x[0, 1] - x[0, 0]
                dy = y[1, 0] - y[0, 0]
            
                # Zero lines
                imax.axhline(0, c='k', zorder=2, lw=2)
                imax.axvline(0, c='k', zorder=2, lw=2)
                
                # Dealiasing boundary
                nyx = comp.kny[comp.ktrans[self.namex[1]]]
                nyy = comp.kny[comp.ktrans[self.namey[1]]]
                if self.namex == 'kx':
                    xsq = na.array([0, nyx, nyx, 0, 0])
                    ysq = na.array([nyy, nyy, -nyy, -nyy, nyy])
                else:
                    xsq = na.array([-nyx, nyx, nyx, -nyx, -nyx])
                    ysq = na.array([nyy, nyy, -nyy, -nyy, nyy])
                xsh = na.array([-dx / 2., dx / 2., dx / 2., -dx / 2., -dx / 2.])
                ysh = na.array([dy / 2., dy / 2., -dy / 2., -dy / 2., dy / 2.])
                imax.plot(2./3. * xsq + xsh, 2./3. * ysq + ysh, 
                        'k--', zorder=2, lw=2)
                
                plot_extent = [xsq[0] - dx / 2., xsq[1] + dx / 2., 
                               ysq[2] - dy / 2., ysq[0] + dy / 2.]
            
            else:
                xlen = comp.length[comp.xtrans[self.namex]]
                ylen = comp.length[comp.xtrans[self.namey]]
                plot_extent = [0, xlen, 0, ylen]
        
        else:
            shape = plane_data.shape
            plot_extent = [-0.5, shape[1] - 0.5, -0.5, shape[0] - 0.5]
        
        # Plot range
        imax.axis(plot_extent)

    def add_labels(self, axtup, fname, cname, row, cindex):
    
        imax = self.image_axes[axtup]
        cbax = self.cbar_axes[axtup]

        # Title
        title = imax.set_title(r'$%s_{%s}$' %(fname, cname), fontsize=14)
        title.set_y(1.1)
        
        # Colorbar
        cbax.xaxis.set_ticks_position('top')
        plt.setp(cbax.get_xticklabels(), fontsize=10)
        
        # Axis labels
        if self.space == 'kspace':
            xstr = r'$k_{%s} \; \mathrm{index}$' %self.namex[1]
            ystr = r'$k_{%s} \; \mathrm{index}$' %self.namey[1]
        else:
            xstr = r'$%s$' %self.namex
            ystr = r'$%s$' %self.namey
            
        if imax.lastrow:
            imax.set_xlabel(xstr)
            plt.setp(imax.get_xticklabels(), fontsize=10)
        else:
            plt.setp(imax.get_xticklabels(), visible=False)
            
        if imax.firstcol:
            imax.set_ylabel(ystr)
            plt.setp(imax.get_yticklabels(), fontsize=10)
        else:
            plt.setp(imax.get_yticklabels(), visible=False)            

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
                self.fig, self.axes = plt.subplots(1, ncols, num=self._n, 
                        figsize=(6 * ncols, 6), squeeze=False)
                                         
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
                    ax = self.axes[0, i]
                        
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
    
    def __init__(self, iter, va_obj):
        AnalysisTask.__init__(self, iter)
        self.va_obj=va_obj
    def run(self, data, it):
        self.va_obj.run()
       


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
