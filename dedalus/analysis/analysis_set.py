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


class AnalysisSet(object):

    def __init__(self, data, ti):
        self.data = data
        self.ti = ti
        self.tasks = []

    def add(self, task):
        self.tasks.append(task)
        task._an = self
        task._n = len(self.tasks)
        task.setup(self.data, self.ti.iteration)

    def run(self):
        for task in self.tasks:
            if self.ti.iteration % task.cadence == 0:
                task.run(self.data, self.ti.iteration)

    def cleanup(self):
        for task in self.tasks:
            task.cleanup(self.data, self.ti.iteration)


class AnalysisTask(object):

    def __init__(self, cadence, *args, **kwargs):
        pass

    def setup(self, data, iter):
        pass

    def run(self, data, iter):
        pass

    def cleanup(self, data, iter):
        pass


class Snapshot(AnalysisTask):
    def __init__(self, cadence, space=None, axis=None, index=None, units=None,
                 dpi=None, cmap=None, even_scale=True):
        """
        Save image of specified plane in data.

        Parameters
        ----------
        cadence : int
            Iteration cadence for running task.
        space: str, optional
            'xspace' or 'kspace'. Default: config setting
        axis : str, optional
            Axis normal to desired slice, i.e.
                'x' for a y-z plane
                'y' for a x-z plane
                'z' for a x-y plane
            Ignored for 2D data. Default: config setting
        index : int or string, optional
            Index for slicing as an integer, or 'top', 'middle', or 'bottom'.
            Ignored for 2D data. Default: config setting
        units : boolean, optional
            True for physical-space layout and units.
            False for array-based layout and indexing.
            Default: config setting
        dpi : int, optional
            DPI setting for images.  Each plot is set to be 4 inches, so each
            plot in the image will be 4*DPI pixels in each direction.
            Default: config setting
        cmap : str, optional
            Name of colormap for plots. Default: config setting
        even_scale : boolean, optional
            Whether to make xspace colorbar range even. Default: True.

        """

        # Store inputs
        self.cadence = cadence
        self.space = space
        self.axis = axis
        self.index = index
        self.units = units
        self.dpi = dpi
        self.cmapname = cmap
        self.even_scale = even_scale

        # Get defaults from config
        if self.space is None:
            self.space = decfg.get('analysis', 'snapshot_space')
        if self.axis is None:
            self.axis = decfg.get('analysis', 'snapshot_axis')
        if self.index is None:
            self.index = decfg.get('analysis','snapshot_index')
        if self.units is None:
            self.units = decfg.getboolean('analysis', 'snapshot_units')
        if self.dpi is None:
            self.dpi = decfg.getint('analysis', 'snapshot_dpi')
        if self.cmapname is None:
            self.cmapname = decfg.get('analysis', 'snapshot_cmap')

    def setup(self, data, it):

        self.firstrun = True

        # Determine if moving patches are required
        if (self.units and (self.space == 'kspace') and
                not data['u']['x']._static_k):
            self._moves = True
            if com_sys.myproc == 0:
                self.patch_lists = {}
                self.patch_collections = {}
        else:
            self._moves = False
            if com_sys.myproc == 0:
                self.images = {}

        # Create local copy space for xspace snapshots
        if self.space == 'xspace':
            self._local_scalar = data._field_classes['ScalarField'](data)

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
            self.fig = plt.figure(self._n, figsize=(scale * w_total,
                                                    scale * h_total))
            for row, (fname, field) in enumerate(data):
                for cindex, comp in field:
                    left = (l_mar + w_im * cindex + l_pad) / w_total
                    bottom = 1 - (t_mar + h_im * (row + 1) - b_pad) / h_total
                    width = w_data / w_total
                    height = h_data / h_total
                    self.image_axes[(row, cindex)] = self.fig.add_axes([left,
                            bottom, width, height])
                    self.image_axes[(row, cindex)].lastrow = (row == nrows - 1)
                    self.image_axes[(row, cindex)].firstcol = (cindex == 0)

                    left = (l_mar + w_im * cindex + l_pad) / w_total
                    bottom = 1 - (t_mar + h_im * row + t_pad + h_cbar) / h_total
                    width = w_cbar / w_total
                    height = h_cbar / h_total
                    self.cbar_axes[(row, cindex)] = self.fig.add_axes([left,
                            bottom, width, height])

            # Title
            height = 1 - (0.6 * t_mar) / h_total
            self.timestring = self.fig.suptitle(r'', y=height, size=16)

            # Directory setup
            if not os.path.exists('frames'):
                os.mkdir('frames')

    def run(self, data, it):

        for row, (fname, field) in enumerate(data):
            for cindex, comp in field:

                # Copy data before transforming
                if self.space == 'xspace':
                    self._local_scalar['kspace'] = comp['kspace']
                    comp = self._local_scalar[0]

                # Retrieve correct slice
                if ((row == 0) and (cindex == 0) and
                        (self.firstrun or self._moves)):
                    packed_data = get_plane(comp, space=self.space,
                            axis=self.axis, index=self.index)
                    plane_data, outindex, namex, x, namey, y = packed_data
                    self.namex, self.namey = namex, namey
                else:
                    plane_data, outindex = get_plane(comp, space=self.space,
                            axis=self.axis, index=self.index,
                            return_position_arrays=False)

                if com_sys.myproc == 0:

                    # In kspace, take logarithm of magnitude, flooring at eps
                    if self.space == 'kspace':
                        kmag = na.abs(plane_data)
                        zero_mask = (kmag == 0)
                        kmag[kmag < comp._eps['kspace']] = comp._eps['kspace']
                        kmag = na.ma.array(kmag, mask=zero_mask)
                        plane_data = na.log10(kmag)

                    # Plot
                    axtup = (row, cindex)
                    if self.firstrun:
                        if self._moves:
                            self.add_patches(axtup, x, y, plane_data)
                        else:
                            self.add_image(axtup, x, y, plane_data)
                        self.add_lines(axtup, x, y, plane_data, comp)
                        self.add_labels(axtup, fname, field.ctrans[cindex])
                    else:
                        if self._moves:
                            self.update_patches(axtup, x, y, plane_data)
                        else:
                            self.update_image(axtup, plane_data)

        if com_sys.myproc == 0:

            # Update time title
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
            self.fig.savefig(outfile, dpi=self.dpi)

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
        cmap = matplotlib.cm.get_cmap(self.cmapname)
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
                    plane_data = na.roll(plane_data,
                            -(plane_data.shape[0] / 2 + 1), axis=0)
                if self.namex != 'kx':
                    plane_data = na.roll(plane_data,
                            -(plane_data.shape[1] / 2 + 1), axis=1)

                extent = [x.min() - dx / 2., x.max() + dx / 2.,
                          y.min() - dy / 2., y.max() + dy / 2.]
            else:
                extent = [x.min(), x.max() + dx,
                          y.min(), y.max() + dy]
        else:
            extent = None

        cmap = matplotlib.cm.get_cmap(self.cmapname)
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
                plane_data = na.roll(plane_data, -(plane_data.shape[0] / 2 + 1),
                        axis=0)
            if self.namex != 'kx':
                plane_data = na.roll(plane_data, -(plane_data.shape[1] / 2 + 1),
                        axis=1)

        # Update values and colorbar
        im.set_array(plane_data)
        if self.space == 'kspace':
            im.set_clim(plane_data.min(), plane_data.max())
        else:
            if self.even_scale:
                lim = na.max(na.abs([plane_data.min(), plane_data.max()]))
                im.set_clim(-lim, lim)
            else:
                im.set_clim(plane_data.min(), plane_data.max())

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
                imax.plot(2. / 3. * xsq + xsh, 2. / 3. * ysq + ysh,
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

    def add_labels(self, axtup, fname, cname):

        imax = self.image_axes[axtup]
        cbax = self.cbar_axes[axtup]

        # Title
        title = imax.set_title(r'$%s_{%s}$' %(fname, cname), size=14)
        title.set_y(1.1)

        # Colorbar
        cbax.xaxis.set_ticks_position('top')
        plt.setp(cbax.get_xticklabels(), size=10)

        # Axis labels
        if self.space == 'kspace':
            xstr = r'$k_{%s}$' %self.namex[1]
            ystr = r'$k_{%s}$' %self.namey[1]
        else:
            xstr = r'$%s$' %self.namex
            ystr = r'$%s$' %self.namey

        if not self.units:
            xstr += r'$\;\mathrm{index}$'
            ystr += r'$\;\mathrm{index}$'

        if imax.lastrow:
            imax.set_xlabel(xstr, size=12)
            plt.setp(imax.get_xticklabels(), size=10)
        else:
            plt.setp(imax.get_xticklabels(), visible=False)

        if imax.firstcol:
            imax.set_ylabel(ystr, size=12)
            plt.setp(imax.get_yticklabels(), size=10)
        else:
            plt.setp(imax.get_yticklabels(), visible=False)


class TrackMode(AnalysisTask):

    def __init__(self, cadence, fieldlist=None, modelist=[], indexlist=[]):
        """
        Record complex amplitude of specified modes to text file.

        Parameters
        ----------
        cadence : int
            Iteration cadence for running task.
        fieldlist : None or list of strings
            List containing names of fields to track, e.g.
                ['u', ...]
            Default: track all fields
        modelist : list of tuples of floats
            List containing physical wavevectors to track, e.g.
                [(0., 0., -3.), ...]
        indexlist : None or list of tuples of ints
            List containing *local* kspace indices to track, e.g.
                [(1, 0, 0), ...]
            None should be passed to all processors without the desired mode.

        Notes
        -----
        Keep in mind that for parallelism the data layouts in k-space when specifying
        modes and indices:

            In 3D: y, z, x
            In 2D: x, y

        """

        # Store inputs
        self.cadence = cadence
        self.fieldlist = fieldlist
        self.modelist = modelist
        self.indexlist = indexlist

    def setup(self, data, it):

        # Default to all fields in data
        if self.fieldlist is None:
            self.fieldlist = data.fields.keys()

        # Construct string defining columns
        if com_sys.myproc == 0:
            columnnames = '# time'
            for mode in self.modelist:
                columnnames += '\t' + str(list(mode))

        for index in self.indexlist:
            if index is None:
                column = ''
            else:
                gi = (index[0] + data['u']['x'].offset['kspace'],) + index[1:]
                column = str(gi)
            column = com_sys.comm.reduce(column, root=0)
            if com_sys.myproc == 0:
                columnnames += '\t' + column

        if com_sys.myproc == 0:

            # Create file for each field component
            for fname in self.fieldlist:
                field = data[fname]
                for cindex, comp in field:
                    name = fname + field.ctrans[cindex]
                    file = open('%s_mode_amplitudes.dat' %name, 'a')
                    file.write("# Dedalus Mode Amplitudes\n")
                    file.write(columnnames)
                    file.write('\n')
                    file.close()

    def run(self, data, it):

        for fname in self.fieldlist:
            field = data[fname]
            for cindex, comp in field:
                comp.require_space('kspace')
                if com_sys.myproc == 0:
                    amplitudes = []

                # Gather mode amplitudes
                for mode in self.modelist:
                    index = comp.find_mode(mode)
                    if index:
                        amp = comp['kspace'][index]
                    else:
                        amp = 0.
                    amp = com_sys.comm.reduce(amp, op=com_sys.MPI.SUM, root=0)
                    if com_sys.myproc == 0:
                        amplitudes.append(amp)

                # Gather index amplitudes
                for index in self.indexlist:
                    if index is None:
                        amp = 0.
                    else:
                        amp = comp['kspace'][index]
                    amp = com_sys.comm.reduce(amp, op=com_sys.MPI.SUM, root=0)
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

    def __init__(self, cadence, fieldlist=None, norm=1., write=True, plot=True,
                 loglog=True, nyquistlines=False, dealiasinglines=True, dpi=None):
        """
        Save and plot power spectrum of specified fields.

        Parameters
        ----------
        cadence : int
            Iteration cadence for running task.
        fieldlist : list of str, optional
            List containing names of fields to track, e.g. ['u', ...].
            Default: None ==> track all fields
        norm : int, optional
            Scaling factor for power in each mode. Default: 1.0
        write : bool, optional
            Write power spectra to text file. Default: True
        plot : bool, optional
            Create and save plot of power spectra. Default: True
        loglog : bool, optional
            Plot loglog if True, semilogy if False. Default: True
        nyquistlines : bool, optional
            Plot lines at individual and composite Nyquist wavenumbers.
            Default: True
        dealiasinglines : bool, optional
            Plot lines at 2/3 of the individual and composite Nyquist
            wavenumbers. Default: True
        dpi : int, optional
            DPI setting for images.  Each plot is set to be 4 inches, so each
            plot in the image will be 4*DPI pixels in each direction.
            Default: config setting

        """

        # Store inputs
        self.cadence = cadence
        self.fieldlist = fieldlist
        self.norm = norm
        self.loglog = loglog
        self.plot = plot
        self.write = write
        self.nyquistlines = nyquistlines
        self.dealiasinglines = dealiasinglines
        self.dpi = dpi

        # Get defaults from config
        if self.dpi is None:
            self.dpi = decfg.getint('analysis', 'powerspectrum_dpi')

    def setup(self, data, it):

        # Default to all fields in data
        if self.fieldlist is None:
            self.fieldlist = data.fields.keys()

        self.firstrun = [True] * len(self.fieldlist)

        if com_sys.myproc == 0:
            if self.plot:

                self.lines = {}
                self.axes = {}

                # Determine grid size
                ncols = len(self.fieldlist)

                # Setup spacing [top, bottom, left, right] and [height, width]
                t_mar, b_mar, l_mar, r_mar = (0.2, 0.2, 0.2, 0.2)
                t_pad, b_pad, l_pad, r_pad = (0.15, 0.03, 0.03, 0.03)
                h_data, w_data = (1., 1.)

                h_im = t_pad + h_data + b_pad
                w_im = l_pad + w_data + r_pad
                h_total = t_mar + h_im + b_mar
                w_total = l_mar + ncols * w_im + r_mar
                scale = 4.0

                # Create figure and axes
                self.fig = plt.figure(self._n, figsize=(scale * w_total,
                                                        scale * h_total))
                for col, fname in enumerate(self.fieldlist):
                    left = (l_mar + w_im * col + l_pad) / w_total
                    bottom = 1 - (t_mar + h_im - b_pad) / h_total
                    width = w_data / w_total
                    height = h_data / h_total
                    self.axes[col] = self.fig.add_axes([left, bottom,
                                                        width, height])
                    self.axes[col].firstcol = (col == 0)

                # Time title
                height = 1 - (0.6 * t_mar) / h_total
                self.timestring = self.fig.suptitle(r'', y=height, size=16)

                # Directory setup
                if not os.path.exists('frames'):
                    os.mkdir('frames')

            if self.write:

                # Create file for each field
                for fname in self.fieldlist:
                    file = open('%s_power_spectra.dat' %fname, 'w')
                    file.write("# Dedalus Power Spectrum\n")
                    file.close()

    def run(self, data, it):

        for col, fname in enumerate(self.fieldlist):
            field = data[fname]

            # Compute spectrum
            k, spectrum = self._compute_spectrum(field, norm=self.norm)

            if com_sys.myproc == 0:
                if self.plot:

                    # Skip if there are not multiple populated modes
                    if spectrum.nonzero()[0].size <= 1:
                        continue

                    ax = self.axes[col]
                    if self.firstrun[col]:

                        # Plot and store lines
                        if self.loglog:
                            ll = ax.loglog(k, spectrum, 'b.-', mew=0, ms=5)
                        else:
                            ll = ax.semilogy(k, spectrum, 'b.-', mew=0, ms=5)
                        self.lines[col] = ll[0]

                        # Title
                        title = ax.set_title(r'$%s$' %fname, size=14)
                        title.set_y(1.05)

                        # Axis labels
                        ax.set_xlabel(r'$k$', size=12)
                        plt.setp(ax.get_xticklabels(), size=10)

                        if ax.firstcol:
                            ax.set_ylabel(r'$E(k)$', size=12)
                            plt.setp(ax.get_yticklabels(), size=10)
                        else:
                            plt.setp(ax.get_yticklabels(), visible=False)

                        # Nyquist and dealiasing lines
                        for kny in field[0].kny:
                            if self.nyquistlines:
                                ax.axvline(kny, ls='dashed', c='k')
                            if self.dealiasinglines:
                                ax.axvline(2. / 3. * kny, ls='dotted', c='k')

                        kmax = na.sqrt(na.sum(field[0].kny ** 2))
                        if self.nyquistlines:
                            ax.axvline(kmax, ls='dashed', c='r')
                        if self.dealiasinglines:
                            ax.axvline(2. / 3. * kmax, ls='dotted', c='r')

                    else:

                        # Update line position and rescale
                        line = self.lines[col]
                        line.set_ydata(spectrum)
                        ax.relim()
                        ax.autoscale_view()

                if self.write:
                    file = open('%s_power_spectra.dat' %fname, 'a')
                    if self.firstrun[col]:
                        columnnames = 'time\t'
                        columnnames += '\t'.join([repr(ki) for ki in k])
                        file.write(columnnames)
                        file.write('\n')
                    tstring = '%s\t' %data.time
                    specstring = '\t'.join([repr(si) for si in spectrum])
                    file.write(tstring + specstring)
                    file.write('\n')
                    file.close()

            if self.firstrun[col]:
                self.firstrun[col] = False

        if com_sys.myproc == 0:
            if self.plot:

                # Update time title
                tstr = r'$t = %6.3f$' %data.time
                self.timestring.set_text(tstr)

                # Save in frames folder
                outfile = 'frames/power_spectra_n%07i.png' %it
                self.fig.savefig(outfile, dpi=self.dpi)

    def _compute_spectrum(self, field, norm):

        # Compute 1D power samples
        kmag = na.sqrt(field[0].k2())
        power = na.zeros_like(kmag)
        for cindex, comp in field:
            power += na.abs(comp['kspace']) ** 2
        if field.ndim == 2:
            power1d = norm * power * 2. * na.pi * kmag
        else:
            power1d = norm * power * 4. * na.pi * kmag ** 2

        # Construct wavevector magnitude bins
        kmax = na.sqrt(na.sum(field[0].kny ** 2))
        n = na.min(field[0].global_shape['xspace']) / 2.
        n = int(na.min([n, 100]))
        kbottom = na.linspace(0, kmax, n, endpoint=False)
        ktop = kbottom + kbottom[1]

        # Bin the power samples
        spectrum = na.zeros_like(kbottom)
        n = na.zeros_like(kbottom)
        for i in xrange(kbottom.size):
            mask = (kmag >= kbottom[i]) & (kmag < ktop[i]) & (power1d != 0)
            spectrum[i] = na.sum(power1d[mask])
            n[i] = na.asfarray(na.sum(mask))

        # Collect from all processes
        if com_sys.nproc != 1:
            spectrum = com_sys.comm.reduce(spectrum, op=com_sys.MPI.SUM, root=0)
            n = com_sys.comm.reduce(n, op=com_sys.MPI.SUM, root=0)

            if com_sys.myproc != 0:
                return (None, None)

        # Return bin centers and 1D power sample averages
        n[n == 0] = 1.
        return (kbottom + kbottom[1] / 2. , spectrum / n)


class VolumeAverage(AnalysisTask):

    def __init__(self, cadence, va):
        """
        Run volume average tasks.

        Parameters
        ----------
        cadence : int
            Iteration cadence for running task.
        va : VolumeAverageSet object
            Set to run

        """

        # Store inputs
        self.cadence = cadence
        self.volume_average_object = va

    def run(self, data, it):
        self.volume_average_object.run()

