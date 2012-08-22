"""
Parallel support.

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi.  All Rights Reserved.

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

import numpy as na
import numpy.lib.stride_tricks as st

class CommunicationSystem(object):
    comm = None
    MPI = None
    def __init__(self):
        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.MPI = MPI
        except ImportError:
            print "Cannot import mpi4py. Parallelism disabled"

        if self.comm:
            self.myproc = self.comm.Get_rank()
            self.nproc = self.comm.Get_size()
            from dedalus.utils.fftw import fftw
            try:
                fftw.fftw_mpi_init()
            except:
                raise NotImplementedError("Cannot initialize fftw's MPI facilities. Rebuild FFTW with parallel support.")
        else:
            self.myproc = 0
            self.nproc = 1

com_sys = CommunicationSystem()

def load_all(field, snap_dir):
    """hacky function to concatenate all MPI tasks data into a single
    global size cube for analysis/testing.

    inputs
    ------
    field -- the name of the field to be retrived
    snap_dir -- the snapshot directory

    returns
    -------
    data -- a numpy array containing the data.
    space -- a string of 'xspace' or 'kspace'
    """
    import h5py
    import os
    import glob
    import numpy as np

    nproc = len(glob.glob(os.path.join(snap_dir,"data.cpu*")))

    data_file = os.path.join(snap_dir, 'data.cpu%04i')

    # prefetch step to get sizes
    data = []
    for i in range(nproc):
        fi = h5py.File(data_file % i)
        time = fi['/time'].value
        space = fi['/fields/u/0'].attrs['space']
        dtype = fi['/fields/u/0'].dtype
        data.append(np.empty(fi['/fields/u/0'].shape, dtype=dtype))
        fi.close()

    if not field.startswith('/fields/'):
        field = os.path.join('/fields',field)

    for i in range(1,nproc):
        fi = h5py.File(data_file % i)
        fi[field].read_direct(data[i])
        fi.close()

    concat_ax = 0

    print "loaded %s at time = %f" % (field, time)
    return np.concatenate(data,axis=concat_ax), space

######
# these reduction operators are light wrappers around mpi functions
def reduce_mean(data):
    if com_sys.comm == None:
        return data.mean()
    else:
        local_sum = data.sum()
        total = com_sys.comm.reduce(local_sum,op=com_sys.MPI.SUM)
        # this should be cached, but should work.
        global_size = com_sys.comm.reduce(data.size,op=com_sys.MPI.SUM)

    if com_sys.myproc == 0:
        return total/global_size

def reduce_sum(data, reduce_all=False):
    local_sum = data.sum()
    if com_sys.comm == None:
        return local_sum

    if reduce_all:
        total = com_sys.comm.allreduce(local_sum,op=com_sys.MPI.SUM)
        return total
    else:
        total = com_sys.comm.reduce(local_sum,op=com_sys.MPI.SUM)
        if com_sys.myproc == 0:
            return total

def reduce_min(data):
    local_min = data.min()
    if com_sys.comm == None:
        return local_min

    global_min = com_sys.comm.reduce(local_min,op=com_sys.MPI.MIN)
    if com_sys.myproc == 0:
        return global_min

def reduce_max(data):
    local_max = data.max()
    if com_sys.comm == None:
        return local_max

    global_max = com_sys.comm.reduce(local_max,op=com_sys.MPI.MAX)
    if com_sys.myproc == 0:
        return global_max

def swap_indices(arr):
    """
    Simple function to swap index [0] and [1].  Useful for
    constructing quantities for the FFTW parallel data objects.

    """

    if type(arr) == na.ndarray:
        out_arr = arr.copy()
    elif type(arr) == list:
        out_arr = list(arr)
    else:
        raise NotImplementedError("swap_indices only implemented for numpy arrays and lists.")

    a = out_arr[1]
    out_arr[1] = out_arr[0]
    out_arr[0] = a

    return out_arr

def pickle(data,name):
    """quickly dump data to a file with name, 1 proc for each file.

    """
    import cPickle
    filen = "%s_proc%05i.dat" % (name, com_sys.myproc)
    outf = open(filen,'w')
    cPickle.dump(data,outf)
    outf.close()

def strided_copy(input):
    """Helper function to 'deep copy' a view using stridetricks."""

    return st.as_strided(input, shape=input.shape, strides=input.strides)

def get_plane(comp, space='xspace', axis='z', index='middle',
              return_position_arrays=True):
    """
    Return 2D slice of data, assembling across processes if needed.

    Parameters
    ----------
    comp : Representation object
        Field component to assemble from.
    space : str, optional
        'xspace' (default) or 'kspace'
    axis : str, optional
        Axis normal to desired slice, defaults to 'z'. Ignored for 2D data.
        i.e. 'x' for a y-z plane
             'y' for a x-z plane
             'z' for a x-y plane
    index : int or string, optional
        Index for slicing as an integer, or 'top', 'middle' (default), or 'bottom'.
        Ignored for 2D data.
    return_position_arrays : boolean, optional
        Return arrays of the x or k positions of the output plane data. Defaults True.

    Returns
    -------
    plane_data : ndarray
        Component values in the output plane
    index : int
        Slice index, transformed if input index was string
    name0 : str, optional
        Name of direction along 0th-dimension of the output plane, e.g. 'x' or 'ky'
    grid0 : ndarray, optional
        Array of positions along 0th-dimension of the output plane
        i.e. 'x' for axis = 'y' or 'z'
             'y' for axis = 'x'
    name1 : str, optional
        Name of direction along 1st-dimension of the output plane
    grid1 : ndarray, optional
        Array of positions along 1st-dimension of the output plane
        i.e. 'z' for axis = 'x' or 'y'
             'y' for axis = 'z'

    Examples
    --------
    >>> get_plane(data['u']['x'], 'kspace', 'x', 0)
    (global_ux[:, :, 0], 'ky', global_ky, 'kz', global_kz)

    """

    # Check space
    comp.require_space(space)

    # Retrieve translation table for requested space
    if space == 'kspace':
        trans = comp.ktrans
    elif space == 'xspace':
        trans = comp.xtrans

    # Determine slice index for string inputs
    if comp.ndim == 2:
        index = 0
    elif index == 'top':
        index = comp.global_shape[space][trans[axis]] - 1
    elif index == 'middle':
        index = comp.global_shape[space][trans[axis]] / 2
    elif index == 'bottom':
        index = 0
    else:
        index = int(index)
        if index < 0: index = index % comp.global_shape[space][trans[axis]]

    # Prepare data from each process
    gather = False
    reduce = False

    if comp.ndim == 2:
        if com_sys.nproc == 1:
            plane_data = strided_copy(comp.data)
        else:
            proc_data = strided_copy(comp.data)
            gather = True

    elif comp.ndim == 3:
        slicelist = [slice(None)] * 3
        if com_sys.nproc == 1:
            slicelist[trans[axis]] = index
            plane_data = strided_copy(comp.data[slicelist])
        else:
            if trans[axis] == 0:
                if ((index >= comp.offset[space]) and
                    (index < comp.offset[space] + comp.local_shape[space][0])):
                    slicelist[trans[axis]] = index - comp.offset[space]
                    proc_data = strided_copy(comp.data[slicelist])
                else:
                    proc_data = 0.
                reduce = True
            else:
                slicelist[trans[axis]] = index
                proc_data = strided_copy(comp.data[slicelist])
                gather = True

    # Gather or reduce if necessary
    if gather:
        gathered_data = com_sys.comm.gather(proc_data, root=0)
        if com_sys.myproc == 0:
            plane_data = na.concatenate(gathered_data)
        else:
            plane_data = None
    elif reduce:
        plane_data = com_sys.comm.reduce(proc_data, op=com_sys.MPI.SUM, root=0)

    # Transpose if necessary
    if (comp.ndim == 2) and (space == 'kspace'):
        plane_data = na.transpose(plane_data)
    if (comp.ndim == 3) and (space == 'kspace') and (axis == 'x'):
        plane_data = na.transpose(plane_data)

    # Position arrays
    if not return_position_arrays:
        return (plane_data, index)
    else:
        if axis == 'x':
            name0, name1 = ('y', 'z')
        elif axis == 'y':
            name0, name1 = ('x', 'z')
        elif axis == 'z':
            name0, name1 = ('x', 'y')

        if space == 'xspace':
            if com_sys.myproc != 0:
                return (None,) * 6

            grid1, grid0 = na.mgrid[slice(float(comp.global_shape[space][trans[name1]])),
                              slice(float(comp.global_shape[space][trans[name0]]))]
            grid0 *= comp.length[trans[name0]] / comp.global_shape[space][trans[name0]]
            grid1 *= comp.length[trans[name1]] / comp.global_shape[space][trans[name1]]

        elif space == 'kspace':
            k = {}
            if comp.ndim == 2:
                if com_sys.nproc == 1:
                    k['x'] = na.transpose(comp.k['x'])
                    k['y'] = comp.k['y']
                else:
                    gathered_kx = com_sys.comm.gather(comp.k['x'], root=0)
                    if com_sys.myproc == 0:
                        k['x'] = na.transpose(na.concatenate(gathered_kx))
                    if hasattr(comp, '_ky'):
                        gathered_ky = com_sys.comm.gather(comp.k['y'], root=0)
                        if com_sys.myproc == 0:
                            k['y'] = na.concatenate(gathered_ky)
                    else:
                        k['y'] = comp.k['y']
            elif comp.ndim == 3:
                k['x'] = comp.k['x'][0, :, :]
                k['z'] = comp.k['z'][:, :, 0]
                if com_sys.nproc == 1:
                    k['y'] = na.transpose(comp.k['y'][:, 0, :])
                else:
                    gathered_ky = com_sys.comm.gather(comp.k['y'], root=0)
                    if com_sys.myproc == 0:
                        k['y'] = na.transpose(na.concatenate(gathered_ky)[:, 0, :])
            if com_sys.myproc != 0:
                return (None,) * 6

            grid0 = k[name0] * na.ones(plane_data.shape)
            grid1 = na.transpose(k[name1]) * na.ones(plane_data.shape)
            name0 = 'k' + name0
            name1 = 'k' + name1

        return (plane_data, index, name0, grid0, name1, grid1)

