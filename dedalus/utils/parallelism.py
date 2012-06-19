"""parallel support.

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

def setup_parallel_objs(global_shape, global_len):
    """Helper function for parallel runs. Given a global shape and
    length, it returns a local shape and length.

    inputs
    ------
    global_shape (tuple of int)
    global_length (tuple of reals)

    returns
    -------
    local_shape, local_len (tuple of ints, tuple of reals)

    """
    
    local_shape = (global_shape[0]/com_sys.nproc,) + global_shape[1:]
    
    local_len = (global_len[0]/com_sys.nproc,) + global_len[1:]

    return local_shape, local_len

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

def reduce_sum(data):
    local_sum = data.sum()
    if com_sys.comm == None:
        return local_sum

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
    """simple function to swap index [0] and [1]. useful for
    constructing quantities for the FFTW parallel data objects.
    
    """
    a = arr[1]
    arr[1] = arr[0]
    arr[0] = a

    return arr
