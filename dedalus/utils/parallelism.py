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
    fi = h5py.File(data_file % 0)
    time = fi['/time'].value
    space = fi['/fields/u/0'].attrs['space']
    local_size = fi['/fields/u/0'].shape
    dtype = fi['/fields/u/0'].dtype
    data = np.empty((nproc,)+local_size, dtype=dtype)
    fi[field].read_direct(data[0])
    fi.close()
    for i in range(1,nproc):
        fi = h5py.File(data_file % i)
        fi[field].read_direct(data[i])
        fi.close()

        
    if space == 'xspace':
        concat_ax = 2
    else:
        concat_ax = 0

    print "loaded %s at time = %f" % (field, time)
    return np.concatenate(data,axis=concat_ax), space
    
