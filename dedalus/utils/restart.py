"""restart support.

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
import os
import cPickle
import h5py
import time
from dedalus.funcs import insert_ipython
from dedalus.utils.parallelism import com_sys
OBJECT_FILENAME='dedalus_obj_%04i.cpkl'

def restart(snap_dir):
    """restarts a run from the snapshot in snap_dir.

    returns a tuple of (Physics (RHS), data, time integrator)

    """
    myproc = com_sys.myproc
    obj_file = open(os.path.join(snap_dir,OBJECT_FILENAME % myproc), 'r')
    RHS = cPickle.load(obj_file)
    data = cPickle.load(obj_file)
    ti = cPickle.load(obj_file)

    RHS._setup_aux_fields(data.time, RHS._aux_fields)
    # now load data from hdf5 file...
    DATA_FILENAME = 'data.cpu%04i' % myproc
    filename = os.path.join(snap_dir, DATA_FILENAME)
    load_all_data(data, filename)
    ti.RHS = RHS
    ti._start_time = time.time()
    return RHS, data, ti

def load_all_data(data, filename):
    """will attempt to load every field in data object

    input/output
    ------------
    data

    input
    -----
    filename -- the hdf5 file to load from

    """
    data_file = h5py.File(filename, mode='r')

    for field in data.fields:
        field_name = '/fields/' + field
        for comp in range(data.fields[field].ncomp):
            field_comp = '%s/%i' % (field_name, comp)
            load_space = data_file[field_comp].attrs['space']
            if load_space != data[field][comp]._curr_space:
                data[field][comp][load_space]
            try:
                output = (field,comp) + tuple(data[field][comp].data.shape)
            except:
                raise KeyError("Data File field %s component %i is missing or corrupt" % (field, comp))
            data[field][comp].data[:] = data_file[field_comp][:]
            data.fields[field][comp]._curr_space = data_file[field_comp].attrs['space']

    data_file.close()
