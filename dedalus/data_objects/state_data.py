"""
The main Dedalus data object. Has a dictionary of fields, each of
which can be a Tensor, Vector, or Scalar object.

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

from collections import OrderedDict
import h5py
import numpy as na
from dedalus.funcs import get_mercurial_changeset_id
from dedalus.utils.api import Timer
from dedalus.utils.logger import mylog
from field_object import create_field_classes

def _reconstruct_data(*args, **kwargs):
    arg1 = args[1]
    field_classes = create_field_classes(args[2], arg1['shape'], arg1['length'])
    new_args = [arg1['time'], arg1['shape'], arg1['length'], field_classes]
    obj = args[0](*new_args)
    obj.__dict__.update(arg1)
    for f, t in args[3]:
        obj.add_field(f, t)
    return obj

class StateData(object):
    """the object containing all relevant data for the state of the
    system. composed of vector and scalar fields, each of which in
    turn is determined at start up with 1,2, or 3 componentes, named
    according to the coordinate system in use (xyz/rthetaphi/rphiz/etc
    etc etc)...or 0,1,2
    """
    timer = Timer()
    def __init__(self, time, shape, length, field_class_dict, field_list=[], params={}):
        self.time = time
        self.shape = shape
        self.ndim = len(self.shape)
        self.length = length
        self._field_classes = field_class_dict
        self.parameters = params
        self.fields = OrderedDict()
        
        for f,t in field_list:
            self.add_field(f, t)
                                
    def clone(self):
        return self.__class__(self.time, self.shape, self.length, self._field_classes, 
                              params=self.parameters)
                              
    def set_time(self, time):
        self.time = time
        for k,f in self.fields.iteritems():
            for i in xrange(f.ncomp):
                if f[i].__class__.__name__ == 'FourierShearRepresentation':
                    f[i]._update_k()
        
    def __getitem__(self, item):
        return self.fields[item]

    def __reduce__(self):
        savedict = {}
        exclude = ['fields', '_field_classes']
        field_keys = zip(self.fields.keys(), [f.__class__.__name__ for f in self.fields.values()])
        # grap representation from first fieldclass
        k = self._field_classes.keys()[0]
        field_class_rep = self._field_classes[k].representation

        for k,v in self.__dict__.iteritems():
            if k not in exclude:
                savedict[k] = v
        return (_reconstruct_data, (self.__class__, savedict, field_class_rep, field_keys))

    def add_field(self, field, field_type):
        """add a new field. There is a SIGNIFICANT performace penalty
        for doing this (creating the FFTW plan), so make sure it does
        not happen inside any loops you care about performance on....

        """
        if field not in self.fields.keys():
            self.fields[field] = self._field_classes[field_type](self)

    def snapshot(self, root_grp):
        """save all fields to the HDF5 group given by input

        input
        -----
        root_grp -- h5py group object where all fields go

        """
        for name, field in self.fields.iteritems():
            fgrp = root_grp.create_group(name)
            field.save(fgrp)

    def create_tmp_data(self, space):
        """create a temporary data space with the dimensions of the
        first field element (we assume all fields have the same shape)
        
        input
        -----
        space -- x or kspace. In parallel, these have different shapes

        output
        ------
        a numpy array of zeros with shape of data space (x or k) and
        dtype set to the same as the data
        
        """

        fi = self.fields[self.fields.keys()[0]][0]
        return na.zeros(fi.local_shape[space], fi.data.dtype)

    def report_counts(self):
        for k,v in self.fields.iteritems():
            mylog.debug("field %s" % k)
            v.report_counts()
