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
from fields import create_field_classes

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
    """
    The object containing all relevant data for the state of the
    system, composed of tensor, vector, and scalar fields.

    """

    timer = Timer()

    def __init__(self, time, shape, length, field_class_dict, field_list=[], params={}):
        """
        The object containing all relevant data for the state of the
        system, composed of tensor, vector, and scalar fields.

        Parameters
        ----------
        time : float
            Simulation time
        shape : tuple of ints
            The shape of the data in xspace: (z, y, x) or (y, x)
        length : tuple of floats
            The length of the data in xspace: (z, y, x) or (y, x)
        field_class_dict : dict
            Dictionary containing customized field classes created by create_field_classes
        field_list : list of tuples (name, type)
            List of tuples containing field names and types
        params : dict
            Additional parameters

        """

        # Store inputs
        self.time = time
        self.shape = shape
        self.ndim = len(self.shape)
        self.length = length
        self._field_classes = field_class_dict
        self.parameters = params
        self.fields = OrderedDict()

        # Add fields
        for name,type in field_list:
            self.add_field(name, type)

    def __getitem__(self, item):
        return self.fields[item]

    def __reduce__(self):
        savedict = {}
        exclude = ['fields', '_field_classes']
        field_keys = zip(self.fields.keys(), [f.__class__.__name__ for f in self.fields.values()])

        # Grab representation from first fieldclass
        k = self._field_classes.keys()[0]
        field_class_rep = self._field_classes[k].representation

        for k,v in self.__dict__.iteritems():
            if k not in exclude:
                savedict[k] = v
        return (_reconstruct_data, (self.__class__, savedict, field_class_rep, field_keys))

    def __iter__(self):
        """Iterate over field objects."""

        for name, field in self.fields.iteritems():
            yield (name, field)

    def clone(self):
        return self.__class__(self.time, self.shape, self.length, self._field_classes,
                              params=self.parameters)

    def set_time(self, time):
        self.time = time
        for name, field in self.fields.iteritems():
            if field.representation.__name__ == 'FourierShearRepresentation':
                for i,c in field:
                    c._update_k()

    def add_field(self, name, fieldtype):
        """
        Add a new field.

        Parameters
        ----------
        name : str
            Name of the field to add
        fieldtype : str
            Type of field to add, e.g. "ScalarField"

        Notes
        -----
        There is a SIGNIFICANT performace penalty
        for doing this (creating the FFTW plans), so make sure it does
        not happen inside any loops you care about performance on.

        """

        if name in self.fields.keys():
            raise ValueError("Field with this name already exists.")
        else:
            self.fields[name] = self._field_classes[fieldtype](self)

    def snapshot(self, root_grp):
        """
        Save all fields to the HDF5 group given by input.

        Parameters
        ----------
        root_grp : h5py group object
            Group object where all fields go

        """

        for name, field in self.fields.iteritems():
            fgrp = root_grp.create_group(name)
            field.save(fgrp)

    def create_tmp_data(self, space):
        """
        Create a temporary data space with the dimensions of the
        first field element (we assume all fields have the same shape)

        Parameters
        ----------
        space : str
            "xspace" or "kspace" data container

        Returns
        -------
        out : ndarray
            Numpy array of zeros with shape of data space (x or k) and
            dtype set to the same as the data

        """

        fi = self.fields[self.fields.keys()[0]][0]
        return na.zeros(fi.local_shape[space], fi.dtype[space])

    def report_counts(self):
        for name, field in self.fields.iteritems():
            mylog.debug("field %s" %name)
            field.report_counts()
