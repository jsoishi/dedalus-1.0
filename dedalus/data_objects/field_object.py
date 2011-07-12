"""The Main Dedalus data object. This is dynamically created with a
given representation when the physics class is initialized.

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

import h5py

def create_field_obj(representation, shape, name):
    """utility function to bind representation and shape to tensor,
    vector, and scalar fields.

    """
    tname = "%sTensorField" % name
    new_tensorclass = type(name, (TensorField,), {'representation': representation,
                                                  'shape': shape})
    vname = "%sVectorField" % name
    new_vectorclass = type(name, (VectorField,), {'representation': representation,
                                                  'shape': shape})
    sname = "%sScalarField" % name
    new_scalarclass = type(name, (ScalarField,), {'representation': representation,
                                                  'shape': shape})

    return new_tensorclass, new_vectorclass, new_scalarclass

def lookup(name, translation_table):
    """this may need to be inlined?

    """
    name = translation_table.get(comp_name, None)
    if name is None:
        raise KeyError
    return name
    
class BaseField(object):
    def __init__(self, ndim=-1):
        self.trans = self.representation.trans # representation must
                                               # provide a translation
                                               # table for coordinate
                                               # names.
        self.components = []
        if ndim == -1:
            self.ndim = len(self.shape)
        for f in self.ndim:
            self.components.append(self.representation(self.shape))

    def __getitem__(self, item):
        """item is a 2-tuple containing a component name and a space
        (x or k).

        """
        comp_name, space = item
        if type(comp_name) == str:
            comp_name = lookup(comp_name, self.trans)
        return self.components[comp_name][space]

    def __setitem__(self, item, data):
        """this needs to ensure the pointer for the field's data
        member doesn't change for FFTW. Currently, we do that by
        slicing the entire data array. 
        """
        comp_name, space = item
        if type(comp_name) == str:
            comp_name = lookup(comp_name, self.trans)

        f = self.components[comp_name]
        if data.size < f.data.size:
            sli = [slice(i/4+1,i/4+i+1) for i in data.shape]
            f.data[sli] = data
        else:
            sli = [slice(i) for i in f.data.shape]
            f.data[:] = data[sli]

        f.__cur_space = space

    def zero(self, item):
        if type(item) == str:
            item = lookup(item, self.trans)

        self.components[item].data[:] = 0.

    def zero_all(self):
        for f in self.components:
            f.data[:] = 0.

class TensorField(BaseField):
    """used mostly for the velocity gradient tensor

    """
    def __init__(self):
        ndim = len(self.shape)**2
        BaseField.__init__(self, ndim=ndim)

class VectorField(BaseField):
    """these should have N components with names defined at simulation initialization time.

    state data will be composed of vectors and scalars
    """
    pass

class ScalarField(BaseField):
    """always has one component

    """
    def __init__(self):
        def BaseField.__init__(self, 1)

    def __getitem__(self, space):
        return self.components[0][space]
    
    def __setitem__(self, space, data):
        BaseField.__setitem__(self, (0,space), data)
        
    def zero(self):
        self.zero_all()

class StateData(object):
    """the object containing all relevant data for the state of the
    system. composed of vector and scalar fields, each of which in
    turn is determined at start up with 1,2, or 3 componentes, named
    according to the coordinate system in use (xyz/rthetaphi/rphiz/etc
    etc etc)...or 0,1,2
    """
    def __init__(self, time, TensorClass, VectorClass, ScalarClass):
        self.time = time
        self.fields = {}
        self.__field_classes = {'tensor': TensorClass,
                                'vector': VectorClass, 
                                'scalar': ScalarClass}

    def add_field(self, field, field_type):
        """add a new field. There is a SIGNIFICANT performace penalty
        for doing this (creating the FFTW plan), so make sure it does
        not happen inside any loops you care about performance on....

        """
        if field not in self.fields.keys():
            self.fields[field] = self.__field_classes[field_type]()

    def snapshot(self, nsnap):
        """NEEDS TO BE UPDATED FOR NEW FIELD TYPES

        """
        filename = "snap_%05i.cpu%04i" % (nsnap, 0)
        outfile = h5py.File(filename, mode='w')
        grp = outfile.create_group('/data')
        dset = outfile.create_dataset('time',data=self.time)
        for f in self.fields.keys():
            dset = grp.create_dataset(f,data=self[f].data)
            dset.attrs["space"] = self[f]._curr_space

        outfile.close()

