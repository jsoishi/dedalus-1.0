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
    new_tensorclass = type(tname, (TensorField,), {'representation': representation,
                                                  'shape': shape})
    vname = "%sVectorField" % name
    new_vectorclass = type(vname, (VectorField,), {'representation': representation,
                                                  'shape': shape})
    sname = "%sScalarField" % name
    new_scalarclass = type(sname, (ScalarField,), {'representation': representation,
                                                  'shape': shape})

    return new_tensorclass, new_vectorclass, new_scalarclass

def lookup(name, translation_table):
    """this may need to be inlined?

    """
    name = translation_table.get(name, None)
    if name is None:
        raise KeyError
    return name
    
class BaseField(object):
    def __init__(self, ndim=-1):
        self.components = []
        if ndim == -1:
            self.ndim = len(self.shape)
        else:
            self.ndim = ndim

        for f in range(self.ndim):
            self.components.append(self.representation(self.shape))

        self.trans = self.components[-1].trans # representation must
                                               # provide a translation
                                               # table for coordinate
                                               # names.


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

        f._curr_space = space

    def __call__(self, comp_name):
        if type(comp_name) == str:
            comp_name = lookup(comp_name, self.trans)

        return self.components[comp_name]

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
        BaseField.__init__(self, 1)

    def __getitem__(self, space):
        return self.components[0][space]
    
    def __setitem__(self, space, data):
        BaseField.__setitem__(self, (0,space), data)

    def __call__(self):
        return self.components[0]
        
    def zero(self):
        self.zero_all()

