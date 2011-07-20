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
import weakref
import h5py

def create_field_classes(representation, shape, length, name):
    """utility function to bind representation and shape to tensor,
    vector, and scalar fields.

    """
    tname = "%sTensorField" % name
    new_tensorclass = type(tname, (TensorField,), {'representation': representation,
                                                  'shape': shape, 'length': length})
    vname = "%sVectorField" % name
    new_vectorclass = type(vname, (VectorField,), {'representation': representation,
                                                  'shape': shape, 'length': length})
    sname = "%sScalarField" % name
    new_scalarclass = type(sname, (ScalarField,), {'representation': representation,
                                                  'shape': shape, 'length': length})

    return {'tensor': new_tensorclass,
            'vector': new_vectorclass,
            'scalar': new_scalarclass}

def lookup(name, translation_table):
    """this may need to be inlined?"""
    
    name = translation_table.get(name, None)
    if name is None:
        raise KeyError
    return name
    
class BaseField(object):
    def __init__(self, sd, ncomp=-1):
        """
        inputs
        ------
        sd -- state data object that creates it. stored as a weak ref

        ncomp (optional) -- the number of components

        """
        
        # self.representation provided in call to create_field_classes
        # self.shape provided in call to create_field_classes
        # self.length provided in call to create_field_classes
        self.sd = weakref.proxy(sd)
        self.ndim = len(self.shape)

        # Construct components
        self.components = []
        if ncomp == -1:
            self.ncomp = self.ndim
        else:
            self.ncomp = ncomp

        for f in xrange(self.ncomp):
            self.components.append(self.representation(self.sd, self.shape, self.length))

        # Take translation table for coordinate names from representation
        self.trans = self.components[-1].trans
                                               
        self.integrating_factor = 0.

    def __getitem__(self, comp_name):
        """If item is not a component number, lookup in translation table."""
        
        if type(comp_name) == str:
            comp_name = lookup(comp_name, self.trans)
        return self.components[comp_name]

    def zero(self, comp_name):
        if type(comp_name) == str:
            comp_name = lookup(comp_name, self.trans)
        self.components[comp_name].data[:] = 0.

    def zero_all(self):
        for f in self.components:
            f.data[:] = 0.

class TensorField(BaseField):
    """Tensor class. Currently used mostly for the velocity gradient tensor."""
    
    def __init__(self, sd, **kwargs):
        ncomp = len(self.shape) ** 2
        BaseField.__init__(self, sd, ncomp=ncomp, **kwargs)

class VectorField(BaseField):
    """these should have N components with names defined at simulation initialization time.

    state data will be composed of vectors and scalars
    """
    pass

class ScalarField(BaseField):
    """Scalar class; always has one component."""
    
    def __init__(self, sd, **kwargs):
        BaseField.__init__(self, sd, ncomp=1, **kwargs)

    def __getitem__(self, item):
        """
        0 call returns the scalar representation object. 
        Other calls passed to the representation object.
        """
        
        if item == 0: return self.components[0]
        return self.components[0][item]

    def __setitem__(self, item, data):
        """Set calls passed to the scalar representation object."""
        
        self.components[0].__setitem__(item, data)

    def __getattr__(self, attr):
        """
        In order to make scalar fields work as though they have no
        internal structure, we provide this method to search the
        attributes of the underlying representation (stored in
        self.components[0]). Thus, a ScalarField will act like a
        single instance of its underlying representation.
        """
        
        return self.components[0].__getattribute__(attr)

    def zero(self):
        self.zero_all()

