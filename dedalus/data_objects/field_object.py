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

def create_field_classes(representation, shape, length):
    """utility function to bind representation and shape to tensor,
    vector, and scalar fields.

    """
    tname = "TensorField"
    new_tensorclass = type(tname, (TensorFieldBase,), {'representation': representation,
                                                  'shape': shape, 'length': length})

    vname = "VectorField"
    new_vectorclass = type(vname, (VectorFieldBase,), {'representation': representation,
                                                  'shape': shape, 'length': length})
    sname = "ScalarField" 
    new_scalarclass = type(sname, (ScalarFieldBase,), {'representation': representation,
                                                  'shape': shape, 'length': length})

    return {tname: new_tensorclass,
            vname: new_vectorclass,
            sname: new_scalarclass}

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
                                               
        self.integrating_factor = None

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

    def save(self, group):
        group.attrs["representation"] = self.representation.__name__
        group.attrs["type"] = self.__class__.__name__
        for f in range(self.ncomp):
            dset = group.create_dataset(str(f), 
                                        self.shape, 
                                        dtype=self.components[f].data.dtype)
            
            self.components[f].save(dset)


class TensorFieldBase(BaseField):
    """Tensor class. Currently used mostly for the velocity gradient tensor."""
    
    def __init__(self, sd, **kwargs):
        ncomp = len(self.shape) ** 2
        BaseField.__init__(self, sd, ncomp=ncomp, **kwargs)

class VectorFieldBase(BaseField):
    """these should have N components with names defined at simulation initialization time.

    state data will be composed of vectors and scalars
    """
    
    def div_free(self):
        """Project off compressible parts of the field."""
    
        kV = 0
        for i in xrange(self.ncomp):
            kV += self[i].k[self.trans[i]] * self[i]['kspace']

        k2 = self[i].k2(no_zero=True)
        
        for i in xrange(self.ncomp):
            self[i]['kspace'] -= self[i].k[self.trans[i]] * kV / k2
    
class ScalarFieldBase(BaseField):
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

