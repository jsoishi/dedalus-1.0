"""
The main Dedalus data object. This is dynamically created with a
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
import numpy as na
from dedalus.utils.logger import mylog

def create_field_classes(representation, shape, length):
    """
    Utility function to bind representation and shape to tensor,
    vector, and scalar fields.

    """
    
    attr_dict = {'representation': representation,
                 'shape': shape, 
                 'length': length}
    
    tname = "TensorField"
    new_tensorclass = type(tname, (TensorFieldBase,), attr_dict)

    vname = "VectorField"
    new_vectorclass = type(vname, (VectorFieldBase,), attr_dict)
    
    sname = "ScalarField" 
    new_scalarclass = type(sname, (ScalarFieldBase,), attr_dict)

    return {tname: new_tensorclass,
            vname: new_vectorclass,
            sname: new_scalarclass}
    
class BaseField(object):
    """Base class for all field objects."""

    def __init__(self, sd, ncomp):
        """
        Basic field object containing field components.
        
        Parameters
        ----------
        sd : StateData object
        ncomp : int
            Number of components in the field.

        """
        
        # self.representation provided in call to create_field_classes
        # self.shape provided in call to create_field_classes
        # self.length provided in call to create_field_classes
        
        # Store inputs
        self.sd = weakref.proxy(sd)
        self.ncomp = ncomp
        self.ndim = len(self.shape)

        # Construct components
        self.components = []
        for i in xrange(self.ncomp):
            self.components.append(self.representation(self.sd, self.shape, self.length))

        # Option for integrating factor                                       
        self.integrating_factor = None
        
        # Translation table to be specified in derived classes
        self.trans = {}
        
    def __getitem__(self, item):
        if type(item) == str:
            item = self.trans[item]
        return self.components[item]
        
    def __setitem__(self, item, data):
        if type(item) == str:
            item = self.trans[item]
        self.components[item] = data

    def __iter__(self):
        for i in xrange(self.ncomp):
            yield (i, self.components[i])

    def zero(self, comp, space='kspace'):
        """Zero specified component, setting space if needed."""
        
        if type(comp) == str:
            comp = self.trans[comp]
        self.components[comp][space] = 0.

    def zero_all(self, space='kspace'):
        """Zero all field components, setting space if needed."""
        
        for c in self.components:
            c[space] = 0.

    def save(self, group):
        group.attrs["representation"] = self.representation.__name__
        group.attrs["type"] = self.__class__.__name__
        for i,c in self:
            dset = group.create_dataset(str(i), 
                                        c.local_shape[c._curr_space], 
                                        dtype=c.data.dtype)
            c.save(dset)

    def report_counts(self):
        """Include transform counts from all components in log."""
    
        for i,c in self:
            mylog.debug("component[%i] forward count = %i" % (i,c.fwd_count))
            mylog.debug("component[%i] rev count = %i" % (i,c.rev_count))

class TensorFieldBase(BaseField):
    """Tensor class. Primarily used for vector covariant derivatives."""
    
    def __init__(self, sd):
        ncomp = sd.ndim ** 2
        BaseField.__init__(self, sd, ncomp)

class VectorFieldBase(BaseField):
    """Vector class. Number of components equal to simulation dimension."""
    
    def __init__(self, sd):
        ncomp = sd.ndim
        BaseField.__init__(self, sd, ncomp)
        #self.trans = self.components[0].trans
        self.trans = {'x': 0, 'y': 1, 'z': 2,
                      0:'x', 1:'y', 2:'z'} # for vector fields

    def div_free(self, verbose=False):
        """
        Project off irrotational part of the vector field.
        
        F = -grad(phi) + curl(A)
        
        ==> laplace(phi) = - div(F) 
        
        """
        
        mylog.debug("Projecting off irrotational part of vector field.")
        
        if verbose:
            # Compute pre-projection power
            power0 = 0.
            for i in xrange(self.ncomp):
                power0 += 0.5 * na.sum(na.abs(self[i]['kspace']) ** 2)
            mylog.debug("Pre-projection power : %f" % power0)
    
        KF = 0
        for i in xrange(self.ncomp):
            KF += self[i].k[self.trans[i]] * self[i]['kspace']

        k2 = self[i].k2(no_zero=True)
        
        for i in xrange(self.ncomp):
            self[i]['kspace'] -= self[i].k[self.trans[i]] * KF / k2
        
        if verbose:
            # Compute post-projection power
            power1 = 0.
            for i in xrange(self.ncomp):
                power1 += 0.5 * na.sum(na.abs(self[i]['kspace']) ** 2)
            mylog.debug("Post-projection power: %f" % power1)
            mylog.debug("Power projected off  : %f" % (power1 - power0))
    
class ScalarFieldBase(BaseField):
    """Scalar class. One component."""
    
    def __init__(self, sd):
        ncomp = 1
        BaseField.__init__(self, sd, ncomp)

    def __getitem__(self, item):
        """
        0 call returns the scalar representation object. 
        Other calls passed to the representation object.
        
        """
        
        if item == 0: 
            return self.components[0]
        else:
            return self.components[0][item]

    def __setitem__(self, item, data):
        """
        0 call sets the scalar representation object.
        Other calls passed to the representation object.
        
        """
        
        if item == 0:
            self.components[0] = data
        else:
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

    def zero(self, space='kspace'):
        """Zero specified component, setting space if needed."""
        
        self.components[0][space] = 0.
