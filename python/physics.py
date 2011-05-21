"""Physics class. This defines fields, and provides a right hand side
to time integrators.

Author: J. S. Oishi <jsoishi@gmail.com>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi.  All Rights Reserved.

  This file is part of pydro.

  pydro is free software; you can redistribute it and/or modify
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

import numpy as na
from data_object import create_data

class Physics(object):
    """This is a base class for a physics object. It needs to provide
    a right hand side for a time integration scheme.
    """
    def __init__(self, shape, representation):
        self._shape = shape
        self._ndims = len(self._shape)
        self._representation = representation
        dataname = "%sData" % self.__class__.__name__
        self.__DataClass = create_data(self._representation, self._shape, dataname)
        self.parameters = {}
        self.fields = {}
        self.aux_fields = {}
    
    def __getitem__(self, item):
         a = self.parameters.get(item, None)
         # if a is None:
         #      a = self.parameters.get(item, None)
         if a is None:
              raise KeyError
         return

    def create_fields(self, t, fields=None):
         if fields == None:
              return self.__DataClass(self.fields, t)
         else:
              return self.__DataClass(fields, t)

    def _setup_parameters(self, params):
        for k,v in params.iteritems():
            self.parameters[k] = v

    def _setup_aux_fields(self, aux):
         for f in aux:
              self.aux_fields[f] = representation(self,shape)

    def RHS(self):
        pass

class Hydro(Physics):
    """incompressible hydrodynamics.

    """
    def __init__(self,*args):
        Physics.__init__(self, *args)
        self.fields = ['ux','uy']
        self._naux = 4
        if self._ndims == 3:
             self.fields.append('uz')
             self._naux = 9

        params = {'nu': 0.}
        
        self.aux_fields = []
        for i in range(self._naux):
             self.aux_fields.append(self._representation(self._shape))
        self._setup_parameters(params)
    
    def RHS(self, data):
        vgradv = self.vgradv(data)
        pressure = self.pressure(data)

        return
    
    def pressure(self, data):
        pass

    def vgradv(self, data):
         gradv = self.create_fields(data.time,fields=range(self._ndims**2))
         i = 0
         trans = {0: 'x', 1: 'y', 2: 'z'}
         slices = self._ndims*(slice(None),)
         for f in self.fields:
              for dim in range(self._ndims):
                  print "%i is d%s/d%s" % (i, f, trans[dim])
                  gradv[i].data[slices] = data[f].deriv(trans[dim])
                  i += 1
         
         return gradv

if __name__ == "__main__":
    import pylab as P
    from fourier_data import FourierData
    from init_cond import taylor_green
    a = Hydro((100,100),FourierData)
    data = a.create_fields(0.)
    data['ux'], data['uy'] = taylor_green(data['ux'],data['uy'])
    test = a.vgradv(data)
    print test[1]._curr_space
    for i in range(4):
        P.subplot(2,2,i+1)
        P.imshow(test[i]['xspace'].real)
        tmp =test[i]['xspace'].real
        print "%i (min, max) = (%10.5e, %10.5e)" % (i, tmp.min(), tmp.max())
        P.colorbar()

    P.show()
