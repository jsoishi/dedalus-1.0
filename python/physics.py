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
from yt.funcs import insert_ipython
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
        self._trans = {0: 'x', 1: 'y', 2: 'z'}
        params = {'nu': 0.}
        
        self.aux_fields = []
        for i in range(self._naux):
             self.aux_fields.append(self._representation(self._shape))
        self._setup_parameters(params)
    
    def RHS(self, data):
        RHS = self.create_fields(data.time)
        vgradv = self.vgradv(data)
        pressure = self.pressure(data, vgradv)
        for f in self.fields:
            RHS[f] = -vgradv[f]['kspace'] + pressure[f]['kspace']
        return RHS
    
    def gradv(self, data):
        """compute stress tensor, du_j/dx_i

        """
        gradv = self.create_fields(data.time,fields=range(self._ndims**2))
        i = 0

        slices = self._ndims*(slice(None),)
        for f in self.fields:
            for dim in range(self._ndims):
                gradv[i] = data[f].deriv(self._trans[dim])
                i += 1
                
        return gradv

    def pressure(self, data, vgradv):
        pressure = self.create_fields(data.time)
        tmp = na.array([data[f].k[self._trans[i]] * vgradv[f]['kspace'] for i,f in enumerate(self.fields)])
        k2 = na.zeros(data['ux'].data.shape)
        for k in data['ux'].k.values():
            k2 += k**2

        k2[k2 == 0] = 1.
        for i,f in enumerate(self.fields):            
            pressure[f] = data[f].k[self._trans[i]] * tmp.sum(axis=0)/k2

        return pressure

    def vgradv(self, data):
        gradv = self.gradv(data)
        vgradv = self.create_fields(data.time)
        trans = {0: 'ux', 1: 'uy', 2: 'uz'}
        for i,f in enumerate(self.fields):
            b = [i * self._ndims + j for j in range(self._ndims)]
            tmp = na.array([data[trans[i]]['xspace'] * gradv[j]['xspace'] for i,j in enumerate(b)])
            vgradv[f] = tmp.sum(axis=0)
            vgradv[f]._curr_space = 'xspace'

        return vgradv

if __name__ == "__main__":
    import pylab as P
    from fourier_data import FourierData
    from init_cond import taylor_green
    a = Hydro((100,100),FourierData)
    data = a.create_fields(0.)
    taylor_green(data['ux'],data['uy'])
    vgv = a.vgradv(data)
    #test = a.pressure(data,vgv)
    test = a.RHS(data)

    for i,f in enumerate(a.fields):
        print test[f]._curr_space
        P.subplot(1,2,i+1)
        P.imshow(test[f]['xspace'].real)
        tmp =test[f]['xspace'].imag
        print "%s (min, max) = (%10.5e, %10.5e)" % (f, tmp.min(), tmp.max())
        P.colorbar()

    P.show()
