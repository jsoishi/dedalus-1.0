"""Data object for fourier transformable fields

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

class FourierData(object):
    """Container for data that can be Fourier transformed. Includes a
    wrapped and specifiable method for performing the FFT. 

    """
    def __init__(self, shape):
        """
        
        Parameters
        ----------
        shape : tuple of ints
            the shape of the data.
        """
        self._shape = shape
        self.dim = len(shape)
        self.data = {'kspace': None,
                     'xspace': None}
        self._ktype = 'complex128'
        self._xtype = 'float64'

    def set_fft(self, method):
        pass

    def allocate_data(self, key):
        if key == 'kspace':
            self.data[key] = na.zeros(self.shape,dtype=self._ktype)
        else:
            self.data[key] = na.zeros(self.shape,dtype=self._xtype)
    
    def forward(self):
        pass

    def backward(self):
        pass
    
class SphericalHarmonicData(FourierData):
    """Pydro should eventually support spherical and cylindrical geometries.

    """
    pass
