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
import fftw3

class Representation(object):
    """a representation of a field. it stores data and provides
    spatial derivatives.

    """

    def __init__(self, shape):
        pass


class FourierData(Representation):
    """Container for data that can be Fourier transformed. Includes a
    wrapped and specifiable method for performing the FFT. 

    Parallelization will go here?

    """
    def __init__(self, shape, dtype='complex128', method='fftw'):
        """
        
        Parameters
        ----------
        shape : tuple of ints
            the shape of the data.
        """
        self._shape = shape
        self._curr_space = 'kspace'
        self.dim = len(shape)
        self.data = na.zeros(self._shape,dtype=dtype)
        names = ['z','y','x']
        kslice = [slice(0,a) for a in self._shape]
        self.k = dict(zip(names[3-self.dim:],na.ogrid[kslice]))
        for key,v in self.k.iteritems():
            self.k[key] = v*2*na.pi/v.size

        self.set_fft(method)

    def __getitem__(self,space):
        """returns data in either xspace or kspace, transforming as necessary.

        """
        if space == self._curr_space:
            pass
        elif space == 'xspace':
            self.forward()
        elif space == 'kspace':
            self.reverse()
        else:
            raise KeyError("space must be either xspace or kspace")
        
        return self.data
        
        
    def set_fft(self, method):
        if method == 'fftw':
            self.fft = fftw3.Plan(self.data,direction='forward', flags=['measure'])
            self.ifft = fftw3.Plan(self.data,direction='backward', flags=['measure'])

    def forward(self):
        self.fft()
        self._curr_space = 'xspace'

    def backward(self):
        self.ifft()
        self._curr_space = 'kspace'

    def deriv(self,dim):
        """take a derivative along dim"""
        if self._curr_space == 'xspace':
            self.backward()
        return self.data * 1j*self.k[dim]

class FourierShearData(FourierData):
    """

    """
    def __init__(self, shape, S, **kwargs):    
        """Shearing box implementation.


        Parameters
        ----------
        S : float
            the *dimensional* shear rate in 1/t units

        """
        self.shear_rate = S
        FourierData.__init__(self, shape, **kwargs)

    def forward(self):
        pass

    def backward(self):
        pass



    
class SphericalHarmonicData(FourierData):
    """Pydro should eventually support spherical and cylindrical geometries.

    """
    pass
