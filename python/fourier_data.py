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
import numpy.fft as fpack
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
    def __init__(self, shape, length=None, dtype='complex128', method='fftw'):
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
        # hard code for now
        self.L = dict(zip(names[3-self.dim:],2.*na.pi*na.ones(self.dim)))

        if not length:
            self.kny = na.pi*na.array(self._shape)/na.array(self.L.values())

        # setup wavenumbers
        kk = []
        for i,dim in enumerate(self.data.shape):
            sl = i*(1,)+(dim,)+(self.dim-i-1)*(1,)
            k = fpack.fftfreq(dim)*self.kny[i]
            k.resize(sl)
            kk.append(k)
        self.k = dict(zip(names[3-self.dim:],kk))

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


        # for now, only numpy's fft is supported
        self.fft = fpack.fft
        self.ifft = fpack.ifft

    def __getitem__(self,inputs):
        """returns data in either xspace or kspace, transforming as necessary.

        """
        space, time = inputs
        if space == self._curr_space:
            pass
        elif space == 'xspace':
            self.forward(time)
        elif space == 'kspace':
            self.reverse(time)
        else:
            raise KeyError("space must be either xspace or kspace")
        
        return self.data

    def forward(self,time):
        deltay = self.shear_rate*time 
        print deltay
        x = na.linspace(-na.pi,na.pi,self._shape[-1],endpoint=False)
        #x = na.linspace(0.,2*na.pi,self._shape[-1],endpoint=False)
        
        self.data = self.fft(self.data,axis=1)
        self.data *= na.exp(1j*self.k['y']*x*deltay)
        if self.dim == 3:
            self.data = self.fft(self.data,axis=2)
        self.data = self.fft(self.data,axis=0)
        self._curr_space = 'xspace'

    def backward(self,time):
        deltay = self.shear_rate*time 
        x = na.linspace(-na.pi,na.pi,endpoint=False)
        z_,y_,x_ = N.ogrid[0:self.data.shape[0],
                           0:self.data.shape[1],
                           0:self.data.shape[2]]

        if self.dim == 3:
            self.data = self.ifft(self.data,axis=2)
        self.data = self.ifft(self.data,axis=0)

        self.data *= na.exp(-1j*self.k['y']*(0.*z_ +0.*y_+x)*deltay)
        self.data = self.ifft(self.data,axis=1)
        self._curr_space = 'kspace'

    
class SphericalHarmonicData(FourierData):
    """Pydro should eventually support spherical and cylindrical geometries.

    """
    pass
