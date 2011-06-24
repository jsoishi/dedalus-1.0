"""functions to enforce hermitian symmetry.

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

import numpy as na

def enforce_hermitian(data):
    """only works for 2D!!!

    """
    n = data.shape
    data[n[0]/2,n[1]/2+1:] = data[n[0]/2,n[1]/2-1:0:-1].conj()
    data[n[0]/2+1:,n[1]/2] = data[n[0]/2-1:0:-1,n[1]/2].conj()
    data[0:1,0:1].imag = 0
    data[n[0]/2:n[0]/2+1,n[1]/2:n[1]/2+1].imag=0
    data[n[0]/2:n[0]/2+1,0:1].imag=0
    data[0:1,n[1]/2:n[1]/2+1].imag=0

def zero_nyquist(data):
    """this zeros the nyquist modes: necessary for using even numbers
    of fourier modes.

    WORKS ONLY FOR 2D!!

    """
    n = data.shape
    data[n[0]/2,:] = 0
    data[:,n[1]/2] = 0
