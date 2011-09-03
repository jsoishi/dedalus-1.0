"""functions to enforce hermitian symmetry.

Authors: J. S. Oishi <jsoishi@gmail.com>
         K. J. Burns <kburns@berkeley.edu>
Affiliation: KIPAC/SLAC/Stanford
License:
  Copyright (C) 2011 J. S. Oishi, K. J. Burns.  All Rights Reserved.

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
from dedalus.utils.parallelism import MPI, comm

def enforce_hermitian(data, verbose=False):
    """
    Make array Hermitian-symmetric about origin (ifftn(data) will be real).  
    Note: this symmetry should propogate from real initial conditions.
    
    """
 
    n = data.shape
    if verbose: print n
    
    if data.size == 1: 
        if verbose: print data
        data.imag = 0
        return


    if comm:

        return
    
    # Flip about k-origin
    nonzero = [slice(1, None, 1)] * data.ndim 
    nonzero_flip = [slice(-1, 0, -1)] * data.ndim
    S = data[..., :n[-1] / 2]
    data[..., n[-1] / 2:][nonzero] = S[nonzero_flip].conj()
    
    # Enforce Hermitian symmetry in final Nyquist space
    if verbose: print 'Descending on a Nyquist:'
    enforce_hermitian(data[..., n[-1] / 2], verbose=verbose)
 
    # Enforce Hermitian symmetry in zero spaces
    zspace = [slice(None)] * data.ndim

    for i in xrange(data.ndim):
        if data.ndim == 1:
            zspace[i] = slice(0, 1)
        else:
            zspace[i] = 0
        if verbose: print 'Descending on a zero:'
        enforce_hermitian(data[zspace], verbose=verbose)
        zspace[i] = slice(None)
 
    
def zero_nyquist(data):
    """Zero out the Nyquist space in each dimension."""
    
    n = data.shape
    nyspace = [slice(None)] * data.ndim 
    
    # Pick out Nyquist space for each dimension and set to zero
    for i in xrange(data.ndim):
        nyspace[i] = n[i] / 2
        data[nyspace] = 0.
        nyspace[i] = slice(None)

