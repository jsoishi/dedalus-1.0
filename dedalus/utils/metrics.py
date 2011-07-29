"""Cosmology metric functions

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

def friedmann(a, H0, Omega_r, Omega_m, Omega_l):
    """right hand side of the first Friedmann equation.

    """
    return H0*na.sqrt(Omega_r / a**4 + Omega_m / a**3 + Omega_l)

def a_friedmann(a, H0, Omega_r, Omega_m, Omega_l):
    """a times the rhs of the first Friedmann equation

    """
    return a * friedmann(a, H0, Omega_r, Omega_m, Omega_l)
