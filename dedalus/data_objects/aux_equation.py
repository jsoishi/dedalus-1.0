"""Auxiliary equation class. This holds simple ODEs that need to be
solved in addition to the fluid variables.

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

class AuxEquation(object):
    def __init__(self, RHS, kwargs, init_cond=0.):
        """a simple class to hold the current value and Right Hand
        Side (RHS) of an ODE.

        """
        self._RHS = RHS
        self.value = init_cond
        self.kwargs = kwargs

    def RHS(self, value):
        return self._RHS(value, **(self.kwargs))

    def __setitem__(self, data):
        self.value = data
