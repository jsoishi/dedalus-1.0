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

class Physics(object):
    """This is a base class for a physics object. It needs to provide
    a right hand side for a time integration scheme.

    """

    def __init__(self,shape):
        self._shape = shape

    def RHS(self):
        pass

class Hydro(Physics):
    
    def __init__(self,shape):
        Physics.__init__(self,shape)
        


