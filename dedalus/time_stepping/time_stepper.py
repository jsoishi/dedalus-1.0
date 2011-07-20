"""Low Storage Runga Kutta integrator, as implemented in the Pencil Code.

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

import time
import numpy as na
from dedalus.funcs import insert_ipython
class TimeStepper:
    """copied from pencil code

    """
    def __init__(self,RHS,itorder=3):
        self.RHS = RHS
        self.itorder = itorder
        if self.itorder==1:
            self.alpha_ts=np.array([ 0., 0., 0. ])
            self.beta_ts =np.array([ 1., 0., 0. ])
        elif self.itorder==2:
            self.alpha_ts=np.array([ 0., -1./2., 0. ])
            self.beta_ts=np.array([ 1./2.,  1.,  0. ])
        elif self.itorder==3:
            #alpha_ts=np.array([0., -2./3., -1.])
            #beta_ts=np.array([1./3., 1., 1./2.])
            #  use coefficients of Williamson (1980)
            self.alpha_ts=np.array([  0. ,  -5./9., -153./128. ])
            self.beta_ts=np.array([ 1./3., 15./16.,    8./15.  ])
        else:
            print "not implemented, stupid."
        
    def rk2n(self,t,f,dt):
        for itsub in xrange(self.itorder):
            llast = (itsub == self.itorder)

            if itsub == 0:
                lfirst = True
                df = 0.
                ds = 0.
            else:
                lfirst = False
                df = self.alpha_ts[itsub]*df
                ds = self.alpha_ts[itsub]*ds
            
            df += self.RHS(f)
            ds = ds + 1.
            dt_beta_ts = dt*self.beta_ts

            f += dt_beta_ts[itsub]*df
            t +=  dt_beta_ts[itsub]*ds
        return t,f
